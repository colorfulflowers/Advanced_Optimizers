import torch
import torch.distributed as dist
from .factorization_util import _pack_bools, _unpack_bools

class AliasHelper:
    """
    Manages ALIAS step size adaptation at different granularities.
    """
    def __init__(self, mode='per-param', d0=1e-5):
        """
        Args:
            mode: 'per-param' (individual LR), 'per-shape' (shared LR for identical shapes), or 'global' (one LR).
            d0: Initial distance approximation.
        """
        self.mode = mode
        self.d0 = d0
        self.buckets = {}

    def get_bucket_key(self, p):
        if self.mode == 'per-param':
            return p
        elif self.mode == 'per-shape':
            return tuple(p.shape)
        elif self.mode == 'global':
            return 'global'
        else:
            raise ValueError(f"Unknown ALIAS mode: {self.mode}")

    def init_state(self, p, state, group, grad):
        key = self.get_bucket_key(p)

        # Initialize the shared bucket if it doesn't exist
        if key not in self.buckets:
            self.buckets[key] = {
                'alias_d': torch.tensor(self.d0, device=p.device, dtype=torch.float32),
                'alias_tilde_d': torch.tensor(0.0, device=p.device, dtype=torch.float32),
                'alias_eta': torch.tensor(0.0, device=p.device, dtype=torch.float32),
                # Track max numerator and sum denominator separately
                'max_diff_acc': torch.tensor(0.0, device=p.device, dtype=torch.float32),
                'step_l1_acc': torch.tensor(0.0, device=p.device, dtype=torch.float32),
                'tilde_d_inc_acc': torch.tensor(0.0, device=p.device, dtype=torch.float32),
                'alias_lr': torch.tensor(self.d0, device=p.device, dtype=torch.float32),
                # Track the total number of parameters in this bucket
                'd_total': torch.tensor(0.0, device=p.device, dtype=torch.float32),
                'd_total_synced': False,
            }

        # Add parameter count to the bucket's total (only once per parameter)
        if not state.get('alias_d_counted', False):
            self.buckets[key]['d_total'] += p.numel()
            state['alias_d_counted'] = True

        # Initialize per-tensor tracking variables
        g_max = grad.max().to(torch.float32)
        g_min = grad.min().to(torch.float32)

        state['alias_prev_grad_max'] = g_max.clone()
        state['alias_prev_grad_min'] = g_min.clone()
        state['alias_prev_update_l1'] = torch.tensor(0.0, device=p.device, dtype=torch.float32)
        state['alias_bucket_key'] = key

        if state.get('factored'):
            d1, d2 = state['effective_shape']
            packed_d2 = (d2 + 7) // 8
            state['alias_prev_sign_packed'] = torch.zeros((d1, packed_d2), dtype=torch.uint8, device=p.device)
        else:
            state['alias_prev_sign'] = torch.zeros_like(grad, dtype=torch.uint8)

    def accumulate(self, p, grad, state, current_lr):
        """Accumulate the eta and tilde_d increments into the bucket."""
        key = state['alias_bucket_key']
        bucket = self.buckets[key]
        
        g_max = grad.max().to(torch.float32)
        g_min = grad.min().to(torch.float32)
        d = p.numel() 
        prev_l1 = state['alias_prev_update_l1']

        # Fetch the LR that was used in the previous step
        prev_used_lr = state.get('alias_used_lr', current_lr)

        # Calculate exact sum limits
        actual_step_l1 = torch.where(prev_l1 > 0, prev_used_lr * d, torch.tensor(0.0))

        # Memory-efficient L_inf approximation formula (Matches Section 3.3)
        approx_max_diff = torch.maximum((g_max - state['alias_prev_grad_min']).abs(),
                                        (state['alias_prev_grad_max'] - g_min).abs())

        if state.get('factored'):
            d1, d2 = state['effective_shape']
            prev_sign_float = _unpack_bools(state['alias_prev_sign_packed'], original_m=d2).to(grad.dtype) * 2.0 - 1.0
            dot_product = torch.sum(grad.view(d1, d2) * prev_sign_float).to(torch.float32)
        else:
            prev_sign_float = state['alias_prev_sign'].to(grad.dtype) * 2.0 - 1.0
            dot_product = torch.sum(grad * prev_sign_float).to(torch.float32)
            
        tilde_d_inc = torch.where(actual_step_l1 > 0, prev_used_lr * dot_product, torch.zeros_like(dot_product))

        # Accumulate global terms independently (MAX for L_inf, SUM for L_1)
        bucket['max_diff_acc'].copy_(torch.maximum(bucket['max_diff_acc'], approx_max_diff))
        bucket['step_l1_acc'].add_(actual_step_l1)
        bucket['tilde_d_inc_acc'].add_(tilde_d_inc)
        
        # Update tensor-specific history for the next step
        state['alias_prev_grad_max'].copy_(g_max)
        state['alias_prev_grad_min'].copy_(g_min)
        state['alias_used_lr'] = current_lr

    def get_lr(self, state, base_lr):
        key = state['alias_bucket_key']
        bucket = self.buckets[key]

        alias_lr = bucket['alias_lr']
        if not isinstance(base_lr, torch.Tensor):
            base_lr_t = torch.tensor(base_lr, dtype=torch.float32, device=alias_lr.device)
        else:
            base_lr_t = base_lr.to(torch.float32)

        return torch.where(bucket['alias_eta'] > 0, alias_lr, base_lr_t)

    def update_post_step(self, state, raw_update, scale_factor: float=1):
        """Store the current step's signs and l1 norm for the next step."""
        if state.get('factored'):
            state['alias_prev_sign_packed'].copy_(_pack_bools(raw_update > 0))
        else:
            state['alias_prev_sign'].copy_((raw_update > 0).to(torch.uint8))
        state['alias_prev_update_l1'].copy_(raw_update.abs().mean().to(torch.float32)/scale_factor)

    def calculate_lr(self):
        """Called at the end of optimizer.step() to compute the new LR for all buckets."""
        for key, bucket in self.buckets.items():
            if dist.is_available() and dist.is_initialized():
                # Sync d_total safely (only once per run to avoid doubling every step)
                if not bucket['d_total_synced']:
                    dist.all_reduce(bucket['d_total'], op=dist.ReduceOp.SUM)
                    bucket['d_total_synced'] = True

                # Sync across nodes (MAX for numerator, SUM for denominators)
                dist.all_reduce(bucket['max_diff_acc'], op=dist.ReduceOp.MAX)
                dist_tensor = torch.stack([bucket['step_l1_acc'], bucket['tilde_d_inc_acc']])
                dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
                bucket['step_l1_acc'].copy_(dist_tensor[0])
                bucket['tilde_d_inc_acc'].copy_(dist_tensor[1])

            # Normalization: 1/d scale
            # This transforms the massive sums O(d) into exact weighted means O(1).
            d_tot = bucket['d_total']
            step_l1_mean = bucket['step_l1_acc'] / d_tot
            tilde_d_inc_mean = bucket['tilde_d_inc_acc'] / d_tot

            eta_inc = torch.where(step_l1_mean > 0, 
                                  bucket['max_diff_acc'] / step_l1_mean, 
                                  torch.zeros_like(bucket['max_diff_acc']))

            bucket['alias_eta'].add_(eta_inc)
            bucket['alias_tilde_d'].add_(tilde_d_inc_mean)

            if (bucket['alias_eta'] > 0).all():
                bucket['alias_d'].copy_(torch.maximum(bucket['alias_d'], bucket['alias_tilde_d']))

            # Option I states gamma^t = sqrt(d^t) / sqrt(eta^t)
            alias_lr = torch.sqrt(torch.relu(bucket['alias_d'])) / torch.sqrt(bucket['alias_eta'])
            bucket['alias_lr'].copy_(torch.where(bucket['alias_eta'] > 0, alias_lr, bucket['alias_lr']))

            # Reset accumulators for the next step
            bucket['max_diff_acc'].zero_()
            bucket['step_l1_acc'].zero_()
            bucket['tilde_d_inc_acc'].zero_()