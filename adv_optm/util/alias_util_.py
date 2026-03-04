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
                'eta_inc_acc': torch.tensor(0.0, device=p.device, dtype=torch.float32),
                'tilde_d_inc_acc': torch.tensor(0.0, device=p.device, dtype=torch.float32),
                'alias_lr': torch.tensor(self.d0, device=p.device, dtype=torch.float32),
            }

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

    def accumulate(self, p, grad, state, prev_lr):
        """Accumulate the eta and tilde_d increments into the bucket."""
        key = state['alias_bucket_key']
        bucket = self.buckets[key]
        
        g_max = grad.max().to(torch.float32)
        g_min = grad.min().to(torch.float32)
        
        prev_l1 = state['alias_prev_update_l1']
        safe_prev_l1 = torch.where(prev_l1 > 0, prev_l1, torch.ones_like(prev_l1))
        
        approx_max_diff = torch.maximum((g_max - state['alias_prev_grad_min']).abs(),
                                        (state['alias_prev_grad_max'] - g_min).abs())
        
        eta_inc = torch.where(prev_l1 > 0, approx_max_diff / safe_prev_l1, torch.zeros_like(prev_l1))
        
        if state.get('factored'):
            d1, d2 = state['effective_shape']
            prev_sign_float = _unpack_bools(state['alias_prev_sign_packed'], original_m=d2).to(grad.dtype) * 2.0 - 1.0
            dot_product = torch.mean(grad.view(d1, d2) * prev_sign_float).to(torch.float32)
        else:
            prev_sign_float = state['alias_prev_sign'].to(grad.dtype) * 2.0 - 1.0
            dot_product = torch.mean(grad * prev_sign_float).to(torch.float32)
            
        tilde_d_inc = prev_lr * dot_product
        tilde_d_inc = torch.where(prev_l1 > 0, tilde_d_inc, torch.zeros_like(tilde_d_inc))
        
        # Accumulate to the shared bucket
        bucket['eta_inc_acc'].add_(eta_inc)
        bucket['tilde_d_inc_acc'].add_(tilde_d_inc)
        
        # Update tensor-specific history
        state['alias_prev_grad_max'].copy_(g_max)
        state['alias_prev_grad_min'].copy_(g_min)

    def get_lr(self, state, base_lr):
        """Fetch the shared LR for this parameter's bucket."""
        key = state['alias_bucket_key']
        bucket = self.buckets[key]

        alias_lr = bucket['alias_lr']
        if not isinstance(base_lr, torch.Tensor):
            base_lr_t = torch.tensor(base_lr, dtype=torch.float32, device=alias_lr.device)
        else:
            base_lr_t = base_lr.to(torch.float32)

        return torch.where(bucket['alias_eta'] > 0, alias_lr, base_lr_t)

    def update_post_step(self, state, raw_update):
        """Store the current step's signs and l1 norm for the next step."""
        if state.get('factored'):
            state['alias_prev_sign_packed'].copy_(_pack_bools(raw_update > 0))
        else:
            state['alias_prev_sign'].copy_((raw_update > 0).to(torch.uint8))
        state['alias_prev_update_l1'].copy_(raw_update.abs().mean().to(torch.float32))

    def calculate_lr(self):
        """Called at the end of optimizer.step() to compute the new LR for all buckets."""
        for key, bucket in self.buckets.items():
            if dist.is_available() and dist.is_initialized():
                dist_tensor = torch.stack([bucket['eta_inc_acc'], bucket['tilde_d_inc_acc']])
                dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
                eta_inc_sum = dist_tensor[0]
                tilde_d_inc_sum = dist_tensor[1]
            else:
                eta_inc_sum = bucket['eta_inc_acc']
                tilde_d_inc_sum = bucket['tilde_d_inc_acc']

            bucket['alias_eta'].add_(eta_inc_sum)
            bucket['alias_tilde_d'].add_(tilde_d_inc_sum)

            if (bucket['alias_eta'] > 0).all():
                current_d_estimate = bucket['alias_tilde_d'] / torch.sqrt(bucket['alias_eta'])
                bucket['alias_d'].copy_(torch.maximum(bucket['alias_d'], current_d_estimate))
            alias_lr = bucket['alias_d'] / torch.sqrt(bucket['alias_eta'])
            bucket['alias_lr'].copy_(torch.where(bucket['alias_eta'] > 0, alias_lr, bucket['alias_lr']))

            # Reset accumulators for the next step
            bucket['eta_inc_acc'].zero_()
            bucket['tilde_d_inc_acc'].zero_()