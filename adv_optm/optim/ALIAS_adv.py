import torch

from typing import Optional

from ..util import param_update
from ..util.OrthoGrad import _orthogonalize_gradient
from ..util.factorization_util import _get_effective_shape, _reconstruct_state, _factorize_state, _pack_bools, _unpack_bools
from ..util.update_util import _get_l1_adaptive_lr
from ..util.scaled_optm import scale_update, is_spectral, init_spectral_norm
from ..util.centered_decay import _init_anchor
from ..util.alias_util import _update_alias_state, _get_alias_lr

class ALIAS_adv(torch.optim.Optimizer):
    """
    Implements an advanced SignSGD algorithm based on ALIAS Parameter-Free tracking.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate (default: 1e-4).
        momentum (float, optional): coefficients for computing
            running average of the gradients (default: 0.9).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.0).
        cautious_wd (bool): Enables Cautious Weight Decay. If True, weight decay is
            applied only to parameter coordinates where the sign of the parameter
            and the sign of the optimizer update align (default: False).
        vector_reshape (bool, optional): whether to reshape 1D vectors into 2D
            matrices to apply low-rank compression (default: True).
        stochastic_rounding (bool, optional): whether to use stochastic
            rounding for BF16 parameter updates (default: True).
        packed_sign (bool): Always use 1-bit compress for uint8 sign to 
            save memory (default: True).
        approx_alias (bool): Controls the L_inf norm approximation logic for ALIAS.
            If True: Uses memory-efficient scalar tracking (max/min) to approximate
            the smoothness term.
            If False: Stores the full previous gradient to calculate exact
            ||grad_t - grad_{t-1}||_inf. (Uses more VRAM). (default: False)
        scaled_optm (bool): Applies scaled updates based on parameter norms (default: False).
        orthogonal_gradient (bool): whether to orthogonalize the gradient (default: False).
        Simplified_AdEMAMix (bool): whether to use the Simplified AdEMAMix update rule.
            This changes the EMA to accumulator and the update numerator to `alpha_grad * grad + mt`, which can be
            more responsive, especially for small batch sizes. (default: False)
        alpha_grad (float): Mixing coefficient for the Simplified AdEMAMix update rule
            (only used when `Simplified_AdEMAMix` is `True`). Controls the weight of the
            current gradient. For small batch sizes, use high values (e.g., 10-100) to be
            more responsive. For large batch sizes, use low values (e.g., 0-1) for
            stability. (default: 100.0)
        freeze_on_flip (bool): Projected SignGD One-hit freeze. Masks updates for
            coordinates where the gradient sign flips compared to the previous step. (default: False)
        l1_adaptive (bool): Scales the update step magnitude dynamically
            by the mean L1 norm of the momentum/gradient to handle gradient heterogeneity.(default: False)
        centered_wd (float): Centered Weight Decay coefficient. Instead of decaying weights
            toward zero, they are decayed toward their initial values (anchors). This
            can be used together with standard weight decay. (default: 0.0)
        centered_wd_mode (str): The quantization format used to store the anchor
            weights to save VRAM. Options include:
            'full': Stores anchors in the original parameter's precision.
            'float8': Uses torch.float8_e4m3fn for a balance of precision and memory.
            'int8': Uses 8-bit block-wise quantization (block size 128).
            'int4': Uses 4-bit block-wise quantization (block size 32).
        nnmf_factor (bool): whether to use the factorization or use the
            uncompressed optimizer. (default: True)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        momentum: float = 0.95,
        # Decoupled/cautious weight decay
        weight_decay: float = 0.0,
        cautious_wd: bool = False,
        # Stochastic Rounding for BF16
        stochastic_rounding: bool = True,
        packed_sign: bool = True, 
        approx_alias: bool = False, 
        # OrthoGrad
        orthogonal_gradient: bool = False,
        # Simplified_AdEMAMix
        alpha_grad: float = 1.0,
        Simplified_AdEMAMix: bool = False,
        # Projected and adaptive sign
        freeze_on_flip: bool = False,
        l1_adaptive: bool = False,
        # Centered WD
        centered_wd: float = 0.0,
        centered_wd_mode: str = 'float8',
        # Scaled Optimizer
        scaled_optm: bool = False,
        # SMMF factorization
        nnmf_factor: bool = False,
        vector_reshape: bool = False,
        # torch.compile
        compiled_optimizer: bool = False,
    ):
        if not lr > 0.0:
            raise ValueError(f"Learning rate must be > 0.0, but got {lr}")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum should be in [0.0, 1.0], but got {momentum}")
        if not weight_decay >= 0.0:
            raise ValueError(f"Weight decay must be >= 0.0, but got {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            cautious_wd=cautious_wd,
            packed_sign=packed_sign,
            approx_alias=approx_alias,
            orthogonal_gradient=orthogonal_gradient,
            alpha_grad=alpha_grad,
            Simplified_AdEMAMix=Simplified_AdEMAMix,
            scaled_optm= scaled_optm,
            freeze_on_flip=freeze_on_flip,
            l1_adaptive=l1_adaptive,
            centered_wd= centered_wd,
            centered_wd_mode= centered_wd_mode,
            nnmf_factor=nnmf_factor,
            vector_reshape=vector_reshape,
        )
        self.stochastic_rounding = stochastic_rounding
        self._init_lr = lr
        super().__init__(params, defaults)

        if self.stochastic_rounding:
            # For deterministic stochastic rounding, we need to seed the generator
            # for each device used by the parameters.
            devices = {p.device for group in self.param_groups for p in group['params'] if p.dtype == torch.bfloat16}
            for device in devices:
                param_update.set_seed(device)

        # Initialize compiled function
        self._compiled_step_parameter = None
        if compiled_optimizer:
            self.compile(fullgraph=True)

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Overrides default load_state_dict to implement a workaround for PyTorch's
        automatic dtype casting. It ensures factorized states remain float32 for
        stability, preserves integer/float8 quantized anchor states, and forces
        standard states onto the parameter's current dtype/device.
        """
        super().load_state_dict(state_dict)
        param_update.post_process_loaded_state(self)

    @property
    def supports_fused_back_pass(self) -> bool:
        return True

    @property
    def supports_memory_efficient_fp16(self) -> bool:
        return True

    @property
    def supports_flat_params(self) -> bool:
        return False

    @torch.no_grad()
    def step_parameter(self, p: torch.Tensor, group: dict, i: Optional[int] = None):
        """Performs a single optimization step on a single parameter."""
        if p.grad is None:
            return

        grad = p.grad
        state = self.state[p]

        lr = group["lr"]

        if group.get('compiled_optimizer', False):
            lr = torch.as_tensor(lr, dtype=torch.float64)

        # State Initialization
        if group["momentum"] > 0 and len(state) == 0:
            state['factored'] = (
                group['nnmf_factor'] and
                not (len(p.shape) == 1 and not group['vector_reshape'])
            )

            dtype = torch.float32 if state['factored'] else p.dtype

            if state['factored']:
                state['effective_shape'] = _get_effective_shape(p.numel())
                d1, d2 = state['effective_shape']
                state['mu_m_nmf'] = torch.zeros(d1, device=p.device, dtype=dtype)
                state['mv_m_nmf'] = torch.zeros(d2, device=p.device, dtype=dtype)
                if group.get("freeze_on_flip", True):
                    state['sign'] = _pack_bools(grad.view(d1, d2) > 0)
                else:
                    packed_d2 = (d2 + 7) // 8
                    state['sign'] = torch.zeros((d1, packed_d2), dtype=torch.uint8, device=p.device)
            else:
                state['exp_avg'] = torch.zeros_like(p, device=p.device, dtype=dtype)
                if group.get("freeze_on_flip", True):
                    state['prev_sign'] = (grad > 0).to(torch.uint8)

            if group.get('scaled_optm', False) and is_spectral(p):
                init_spectral_norm(group, state, p)

            state["step"] = 0

            # ALIAS-specific states
            state['alias_d'] = torch.full((1,), group['lr'], device=p.device, dtype=torch.float32)
            state['alias_d_raw'] = torch.full((1,), group['lr'], device=p.device, dtype=torch.float32)
            state['alias_eta'] = torch.zeros(1, device=p.device, dtype=torch.float32)
            state['alias_prev_lr'] = torch.zeros(1, device=p.device, dtype=torch.float32)

            # State allocation depends on approx_alias setting
            if group['approx_alias']:
                state['alias_prev_grad_max'] = torch.zeros(1, device=p.device, dtype=torch.float32)
                state['alias_prev_grad_min'] = torch.zeros(1, device=p.device, dtype=torch.float32)
            else:
                # Store full gradient if exact computation is required
                # Note: This increases VRAM usage significantly (+1 state)
                state['alias_prev_grad'] = torch.zeros_like(grad, device=p.device, dtype=grad.dtype)

            # Pack memory initialization based on boolean shape arithmetic
            if group['packed_sign']:
                n_elems = p.numel()
                packed_m = (n_elems + 7) // 8
                state['alias_prev_sign'] = torch.zeros(packed_m, dtype=torch.uint8, device=p.device)
            else:
                state['alias_prev_sign'] = torch.zeros_like(p, dtype=torch.uint8, memory_format=torch.preserve_format)


            _init_anchor(p, state, group)

        random_int_tensor = None

        if group.get('compiled_optimizer', False):
            if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                # Pre-generate random tensor for stochastic rounding if needed.
                random_int_tensor = param_update._get_random_int_for_sr(p)
            step_param_fn = self._compiled_step_parameter
        else:
            step_param_fn = self._step_parameter

        step_param_fn(p, grad, state, group, lr, random_int_tensor)

        state['step'] += 1


    def _step_parameter(self, p, grad, state, group, lr, random_int_tensor):
        if grad.dtype != torch.float32 and state.get('factored', False):
            grad = grad.float()

        if group["orthogonal_gradient"]:
            grad = _orthogonalize_gradient(p, grad)

        momentum = group["momentum"]
        Simplified_AdEMAMix = group["Simplified_AdEMAMix"]
        alpha_grad = group["alpha_grad"]
        freeze_on_flip = group.get("freeze_on_flip", False)
        if not Simplified_AdEMAMix:
            alpha_grad = 0
        if momentum == 0:
            alpha_grad = 1

        step = state['step']

        # ALIAS Parameter-Free calculation
        alias_d = state['alias_d']
        alias_d_raw = state['alias_d_raw']
        alias_eta = state['alias_eta']

        # Select states based on approximation mode
        if group['approx_alias']:
            prev_grad_max = state['alias_prev_grad_max']
            prev_grad_min = state['alias_prev_grad_min']
            prev_grad = None
        else:
            prev_grad_max = None
            prev_grad_min = None
            prev_grad = state['alias_prev_grad']

        prev_sign = state.get('alias_prev_sign') if step > 1 else None
        prev_lr = state.get('alias_prev_lr') if step > 1 else torch.tensor(group['lr'], device=p.device, dtype=torch.float32)

        _update_alias_state(
            grad=grad,
            alias_d=alias_d,
            alias_d_raw=alias_d_raw,
            alias_eta=alias_eta,
            prev_grad_max=prev_grad_max,
            prev_grad_min=prev_grad_min,
            prev_grad=prev_grad,
            prev_sign=prev_sign,
            prev_lr=prev_lr,
            approx_alias=group['approx_alias'],
            packed_sign=group['packed_sign'],
            original_shape=p.shape
        )

        alias_lr = _get_alias_lr(alias_d, alias_eta, group['lr'])

        if state.get('factored'):
            # Factored Path
            d1, d2 = state['effective_shape']
            grad_reshaped = grad.view(d1, d2)

            if freeze_on_flip:
                prev_sign_packed = state['sign'].clone()

            if momentum > 0:
                # Reconstruct momentum m_{t-1}
                exp_avg = _reconstruct_state((state['mu_m_nmf'], state['mv_m_nmf'], state['sign'], d2), signed=True)
                exp_avg.mul_(momentum).add_(grad_reshaped)

                if Simplified_AdEMAMix:
                    raw_update = exp_avg + (grad_reshaped * alpha_grad)
                else:
                    raw_update = exp_avg.clone()

                # Compress new momentum m_t and store factors
                state['mu_m_nmf'], state['mv_m_nmf'], state['sign'] = _factorize_state(exp_avg, signed=True)
            else:
                raw_update = grad_reshaped.clone()
                if freeze_on_flip:
                    state['sign'] = _pack_bools(raw_update > 0)

            l1_mean = _get_l1_adaptive_lr(p, raw_update, state, group, 1)

            update = raw_update.sign_()
            update = update.view(p.shape)

            if freeze_on_flip:
                # Fast binary diff (XOR) from momentum sign directly
                flipped_packed = prev_sign_packed ^ state['sign']
                flipped_mask = _unpack_bools(flipped_packed, original_m=d2).view_as(update)
                update = torch.where(flipped_mask, 0.0, update)
                del prev_sign_packed, flipped_packed, flipped_mask

        else:
            # Fallback to standard SignSGD logic
            if momentum > 0:
                exp_avg = state["exp_avg"]
                exp_avg.mul_(momentum).add_(grad)

                if Simplified_AdEMAMix:
                    raw_update = exp_avg + (grad * alpha_grad)
                else:
                    raw_update = exp_avg.clone()
            else:
                raw_update = grad.clone()

            l1_mean = _get_l1_adaptive_lr(p, raw_update, state, group, 1)

            # SignSGD Logic
            update = raw_update.sign_()

            if freeze_on_flip:
                current_sign = (raw_update > 0).to(torch.uint8)
                update = torch.where(current_sign == state['prev_sign'], update, 0.0)
                state['prev_sign'] = current_sign

        if l1_mean is not None:
            update.mul_(l1_mean)

        # Store states for the *next* ALIAS calculation
        if group['packed_sign']:
            flat_update = update.view(1, -1)
            packed = _pack_bools(flat_update > 0)
            state['alias_prev_sign'].copy_(packed.view(-1))
        else:
            state['alias_prev_sign'].copy_((update > 0).to(torch.uint8))

        # Dynamic Learning Rate Selection
        lr_t = torch.tensor(group['lr'], dtype=torch.float32)
        lr = torch.where(alias_eta > 0, alias_lr, lr_t).squeeze()
        state['alias_prev_lr'].copy_(lr)

        # Apply scaling and weight decay
        if group.get('scaled_optm', False):
            update = scale_update(p, update, lr, vector_state=state.get('spectral_v'))
        else:
            update.mul_(lr)

        param_update.apply_parameter_update(self, p, group, update, lr, random_int_tensor=random_int_tensor)

    def compile(self, *args, **kwargs):
        self._compiled_step_parameter = torch.compile(self._step_parameter, *args, **kwargs)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is not None:
                    self.step_parameter(p, group, i)

        return loss
