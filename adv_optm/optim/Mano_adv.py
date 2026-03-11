import torch
import math
from typing import Optional, Callable

from ..util import param_update
from ..util.Muon_util import _is_suitable_for_muon
from ..util.Mano_util import mano_orthogonalization, mano_rms_rescaling
from ..util.factorization_util import _get_effective_shape, _factorize_state, _reconstruct_state
from ..util.Kourkoutas import KourkoutasHelper
from ..util import Muon_AuxAdam
from ..util.scaled_optm import spectral_normalization, is_spectral, init_spectral_norm
from ..util.centered_decay import _init_anchor

class Mano_adv(torch.optim.Optimizer):
    """
    Implements the Mano (Manifold Normalized Optimizer) algorithm with advanced features.

    Mano performs optimization by projecting the momentum onto the tangent space of the
    parameter manifold (specifically the rotational Oblique manifold) and normalizing it.
    This replaces the Newton-Schulz iterations found in Muon with a cheaper, geometrically
    aware projection.

    Reference: "Mano: Restriking Manifold Optimization for LLM Training"

    When `use_mano` is False for a group, it falls back to
    an internal advanced AdamW implementation (AuxAdam).

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): learning rate (default: 1e-3).
        beta1 (float): momentum factor (default: 0.95).
        weight_decay (float): weight decay (L2 penalty) (default: 0.1).
        cautious_wd (bool): Enables Cautious Weight Decay. (default: False).
        nesterov (bool): enables Nesterov momentum (default: False).
        stochastic_rounding (bool): whether to use stochastic rounding for BF16 parameter updates (default: True).
        vector_reshape (bool): whether to reshape 1D vectors into 2D matrices to apply low-rank compression (default: True).
        nnmf_factor (bool): whether to use SMMF factorization for the momentum state (default: False).
        use_mano (bool | None): whether to use Mano or AuxAdamW. MUST be provided either here or via `optim_type`.
        Simplified_AdEMAMix (bool): whether to use the Simplified AdEMAMix update rule for momentum. (default: False).
        alpha_grad (float): Mixing coefficient for Simplified AdEMAMix. (default: 100.0).
        rotate_method (str): Method to choose the manifold rotation dimension ('fixed', 'auto_ft', 'auto_adjusted_ft').
            1. 'fixed': Rotates on the largest dim.
            2. 'auto_ft': original Mano rotation on Arbitrary dim.
            3. 'auto_adjusted_ft': Adjust the dimension rotation frequency according to their sizes, i.e., if one axis is X times larger, choose this axis X times more frequently in Mano.
            (default: 'auto_ft').
        centered_wd (float): Centered Weight Decay coefficient. Instead of decaying weights
            toward zero, they are decayed toward their initial values (anchors). This
            can be used together with standard weight decay. (default: 0.0)
        centered_wd_mode (str): The quantization format used to store the anchor
            weights to save VRAM. Options include:
            'full': Stores anchors in the original parameter's precision.
            'float8': Uses torch.float8_e4m3fn for a balance of precision and memory.
            'int8': Uses 8-bit block-wise quantization (block size 128).
            'int4': Uses 4-bit block-wise quantization (block size 32).
        compiled_optimizer (bool): Whether to compile the step function. (default: False).

        --- Auxiliary AdamW_adv Parameters (used for 'adam' groups) ---
        adam_betas (tuple[float, float]): Betas for the AdamW optimizer part.
        adam_eps (float): Epsilon for the AdamW optimizer part.
        adam_weight_decay (float): Weight decay for the AdamW optimizer part.
        adam_use_bias_correction (bool): Bias correction for AdamW.
        adam_use_atan2 (bool): Atan2 update rule for AdamW.
        adam_cautious_mask (bool): Cautious masking for AdamW.
        adam_grams_moment (bool): Grams-style updates for AdamW.
        adam_orthogonal_gradient (bool): OrthoGrad for AdamW.
        adam_use_AdEMAMix (bool): AdEMAMix for AdamW.
        adam_beta3_ema (float): Beta3 for AdEMAMix.
        adam_alpha (float): Alpha for AdEMAMix.
        adam_kourkoutas_beta (bool): Kourkoutas-β for AdamW.
        adam_nnmf_factor (bool): 1-bit factored for AdamW.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta1: float = 0.95,
        weight_decay: float = 0.0,
        cautious_wd: bool = False,
        # Stochastic Rounding for BF16
        stochastic_rounding: bool = True,
        # SMMF factorization
        nnmf_factor: bool = False,
        vector_reshape: bool = False,
        # Boolean to spilt param
        use_mano: bool | None = True,
        # Momentum Variants
        nesterov: bool = True,
        Simplified_AdEMAMix: bool = False,
        alpha_grad: float = 100.0,
        # Manifold Rotation Dimension
        rotate_method: str = 'auto_adjusted_ft',
        # Scaled Optimizer
        scaled_optm: bool = False,
        # Centered WD
        centered_wd: float = 0.0,
        centered_wd_mode: str = 'float8',
        # torch.compile
        compiled_optimizer: bool = False,
        # --- AdamW_adv specific parameters ---
        adam_betas: tuple[float, float] = (0.9, 0.99),
        adam_eps: float = 1e-8,
        adam_weight_decay: float = 0.0,
        adam_use_bias_correction: bool = True,
        adam_use_atan2: bool = False,
        adam_cautious_mask: bool = False,
        adam_grams_moment: bool = False,
        adam_orthogonal_gradient: bool = False,
        adam_use_AdEMAMix: bool = False,
        adam_beta3_ema: float = 0.9999,
        adam_alpha: float = 5.0,
        adam_kourkoutas_beta: bool = False,
        adam_beta2_min: float = 0.9,
        adam_ema_alpha: float = 0.95,
        adam_tiny_spike: float = 1e-9,
        adam_k_warmup_steps: int = 0,
        adam_nnmf_factor: bool = False,
    ):
        if not (lr >= 0.0):
            raise ValueError(f"Learning-rate should be >= 0.0. Got {lr}")
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"beta1 should be in [0.0, 1.0). Got {beta1}")
        if not (weight_decay >= 0.0):
            raise ValueError(f"Weight-decay should be >= 0.0. Got {weight_decay}")
        if Simplified_AdEMAMix and nesterov:
            print("Warning: nesterov is incompatible with Simplified_AdEMAMix, Disabling nesterov.")
            nesterov = False

        defaults = {
            "lr": lr, "beta1": beta1, "weight_decay": weight_decay, "cautious_wd": cautious_wd,
            "nnmf_factor": nnmf_factor, "vector_reshape": vector_reshape,
            "nesterov": nesterov,
            "Simplified_AdEMAMix": Simplified_AdEMAMix, "alpha_grad": alpha_grad,
            "rotate_method": rotate_method,
            "scaled_optm": scaled_optm,
            "centered_wd": centered_wd, "centered_wd_mode": centered_wd_mode,
            'compiled_optimizer': compiled_optimizer,
            "use_mano": use_mano,
            # AdamW_adv defaults
            "adam_betas": adam_betas, "adam_eps": adam_eps, "adam_weight_decay": adam_weight_decay,
            "adam_use_bias_correction": adam_use_bias_correction, "adam_use_atan2": adam_use_atan2,
            "adam_cautious_mask": adam_cautious_mask, "adam_grams_moment": adam_grams_moment,
            "adam_orthogonal_gradient": adam_orthogonal_gradient,
            "adam_use_AdEMAMix": adam_use_AdEMAMix, "adam_beta3_ema": adam_beta3_ema, "adam_alpha": adam_alpha,
            "adam_kourkoutas_beta": adam_kourkoutas_beta, "adam_beta2_min": adam_beta2_min,
            "adam_ema_alpha": adam_ema_alpha, "adam_tiny_spike": adam_tiny_spike,
            "adam_k_warmup_steps": adam_k_warmup_steps,
            "adam_nnmf_factor": adam_nnmf_factor,
        }
        self.stochastic_rounding = stochastic_rounding
        self.compiled_optimizer = compiled_optimizer
        self._init_lr = lr

        super().__init__(params, defaults)

        # Validate that every group has a determined optimizer type
        for i, group in enumerate(self.param_groups):
            if group.get('use_mano') is None and group.get('optim_type') is None:
                # Automatic shape-based detection if not explicit
                has_mano_shape = False
                for p in group['params']:
                    # Use Muon's suitability check as Mano also applies to 2D matrices
                    has_mano_shape = _is_suitable_for_muon(p)
                    if has_mano_shape:
                        group['use_mano'] = True
                    else:
                        group['use_mano'] = False

            if group.get('use_mano') is None: # Fallback
                 group['use_mano'] = group.get('optim_type') == 'mano'

            for p in group['params']:
                self.__init_state(p, group)

            # Initialize step for rotation
            group['steps'] = 0

        self.kourkoutas_helper = None
        if any(group.get('adam_kourkoutas_beta', False) for group in self.param_groups):
            self.kourkoutas_helper = KourkoutasHelper(self)

        if self.stochastic_rounding:
            devices = {p.device for group in self.param_groups for p in group['params'] if p.dtype == torch.bfloat16}
            for device in devices:
                param_update.set_seed(device)

        # Initialize compiled function
        self._compiled_mano_step_parameter = None
        self._compiled_adam_step_parameter = None
        if compiled_optimizer:
            self.compile(fullgraph=True)

    @property
    def supports_fused_back_pass(self):
        return True

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return False

    def init_step(self):
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                self.__init_state(p, group)

    @torch.no_grad()
    def __init_state(self, p, group):
        state = self.state[p]

        if 'is_mano' in state:
            return

        if group['use_mano']:
            state['factored'] = (
                group['nnmf_factor'] and
                not (len(p.shape) == 1 and not group['vector_reshape'])
            )

            dtype = torch.float32 if state['factored'] else p.dtype
            device = p.device

            if state['factored']:
                state['effective_shape'] = _get_effective_shape(p.numel())
                d1, d2 = state['effective_shape']
                state['mu_mbuf_nmf'] = torch.zeros(d1, device=device, dtype=dtype)
                state['mv_mbuf_nmf'] = torch.zeros(d2, device=device, dtype=dtype)
                packed_d2 = (d2 + 7) // 8
                state['sign_buf'] = torch.zeros((d1, packed_d2), dtype=torch.uint8, device=device)
            else:
                state['momentum_buffer'] = torch.zeros_like(p)

            state['step'] = 0
            
            group['adam_kourkoutas_beta'] = False
            state['is_mano'] = True 

        else: # AdamW
            Muon_AuxAdam._init_auxadam_state(self, p, group)
            state['is_mano'] = False


        if group.get('scaled_optm', False) and is_spectral(p):
            init_spectral_norm(group, state, p)

        _init_anchor(p, state, group)

    @torch.no_grad()
    def step_parameter(self, p: torch.Tensor, group: dict, i: int | None = None):
        grad = p.grad
        if grad is None:
            return

        state = self.state[p]
        self.__init_state(p, group)

        is_compiled = group.get('compiled_optimizer', False)
        random_int_tensor = None
        if p.dtype == torch.bfloat16 and self.stochastic_rounding and is_compiled:
            random_int_tensor = param_update._get_random_int_for_sr(p)

        if not state['is_mano']: # AdamW path
            step = state['step']
            beta1_adam, beta2_adam = group['adam_betas']

            if self.kourkoutas_helper:
                self.kourkoutas_helper.maybe_prepare_step(step, p.device)
                beta2_adam = self.kourkoutas_helper.get_beta2(p, group)

            if group['adam_use_bias_correction']:
                current_step = step + 1
                bias_correction1 = 1.0 - beta1_adam ** current_step
                sqrt_bias_correction2 = (1.0 - beta2_adam ** current_step)**0.5
            else:
                bias_correction1 = 1.0
                sqrt_bias_correction2 = 1.0

            step_size = group['lr'] / bias_correction1

            if is_compiled:
                step_size = torch.as_tensor(step_size)
                adam_step_param = self._compiled_adam_step_parameter
            else:
                adam_step_param = Muon_AuxAdam._adam_step_parameter

            adam_step_param(self, p, grad, state, group, beta1_adam, beta2_adam, sqrt_bias_correction2, step_size, random_int_tensor)
            state['step'] += 1

        else: # Mano path
            if is_compiled:
                lr = torch.as_tensor(group['lr'])
                mano_step_param = self._compiled_mano_step_parameter
            else:
                lr = group['lr']
                mano_step_param = self._mano_step_parameter

            mano_step_param(p, grad, state, group, lr, random_int_tensor)
            state['step'] += 1

    def compile(self, *args, **kwargs):
        self._compiled_mano_step_parameter = torch.compile(self._mano_step_parameter, *args, **kwargs)
        self._compiled_adam_step_parameter = torch.compile(Muon_AuxAdam._adam_step_parameter, *args, **kwargs)

    @torch.no_grad()
    def _mano_step_parameter(self, p, grad, state, group, lr, random_int_tensor):

        beta1 = group['beta1']
        nesterov = group['nesterov']
        Simplified_AdEMAMix = group['Simplified_AdEMAMix']
        rotate_method = group['rotate_method']
        alpha_grad = group['alpha_grad']
        weight_decay = group['weight_decay']

        # Ensure 2D view for Mano operations
        if state.get('factored', False):
            d1, d2 = state['effective_shape']
            p_flat = p.view(d1, d2)
        elif p.ndim > 2:
            # Flatten first dimension vs rest (Standard Conv2d handling)
            p_flat = p.view(p.shape[0], -1)
        else:
            p_flat = p

        if p_flat.ndim == 1:
            # Vectors
            dim = 0
        else:
            R, C = p_flat.shape
            if rotate_method == 'fixed':
                dim = 0 if R > C else 1
            elif rotate_method == 'auto_adjusted_ft':
                dim = 0 if (state['step'] % (R + C)) < R else 1
            else: # 'auto_ft'
                # Default Mano Rotation
                dim = int(state['step'] % 2)

        if grad.dtype != torch.float32 and state.get('factored', False):
            grad = grad.float()

        if state['factored']: # Factored Momentum
            d1, d2 = state['effective_shape']
            grad_reshaped = grad.view(d1, d2)

            # Reconstruct momentum
            mt_buf = _reconstruct_state((state['mu_mbuf_nmf'], state['mv_mbuf_nmf'], state['sign_buf'], d2), signed=True)

            if not Simplified_AdEMAMix:
                mt_buf.lerp_(grad_reshaped, 1 - beta1)
            else:
                mt_buf.mul_(beta1).add_(grad_reshaped)

            if nesterov:
                momentum_update = grad_reshaped.lerp(mt_buf, beta1)
            elif Simplified_AdEMAMix:
                momentum_update = torch.add(mt_buf, grad_reshaped, alpha=alpha_grad)
            else:
                momentum_update = mt_buf.clone()

            # Refactorize and store state
            state['mu_mbuf_nmf'], state['mv_mbuf_nmf'], state['sign_buf'] = _factorize_state(mt_buf, signed=True)
            del mt_buf


        else: # Standard Momentum (Linear, Conv2d flattened)

            mt_buf = state['momentum_buffer']

            if not Simplified_AdEMAMix:
                mt_buf.lerp_(grad, 1 - beta1)
            else:
                mt_buf.mul_(beta1).add_(grad)

            if nesterov:
                momentum_update = grad.lerp(mt_buf, beta1)
            elif Simplified_AdEMAMix:
                momentum_update = torch.add(mt_buf, grad, alpha=alpha_grad)
            else:
                momentum_update = mt_buf.clone()

        # Apply Mano Geometric Projection
        update_flat = mano_orthogonalization(p_flat, momentum_update, dim)

        if group.get('scaled_optm', False) and p.ndim != 1:
            # Spectral normalization
            update = update_flat.view(p.shape)
            update = spectral_normalization(update, vector_state=state.get('spectral_v'), lr=lr)
        else:
            # Apply Mano-RMS Rescaling
            update_flat = mano_rms_rescaling(p_flat, update_flat, dim, lr)
            update = update_flat.view(p.shape)

        param_update.apply_parameter_update(self, p, group, update, lr=lr, wd=weight_decay, random_int_tensor=random_int_tensor)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                self.step_parameter(p, group, i)

        return loss
