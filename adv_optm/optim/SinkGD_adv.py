import torch
from typing import Optional

from ..util import param_update
from ..util.factorization_util import _get_effective_shape, _reconstruct_state, _factorize_state, _pack_bools, _unpack_bools
from ..util.scaled_optm import scale_update, is_spectral, init_spectral_norm
from ..util.centered_decay import _init_anchor
from ..util.sinkhorn_util import apply_sr_sinkhorn


class SinkGD_adv(torch.optim.Optimizer):
    """
    Implements an advanced SinkGD (Sinkhorn Gradient Descent) algorithm.
    Based on 'Gradient Multi-Normalization for Efficient LLM Training'.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate (default: 1e-4).
        momentum (float, optional): coefficients for computing running average of the gradients. 
            Default is 0.0 as SinkGD is designed to be a highly efficient stateless optimizer.
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.0).
        cautious_wd (bool): Enables Cautious Weight Decay.
        vector_reshape (bool, optional): whether to reshape 1D vectors into 2D matrices (default: True).
        stochastic_rounding (bool, optional): whether to use stochastic rounding for BF16 (default: True).
        sinkhorn_iters (int, optional): The number of alternating row/col normalization steps. (default: 5)
        Simplified_AdEMAMix (bool): whether to use the Simplified AdEMAMix update rule. (default: False)
        alpha_grad (float): Mixing coefficient for Simplified AdEMAMix. (default: 1.0)
        centered_wd (float): Centered Weight Decay coefficient. (default: 0.0)
        centered_wd_mode (str): Quantization format for anchor weights. (default: 'float8')
        nnmf_factor (bool): whether to use the factorization or use the uncompressed optimizer. (default: False)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        cautious_wd: bool = False,
        stochastic_rounding: bool = True,
        sinkhorn_iters: int = 5,
        alpha_grad: float = 1.0,
        Simplified_AdEMAMix: bool = False,
        centered_wd: float = 0.0,
        centered_wd_mode: str = 'float8',
        scaled_optm: bool = False,
        nnmf_factor: bool = False,
        vector_reshape: bool = True,
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
            vector_reshape=vector_reshape,
            sinkhorn_iters=sinkhorn_iters,
            alpha_grad=alpha_grad,
            Simplified_AdEMAMix=Simplified_AdEMAMix,
            scaled_optm=scaled_optm,
            centered_wd=centered_wd,
            centered_wd_mode=centered_wd_mode,
            nnmf_factor=nnmf_factor,
        )
        self.stochastic_rounding = stochastic_rounding
        self._init_lr = lr
        super().__init__(params, defaults)

        if self.stochastic_rounding:
            devices = {p.device for group in self.param_groups for p in group['params'] if p.dtype == torch.bfloat16}
            for device in devices:
                param_update.set_seed(device)

        self._compiled_step_parameter = None
        if compiled_optimizer:
            self.compile(fullgraph=True)

    def load_state_dict(self, state_dict: dict) -> None:
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
        if p.grad is None:
            return

        grad = p.grad
        state = self.state[p]

        # Conditionally allocate state. Pure SinkGD (stateless) avoids any allocations
        needs_state = (
            group["momentum"] > 0 or 
            group.get("scaled_optm", False) or 
            group.get("centered_wd", 0.0) > 0.0
        )

        if needs_state and len(state) == 0:
            state['factored'] = (
                group['nnmf_factor'] and
                not (len(p.shape) == 1 and not group['vector_reshape'])
            )

            dtype = torch.float32 if state['factored'] else p.dtype

            if group["momentum"] > 0:
                if state['factored']:
                    state['effective_shape'] = _get_effective_shape(p.numel())
                    d1, d2 = state['effective_shape']
                    
                    if group["momentum"] > 0:
                        state['mu_m_nmf'] = torch.zeros(d1, device=p.device, dtype=dtype)
                        state['mv_m_nmf'] = torch.zeros(d2, device=p.device, dtype=dtype)
                        packed_d2 = (d2 + 7) // 8
                        state['sign'] = torch.zeros((d1, packed_d2), dtype=torch.uint8, device=p.device)
                else:
                    if group["momentum"] > 0:
                        state['exp_avg'] = torch.zeros_like(p, device=p.device, dtype=dtype)

            if group.get('scaled_optm', False) and is_spectral(p):
                init_spectral_norm(group, state, p)

            _init_anchor(p, state, group)

        lr = group["lr"]
        random_int_tensor = None

        if group.get('compiled_optimizer', False):
            if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                random_int_tensor = param_update._get_random_int_for_sr(p)
            lr = torch.as_tensor(lr)
            step_param_fn = self._compiled_step_parameter
        else:
            step_param_fn = self._step_parameter

        step_param_fn(p, grad, state, group, lr, random_int_tensor)

    def _step_parameter(self, p, grad, state, group, lr, random_int_tensor):
        if grad.dtype != torch.float32 and state.get('factored', False):
            grad = grad.float()

        sinkhorn_iters = group.get("sinkhorn_iters", 5)
        momentum = group["momentum"]
        Simplified_AdEMAMix = group["Simplified_AdEMAMix"]
        alpha_grad = group["alpha_grad"]

        if not Simplified_AdEMAMix:
            alpha_grad = 0
        elif momentum == 0:
            alpha_grad = 1

        if state.get('factored'):
            d1, d2 = state['effective_shape']
            grad_reshaped = grad.view(d1, d2)


            if momentum > 0:
                exp_avg = _reconstruct_state((state['mu_m_nmf'], state['mv_m_nmf'], state['sign'], d2), signed=True)
                exp_avg.mul_(momentum).add_(grad_reshaped)

                if Simplified_AdEMAMix:
                    raw_update = exp_avg + (grad_reshaped * alpha_grad)
                else:
                    raw_update = exp_avg.clone()

                state['mu_m_nmf'], state['mv_m_nmf'], state['sign'] = _factorize_state(exp_avg, signed=True)
            else:
                raw_update = grad_reshaped.clone()

            raw_update = raw_update.view(p.shape)

            # Apply SR-Sinkhorn projection
            update = apply_sr_sinkhorn(raw_update, sinkhorn_iters)

        else:
            if momentum > 0:
                exp_avg = state["exp_avg"]
                exp_avg.mul_(momentum).add_(grad)

                if Simplified_AdEMAMix:
                    raw_update = exp_avg + (grad * alpha_grad)
                else:
                    raw_update = exp_avg.clone()
            else:
                raw_update = grad.clone()

            # Apply SR-Sinkhorn projection
            update = apply_sr_sinkhorn(raw_update, sinkhorn_iters)


        if group.get('scaled_optm', False):
            update = scale_update(p, update, lr, vector_state=state.get('spectral_v'))
        else:
            update.mul_(lr)

        param_update.apply_parameter_update(self, p, group, update, lr, random_int_tensor=random_int_tensor)

    def compile(self, *args, **kwargs):
        self._compiled_step_parameter = torch.compile(self._step_parameter, *args, **kwargs)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is not None:
                    self.step_parameter(p, group, i)

        return loss