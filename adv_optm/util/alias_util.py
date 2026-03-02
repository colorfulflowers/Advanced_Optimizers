import torch
import math

from .factorization_util import _unpack_bools

@torch.no_grad()
def _update_alias_state(
    grad: torch.Tensor,
    alias_d: torch.Tensor,
    alias_d_raw: torch.Tensor,
    alias_eta: torch.Tensor,
    # State for memory efficient approx (approx_var=True)
    prev_grad_max: torch.Tensor | None,
    prev_grad_min: torch.Tensor | None,
    # State for exact computation (approx_var=False)
    prev_grad: torch.Tensor | None,
    # Common states
    prev_sign: torch.Tensor | None,
    prev_lr: torch.Tensor,
    approx_var: bool,
    packed_sign: bool,
    original_shape: tuple | torch.Size,
) -> None:
    """
    Updates the ALIAS parameter-free adaptive terms (d and eta).
    """
    numel = math.prod(original_shape) if isinstance(original_shape, tuple) else original_shape.numel()

    # Update Distance d^t (Numerator)
    if prev_sign is not None:
        if packed_sign:
            unpacked = _unpack_bools(prev_sign.view(1, -1), original_m=numel)
            prev_sign_t = torch.where(unpacked.view(original_shape), 1.0, -1.0).to(grad.dtype)
        else:
            prev_sign_t = torch.where(prev_sign > 0, 1.0, -1.0).to(grad.dtype)

        # Dot product: <grad, sign(prev_update)>
        dot_product = (prev_sign_t * grad).sum()

        alias_d_raw.add_(prev_lr * dot_product)
        alias_d.copy_(torch.maximum(alias_d, alias_d_raw))

    # Update Smoothness eta^t (Denominator)
    step_l1_norm = (prev_lr * numel).clamp_min_(1e-12)

    if approx_var:
        # Memory-efficient ALIAS approximation
        current_grad_max = grad.max()
        current_grad_min = grad.min()

        if prev_grad_max is not None and prev_grad_min is not None:
            # || grad^t - grad^{t-1} ||_inf approx
            approx_inf_norm = torch.maximum(
                (current_grad_max - prev_grad_min).abs(),
                (prev_grad_max - current_grad_min).abs()
            )
            alias_eta.add_(approx_inf_norm / step_l1_norm)

        # Store current stats for next step
        if prev_grad_max is not None: prev_grad_max.copy_(current_grad_max)
        if prev_grad_min is not None: prev_grad_min.copy_(current_grad_min)

    else:
        # Exact ALIAS computation
        # Requires full prev_grad tensor
        if prev_grad is not None:
            # Exact || grad^t - grad^{t-1} ||_inf
            exact_inf_norm = (grad - prev_grad).abs().max()
            alias_eta.add_(exact_inf_norm / step_l1_norm)
            # Update history
            prev_grad.copy_(grad)

@torch.no_grad()
def _get_alias_lr(
    alias_d: torch.Tensor,
    alias_eta: torch.Tensor,
    base_lr: float, 
) -> torch.Tensor:
    """
    Calculates the ALIAS adaptive learning rate.
    """
    # gamma^t = sqrt(d^t / eta^t)
    return torch.where(
        alias_eta > 0,
        torch.sqrt(alias_d / alias_eta.clamp_min(1e-12)),
        torch.tensor(base_lr, dtype=alias_d.dtype)
    )