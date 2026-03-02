import torch
import math

from .factorization_util import _unpack_bools

@torch.no_grad()
def _update_alias_state(
    grad: torch.Tensor,
    alias_d: torch.Tensor,
    alias_d_raw: torch.Tensor,
    alias_eta: torch.Tensor,
    # State for memory efficient approx (approx_alias=True)
    prev_grad_max: torch.Tensor | None,
    prev_grad_min: torch.Tensor | None,
    # State for exact computation (approx_alias=False)
    prev_grad: torch.Tensor | None,
    # Common states
    prev_sign: torch.Tensor | None,
    prev_lr: torch.Tensor,
    approx_alias: bool,
    packed_sign: bool,
    original_shape: tuple | torch.Size,
    step: int,
) -> None:
    """
    Updates the ALIAS parameter-free adaptive terms (d and eta).
    """
    numel = math.prod(original_shape) if isinstance(original_shape, tuple) else original_shape.numel()

    # Update Distance d^t (Numerator)
    if prev_sign is not None and step > 0:
        if packed_sign:
            unpacked = _unpack_bools(prev_sign.view(1, -1), original_m=numel)
            prev_sign_t = torch.where(unpacked.view(original_shape), 1.0, -1.0).to(grad.dtype)
        else:
            prev_sign_t = torch.where(prev_sign > 0, 1.0, -1.0).to(grad.dtype)

        dot_product = (prev_sign_t * grad).sum()
        alias_d_raw.add_(prev_lr * dot_product)
        alias_d.copy_(torch.maximum(alias_d, alias_d_raw))

    # Update Smoothness eta^t (Denominator)
    # Based on Algorithm 2: Denominator is the L_inf norm of the step difference.
    # Because it's a Sign update, the max absolute value is simply the learning rate.
    step_inf_norm = prev_lr

    if approx_alias:
        # Memory-efficient ALIAS approximation
        current_grad_max = grad.max()
        current_grad_min = grad.min()

        if prev_grad_max is not None and prev_grad_min is not None and step > 0:
            approx_inf_norm = torch.maximum(
                (current_grad_max - prev_grad_min).abs(),
                (prev_grad_max - current_grad_min).abs()
            )
            # Approximate the L1 norm of the grad diff: (L_inf * numel)
            approx_l1_norm = approx_inf_norm * numel
            alias_eta.add_(approx_l1_norm / step_inf_norm)

        # Store current stats for next step
        if prev_grad_max is not None: prev_grad_max.copy_(current_grad_max)
        if prev_grad_min is not None: prev_grad_min.copy_(current_grad_min)

    else:
        # Exact ALIAS computation
        # Requires full prev_grad tensor
        if prev_grad is not None:
            if step > 0:
                # Exact L1 norm of the grad diff
                exact_l1_norm = (grad - prev_grad).abs().sum()
                alias_eta.add_(exact_l1_norm / step_inf_norm)

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