import torch

from . import param_update

import math

def scale_update(
    p: torch.Tensor,
    update: torch.Tensor,
    lr: float,
    vector_state: torch.Tensor | None = None,
    depth: int = 1,
) -> torch.Tensor:
    """
    Applies adaptive scaling to the parameter update based on the parameter's
    role (DoRA, OFT, or LoRA/Full Finetuning).

    Args:
        p: The original parameter tensor.
        update: The computed gradient/update tensor to be scaled.
        lr: The learning rate.
        vector_state: The singular vector state used for spectral normalization.

    Returns:
        The scaled update tensor.
    """
    is_dora_scale = getattr(p, '_is_dora_scale', False)
    is_oft = getattr(p, '_is_oft', False)

    # DoRA Magnitude Scales (1D) or 1D Bias/Norm layers
    if p.ndim < 2 or is_dora_scale:
        return l2_normalization(update, dim=None, lr=lr)

    # Orthogonal Fine-Tuning (OFT)
    # This guarantees O(1) update complexity scaling, independent of block sizes.
    if is_oft:
        n = update.shape[1]
        # Calculate block size (b)
        b = (1 + math.sqrt(1 + 8 * n)) / 2
        target_norm = math.sqrt(b / 8)
        scale = target_norm / math.sqrt(n)
        return rms_normalization(update, dim=1, lr=lr * scale)

    # LoRA Factors or Full Finetuning weights
    # Scales update to maintain consistent spectral norm across different layer sizes and ranks.
    if p.ndim >= 2:
        if getattr(p, '_is_lora_A', False):
            rank = p.shape[0]
            B = getattr(p, '_lora_pair', None)
            if B is not None:
                # Symmetric treatment: normalize σ(B @ δA) to full-training target.
                # Guard: B is zero-initialised; skip until it has meaningful scale.
                sigma_B = B.detach().norm()
                if sigma_B > 1e-4:
                    return spectral_normalization_lora_A(update, B.detach(),
                                                        vector_state, lr / depth)
            return l2_normalization(update, dim=1, lr=lr / math.sqrt(rank))

        if getattr(p, '_is_lora_B', False):
            A = getattr(p, '_lora_pair', None)
            if A is not None:
                return spectral_normalization_lora_B(update, A.detach(),
                                                    vector_state, lr / depth)
            return spectral_normalization(update, vector_state, lr / depth)

        return spectral_normalization(update, vector_state, lr / depth)

    return update.mul_(lr)


def scale_eps(group: dict, p) -> float:
    """
    Scales Adam eps to be scale-invariant.
    """
    if group.get('scaled_optm', False):
        pair = getattr(p, '_lora_pair', None)

        if pair is not None:
            # LoRA pair: eps based on effective weight numel (d_out * d_in), not
            # parameter numel (d_out * rank or rank * d_in).
            # Both factors jointly represent one weight matrix, so both use the
            # same formula — consistent with the pair-aware spectral norm and K-β.
            # Depth scaling applies to both: after the spectral norm fix, the
            # combined step compounds through depth like any full-rank weight.
            if getattr(p, '_is_lora_B', False):
                d_out = p.shape[0]
                d_in  = pair.numel() // pair.shape[0]
            else:  # _is_lora_A
                d_in  = p.numel() // p.shape[0]
                d_out = pair.shape[0]
            effective_numel = d_out * d_in
            adaptive_eps = (1.0 / group['n_layers']) * (1.0 / math.sqrt(effective_numel))

        elif getattr(p, '_is_dora_scale', False) or getattr(p, '_is_oft', False) \
                or p.ndim < 2:
            adaptive_eps = (1.0 / math.sqrt(p.numel()))
        else:
            adaptive_eps = (1.0 / group['n_layers']) * (1.0 / math.sqrt(p.numel()))
    else:
        adaptive_eps = group['eps']
    return adaptive_eps

def adjust_wds(wd: float, cwd: float, p: torch.Tensor) -> tuple[float, float]:
    """
    Adjusts standard weight decay and centered weight decay.
    """
    # DoRA Scale (Magnitude Vector)
    if getattr(p, '_is_dora_scale', False):
        return wd, cwd

    if getattr(p, '_is_oft', False):
        return wd, 0.0

    if p.ndim >= 2:
        is_lora = getattr(p, '_is_lora_A', False) or getattr(p, '_is_lora_B', False)
        if is_lora:
            return wd, 0.0

        return wd, cwd

    else:
        # 1D Biases or generic 1D parameters
        # Centered WD safely regularizes the delta without collapsing base feature variance.
        return 0.0, cwd


def scale_wds(wd: float, cwd: float, p: torch.Tensor) -> tuple[float, float]:
    """
    Scales standard weight decay and centered weight decay based on the parameter's
    shape and type to maintain effective regularization strength.
    """
    if p.ndim >= 2:
        fan_in = p.numel() // p.shape[0]
        return wd / fan_in, cwd / fan_in

    # 1D tensors (like DoRA scale and Biases)
    return wd, cwd


@torch.no_grad()
def l2_normalization(update: torch.Tensor, dim: int | None, lr: float) -> torch.Tensor:
    """Performs L2 normalization on the update tensor."""
    norm = torch.linalg.vector_norm(update, ord=2, dim=dim, keepdim=True).clamp_min_(1e-12)
    return update.mul_(lr / norm)


@torch.no_grad()
def rms_normalization(update: torch.Tensor, dim: int | None, lr: float) -> torch.Tensor:
    """Performs Root Mean Square normalization on the update tensor."""
    n = update.numel() if dim is None else update.shape[dim]
    norm = torch.linalg.vector_norm(update, ord=2, dim=dim, keepdim=True).clamp_min_(1e-12)
    scale_n = math.sqrt(n)
    return update.mul_(lr * scale_n / norm)


def is_spectral(p: torch.Tensor) -> bool:
    """Determines if a parameter should undergo spectral normalization updates."""
    if getattr(p, '_is_lora_A', False) or getattr(p, '_is_lora_B', False):
        return True
    if getattr(p, '_is_oft', False) or getattr(p, '_is_dora_scale', False) or p.ndim == 1:
        return False
    return getattr(p, 'is_hidden', True)

@torch.no_grad()
def init_spectral_norm(group: dict, state: dict, p: torch.Tensor):
    """Initializes the singular vector 'v' for the Power Iteration method."""
    gen = param_update.get_generator(p.device)

    A = getattr(p, '_lora_pair', None)
    if getattr(p, '_is_lora_B', False) and A is not None:
        # v must live in d_in-space of the *combined* product δB @ A,
        # not in rank-space (p.numel()//p.shape[0] would give rank).
        v_dim = A.numel() // A.shape[0]   # = d_in of the original layer
    else:
        v_dim = p.numel() // p.shape[0]

    v = torch.randn(v_dim, device=p.device, dtype=p.dtype, generator=gen)
    state['spectral_v'] = v.div_(v.norm().clamp_min_(1e-12))

@torch.no_grad()
def spectral_normalization(update: torch.Tensor, vector_state: torch.Tensor, lr: float) -> torch.Tensor:
    """
    Applies Spectral Normalization via a single step of Power Iteration.
    Implementation follows: "Scalable Optimization in the Modular Norm" (arXiv:2405.14813).
    """
    d_out = update.shape[0]
    d_in = update.numel() // d_out
    update = update.to(vector_state.dtype)
    update_flat = update.view(d_out, d_in)
    # Target scale derived from the "Modular Norm" paper
    target_scale = math.sqrt(d_out / d_in)
    # Power Iteration step to estimate the largest singular value (sigma)
    # u = Wv
    u = torch.mv(update_flat, vector_state)
    # v_new = W.T u
    v_new = torch.mv(update_flat.mT, u)

    v_norm = torch.linalg.vector_norm(v_new)

    # Stability: Only update the state if the norm is significant
    candidate_v = v_new / v_norm
    next_state = torch.where(v_norm >= 0.5, candidate_v, vector_state)
    vector_state.copy_(next_state.to(vector_state.dtype))

    Av = torch.mv(update_flat, vector_state)

    # Calculate sigma (the spectral norm)
    sigma = torch.linalg.vector_norm(Av)

    # Rescale update
    scale = lr * (target_scale / sigma.clamp_min_(1e-12))
    return update.mul_(scale)

@torch.no_grad()
def spectral_normalization_lora_B(
    update_B: torch.Tensor,
    A: torch.Tensor,
    vector_state: torch.Tensor,
    lr: float,
) -> torch.Tensor:
    """
    Spectral normalization for lora_B using the *combined* weight-space step.

    Never materialises the [d_out, d_in] product. Instead uses the identity:
        M = δB @ A   (shape [d_out, d_in])
        M v  = δB @ (A @ v)        — one [rank] mv + one [d_out] mv
        Mᵀ u = Aᵀ @ (δBᵀ @ u)     — one [rank] mv + one [d_in]  mv
    This keeps O(rank · (d_out + d_in)) instead of O(d_out · d_in).
    """
    d_out = update_B.shape[0]
    rank  = update_B.numel() // d_out
    d_in  = vector_state.shape[0]          # set correctly by init_spectral_norm

    dtype = vector_state.dtype
    B_flat = update_B.view(d_out, rank).to(dtype)
    A_flat = A.view(rank, d_in).to(dtype)

    # Target matches full training: √(d_out / d_in_orig)
    target_scale = math.sqrt(d_out / d_in)

    # Power iteration step
    Av     = torch.mv(A_flat, vector_state)          # [rank]
    u      = torch.mv(B_flat, Av)                    # [d_out]
    BTu    = torch.mv(B_flat.mT, u)                  # [rank]
    v_new  = torch.mv(A_flat.mT, BTu)                # [d_in]

    v_norm = torch.linalg.vector_norm(v_new)
    candidate_v = v_new / v_norm
    next_state  = torch.where(v_norm >= 0.5, candidate_v, vector_state)
    vector_state.copy_(next_state)

    # σ estimate with updated v
    Av    = torch.mv(A_flat, vector_state)
    u     = torch.mv(B_flat, Av)
    sigma = torch.linalg.vector_norm(u)

    scale = lr * (target_scale / sigma.clamp_min_(1e-12))
    return update_B.mul_(scale)

@torch.no_grad()
def spectral_normalization_lora_A(
    update_A: torch.Tensor,
    B: torch.Tensor,
    vector_state: torch.Tensor,
    lr: float,
) -> torch.Tensor:
    d_out = B.shape[0]
    rank  = B.numel() // d_out
    d_in  = vector_state.shape[0]

    dtype  = vector_state.dtype
    A_flat = update_A.view(rank, d_in).to(dtype)
    B_flat = B.view(d_out, rank).to(dtype)

    target_scale = math.sqrt(d_out / d_in)

    Av     = torch.mv(A_flat, vector_state)          # [rank]
    u      = torch.mv(B_flat, Av)                    # [d_out]
    BTu    = torch.mv(B_flat.mT, u)                  # [rank]
    v_new  = torch.mv(A_flat.mT, BTu)                # [d_in]

    v_norm = torch.linalg.vector_norm(v_new)
    candidate_v = v_new / v_norm
    next_state  = torch.where(v_norm >= 0.5, candidate_v, vector_state)
    vector_state.copy_(next_state)

    Av    = torch.mv(A_flat, vector_state)
    u     = torch.mv(B_flat, Av)
    sigma = torch.linalg.vector_norm(u)

    scale = lr * (target_scale / sigma.clamp_min_(1e-12))
    return update_A.mul_(scale)