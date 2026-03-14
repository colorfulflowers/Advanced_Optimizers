import torch

import math

@torch.no_grad()
def mano_orthogonalization(
    p: torch.Tensor,
    g: torch.Tensor,
    dim: int,
    n_layers: int | None = None,
) -> torch.Tensor:
    """
    Core geometric projection for Mano.

    Algorithm:
    1. Project momentum (g) onto the tangent space of the Oblique manifold at p.
    2. Normalize result to stay on manifold.
    """
    g = g.view(p.shape) # p_flat

    # Project momentum onto Tangent Space
    # p_unit = p / ||p||
    p_norm = torch.norm(p, p=2, dim=dim, keepdim=True).clamp_min_(1e-12)
    p_unit = p / p_norm

    # dot = <g, p_unit>
    dot_prod = torch.sum(g * p_unit, dim=dim, keepdim=True)
    # tangent_momentum = g - dot * p_unit
    tangent_momentum = g - (dot_prod * p_unit)

    # Manifold Normalization (Mapping back to Oblique)
    # u = tangent / ||tangent||
    if n_layers is None:
        tm_norm = torch.norm(tangent_momentum, p=2, dim=dim, keepdim=True).clamp_min_(1e-12)
    else:
        n_k = p.shape[dim]
        n_j = p.numel() // n_k
        eps = (1.0 / n_layers) * math.sqrt(n_k / n_j)
        tm_norm = torch.norm(tangent_momentum, p=2, dim=dim, keepdim=True).add_(eps)
    u = tangent_momentum / tm_norm
    return u

@torch.no_grad()
def mano_rms_rescaling(
    p: torch.Tensor,
    u: torch.Tensor,
    dim: int,
    lr: float,
) -> torch.Tensor:
    # Formula: update = 0.2 * sqrt(n_k) * u
    # where n_k is the size of the dimension being normalized over.
    n_k = p.shape[dim]
    scaling_factor = 0.2 * math.sqrt(n_k)

    # Apply learning rate
    return u.mul_(lr * scaling_factor)

def _pseudo_rand(step: int, R: int, C: int) -> float:
    """Fast deterministic random, returning float in [0, 1]"""
    # Use python's arbitrarily large integers; bitwise masking ensures it mimics C's 32-bit unsigned math
    seed = (step * 3141592653) ^ (R * 2718281829) ^ (C * 161803398)
    seed = ((seed ^ (seed >> 16)) * 0x85ebca6b) & 0xFFFFFFFF
    seed = ((seed ^ (seed >> 13)) * 0xc2b2ae35) & 0xFFFFFFFF
    seed = (seed ^ (seed >> 16)) & 0xFFFFFFFF
    return seed / 4294967295.0

def get_mano_dim(p_flat: torch.Tensor, rotate_method: str, step: int, state: dict | None = None) -> int:
    """
    Determines the dimension along which to apply Mano orthogonalization.
    
    Args:
        p_flat (torch.Tensor): The flattened parameter tensor (1D or 2D).
        rotate_method (str): Method to choose the manifold rotation dimension 
                             ('fixed', 'auto_ft', 'auto_adjusted_ft', 'stochastic_adjusted_ft', 'accum_stochastic_adjusted_ft').
        step (int): The current optimizer step for the parameter.
        state (dict, optional): The optimizer state dict for the parameter (needed for accum_stochastic_adjusted_ft).

    Returns:
        int: The dimension (0 or 1) to rotate/orthogonalize on.
    """
    if p_flat.ndim == 1:
        # Vectors
        return 0

    rotate_method = 'accum_stochastic_adjusted_ft'

    R, C = p_flat.shape
    if rotate_method == 'fixed':
        return 0 if R > C else 1
    elif rotate_method == 'auto_adjusted_ft':
        return 0 if (step % (R + C)) < R else 1
    elif rotate_method == 'stochastic_adjusted_ft':
        rand_val = _pseudo_rand(step, R, C)
        return 0 if rand_val < (R / (R + C)) else 1
    elif rotate_method == 'accum_stochastic_adjusted_ft':
        minor_dim = 1 if C <= R else 0
        major_dim = 1 - minor_dim
        minor_prob = min(R, C) / (R + C)

        if state is not None:
            if 'mano_accum_prob' not in state:
                state['mano_accum_prob'] = minor_prob
            current_prob = state['mano_accum_prob']
        else:
            current_prob = minor_prob

        rand_val = _pseudo_rand(step, R, C)

        if rand_val < current_prob:
            # Hit minority dimension, reset probability
            state['mano_accum_prob'] = minor_prob
            return minor_dim
        else:
            # Missed minority dimension, accumulate probability for next steps
            state['mano_accum_prob'] += minor_prob
            return major_dim
    else: # 'auto_ft'
        # Default Mano Rotation
        return int(step % 2)
