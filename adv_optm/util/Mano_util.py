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
    if n_layers is None:
        p_norm = torch.norm(p, p=2, dim=dim, keepdim=True).clamp_min_(1e-12)
    else:
        # Scale-invariant eps
        n_k = p.shape[dim]
        n_j = p.numel() // n_k
        eps = (1.0 / n_layers) * math.sqrt(n_k / n_j)
        p_norm = torch.norm(p, p=2, dim=dim, keepdim=True).add_(eps)
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
