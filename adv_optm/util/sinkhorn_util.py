import math
import torch
from .factorization_util import _get_effective_shape

def apply_sr_sinkhorn(update: torch.Tensor, iters: int = 5) -> torch.Tensor:
    """
    Applies Square-Root Sinkhorn (SR-Sinkhorn) multi-normalization.
    As described in 'Gradient Multi-Normalization for Efficient LLM Training'.
    
    This technique normalizes a 2D matrix alternatively by its row-wise L2 norm 
    and column-wise L2 norm, driving it toward a fixed point that uniformly 
    distributes update magnitudes, effectively replacing EMA state dependencies.
    """
    original_shape = update.shape

    # Handle 1D Tensors via factorization or standard fallback
    if update.dim() == 1:
        d1, d2 = _get_effective_shape(update.numel())
        if d1 == 1 or d2 == 1:
            # Cannot be meaningfully factorized into 2D.
            # Fallback to standard L2 normalization scaled by sqrt(dim)
            # This matches the Frobenius norm of a signed vector / Sinkhorn output.
            norm = update.norm(p=2).clamp_min_(1e-12)
            return update.div_(norm).mul_(math.sqrt(update.numel()))

        # View as 2D for the Sinkhorn operation
        update_2d = update.view(d1, d2)
    else:
        # Flatten >= 3D tensors into 2D matrices
        update_2d = update.view(update.shape[0], -1)

    m, n = update_2d.shape
    sqrt_n = math.sqrt(n)
    sqrt_m = math.sqrt(m)

    # In-place alternating Sinkhorn normalization steps
    for _ in range(iters):
        # 1. Row-wise L2 normalization: X <- sqrt(n) * Q(X)^-1 * X
        row_norm = update_2d.norm(p=2, dim=1, keepdim=True).clamp_min_(1e-12)
        update_2d.div_(row_norm).mul_(sqrt_n)

        # 2. Column-wise L2 normalization: X <- sqrt(m) * X * R(X)^-1
        col_norm = update_2d.norm(p=2, dim=0, keepdim=True).clamp_min_(1e-12)
        update_2d.div_(col_norm).mul_(sqrt_m)

    return update_2d.view(original_shape)