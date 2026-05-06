import torch
import math

def _block_size_from_n_elements(n_elements: int) -> int:
    """Invert  n = bs*(bs-1)/2  to recover block_size."""
    return round((1 + math.sqrt(1 + 8 * n_elements)) / 2)

def get_geodesic_decay_scaler(p: torch.Tensor) -> torch.Tensor:
    """
    Computes the scalar multiplier for geodesic weight decay.

    Near identity (‖Q‖ ≈ 0), this returns 1.0 (standard L2 decay). 
    Far from identity, it decays towards 0, respecting the bounded 
    geometry of SO(n) so large rotations aren't infinitely penalized.

    Args:
        p: Tensor of shape (rank, n_elements) representing the upper triangular elements

    Returns:
        Tensor of shape (rank, 1) to be multiplied by wd and p.
    """
    block_size = _block_size_from_n_elements(p.shape[-1])

    # Sum of squared elements equals the sum of squared eigenvalues of Q
    block_norm_sq = (p * p).sum(dim=-1, keepdim=True)   # (r, 1)

    # We estimate the average squared eigenvalue by dividing by the number of pairs (d/2)
    mean_eigenval_sq = block_norm_sq / (block_size // 2.0)

    decay_scaler = 1.0 / (1.0 + mean_eigenval_sq)

    return decay_scaler


def apply_riemannian_preconditioning(p: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    """
    Scales the update by the exact Inverse Metric of the Cayley transform.

    Uses True Matrix Preconditioning: M @ G @ M where M = (I - Q^2),
    and Q is the skew-symmetric matrix form of parameters p.
    """
    n_el = p.shape[-1]
    block_size = _block_size_from_n_elements(n_el)
    device, dtype = p.device, p.dtype

    # 1. Get indices for upper triangular elements
    rows, cols = torch.triu_indices(block_size, block_size, 1, device=device)
    
    # 2. Flatten any prepended batch dimensions for processing
    orig_shape = p.shape
    p_flat = p.view(-1, n_el)
    update_flat = update.view(-1, n_el)
    batch_size = p_flat.shape[0]

    # Initialize matrices
    Q = torch.zeros(batch_size, block_size, block_size, device=device, dtype=dtype)
    G = torch.zeros(batch_size, block_size, block_size, device=device, dtype=dtype)
    batch_idx = torch.arange(batch_size, device=device)[:, None]

    # Construct skew-symmetric parameter matrix Q
    Q = Q.index_put((batch_idx, rows, cols), p_flat)
    Q = Q - Q.transpose(-2, -1)
    
    # Construct skew-symmetric gradient matrix G
    G = G.index_put((batch_idx, rows, cols), update_flat)
    G = G - G.transpose(-2, -1)

    # Compute True Matrix Preconditioner M = I - Q^2
    # Since Q is skew-symmetric, Q^2 is negative semi-definite, ensuring M >= I.
    I = torch.eye(block_size, device=device, dtype=dtype).unsqueeze(0)
    Q_sq = torch.bmm(Q, Q)
    M = I - Q_sq

    # Apply exact preconditioning: G_prec = M @ G @ M
    G_prec = torch.bmm(torch.bmm(M, G), M)
    
    # Extract the preconditioned upper-triangular elements
    update_prec_flat = G_prec[batch_idx, rows, cols]

    # Update inplace to respect original function signature/behavior
    update.copy_(update_prec_flat.view(orig_shape))

    return update
