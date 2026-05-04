import torch
import math

def _block_size_from_n_elements(n_elements: int) -> int:
    """Invert  n = bs*(bs-1)/2  to recover block_size."""
    return round((1 + math.sqrt(1 + 8 * n_elements)) / 2)

def get_geodesic_decay_scaler(p: torch.Tensor) -> torch.Tensor:
    """
    Computes the True Matrix geodesic weight decay direction.
    """
    n_el = p.shape[-1]
    block_size = _block_size_from_n_elements(n_el)
    device, dtype = p.device, p.dtype

    # 1. Get indices for upper triangular elements
    rows, cols = torch.triu_indices(block_size, block_size, 1, device=device)
    
    # 2. Flatten any prepended batch dimensions for processing
    orig_shape = p.shape
    p_flat = p.view(-1, n_el)
    batch_size = p_flat.shape[0]
    
    # 3. Construct skew-symmetric matrix Q
    Q = torch.zeros(batch_size, block_size, block_size, device=device, dtype=dtype)
    batch_idx = torch.arange(batch_size, device=device)[:, None]
    
    Q = Q.index_put((batch_idx, rows, cols), p_flat)
    Q = Q - Q.transpose(-2, -1)
    
    # 4. Compute True Matrix Dampener: I - (Q^2 / n_el)
    # Since Q is skew-symmetric, Q^2 is negative semi-definite.
    # Therefore, (I - Q^2) is positive definite and safely invertible.
    I = torch.eye(block_size, device=device, dtype=dtype).unsqueeze(0)
    Q_sq = torch.bmm(Q, Q)
    
    # Mirroring your scalar logic: 1 / (1 + ||Q||^2 / n_el)
    dampener_matrix = I - (Q_sq / n_el)
    
    # 5. Apply the dampener to Q: inv(Dampener) @ Q
    # Using linalg.solve is faster and numerically more stable than explicit inverse
    # Upcast to float32 for the solve operation
    # BFloat16 is not supported by MAGMA's batched LU solver.
    decay_Q = torch.linalg.solve(
        dampener_matrix.to(torch.float32), 
        Q.to(torch.float32)
    ).to(dtype)

    # 6. Extract the preconditioned upper-triangular elements
    decay_p_flat = decay_Q[batch_idx, rows, cols]
    
    # 7. Return reshaped to match original p
    return decay_p_flat.view(orig_shape)


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
