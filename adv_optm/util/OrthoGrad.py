import torch

def _orthogonalize_gradient(p: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """
    Projects the gradient `grad` to be orthogonal to the parameter `p`.
    Modified from:
    https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/blob/720d2444df12b851d6cb417ab08cf125c822b2ae/orthograd.py
    """
    is_lora_A = getattr(p, '_is_lora_A', False)
    is_lora_B = getattr(p, '_is_lora_B', False)

    # Pair-aware Full-Weight Orthogonalization
    if is_lora_A or is_lora_B:
        pair = getattr(p, '_lora_pair', None)
        if pair is not None:
            # Explicit zero-initialization gating (same as Muon/scaled_optm stabilization)
            return _orthogonalize_lora_gradient_pair_granular(p, grad, pair.detach(), is_lora_A)

    # Granular fallback for OFT or unpaired LoRA
    if getattr(p, '_is_oft', False) or is_lora_A:
        return _orthogonalize_gradient_granular(p, grad, dim=1)
    elif is_lora_B:
        return _orthogonalize_gradient_granular(p, grad, dim=0)

    # Standard global orthogonalization
    original_shape = grad.shape
    original_dtype = grad.dtype
    w = p.view(-1).float()
    g = grad.view(-1).float()
    w_norm_sq = torch.dot(w, w).add_(1e-30)
    proj = torch.dot(w, g) / w_norm_sq
    g_orth = g.sub(w * proj)
    g_norm = g.norm(2)
    g_orth_norm = g_orth.norm(2).add_(1e-30)
    g_orth_scaled = g_orth * (g_norm / g_orth_norm)
    return g_orth_scaled.view(original_shape).to(original_dtype)

def _orthogonalize_lora_gradient_pair(p: torch.Tensor, grad: torch.Tensor, pair: torch.Tensor, is_lora_A: bool, eps: float = 1e-30) -> torch.Tensor:
    """
    Projects the LoRA gradient to be orthogonal to the combined weight matrix W = BA.
    Avoids materializing the full d_out x d_in matrix by operating on O(rank^2) metrics.
    """
    original_dtype = grad.dtype
    p_f32 = p.float()
    grad_f32 = grad.float()
    pair_f32 = pair.float()

    if is_lora_A:
        # p is A (rank, d_in), pair is B (d_out, rank)
        # W = B @ A, G = B @ grad
        # M = B^T @ B -> shape: (rank, rank)
        M = pair_f32.mT @ pair_f32

        # <W, W> = Tr(A^T @ M @ A)
        MA = M @ p_f32
        w_norm_sq = torch.sum(p_f32 * MA).add_(eps)

        # <W, G> = Tr(A^T @ M @ grad)
        Mg = M @ grad_f32
        dot_prod = torch.sum(p_f32 * Mg)

    else: 
        # p is B (d_out, rank), pair is A (rank, d_in)
        # W = B @ A, G = grad @ A
        # N = A @ A^T -> shape: (rank, rank)
        N = pair_f32 @ pair_f32.mT

        # <W, W> = Tr(B @ N @ B^T)
        BN = p_f32 @ N
        w_norm_sq = torch.sum(p_f32 * BN).add_(eps)

        # <W, G> = Tr(B @ N @ grad^T)
        gN = grad_f32 @ N
        dot_prod = torch.sum(p_f32 * gN)

    # Calculate the global projection scalar in the combined space
    proj = dot_prod / w_norm_sq

    # Project the gradient in the local parameter space
    g_orth = grad_f32.sub(p_f32 * proj)

    # Magnitude Preservation (matching original OrthoGrad logic)
    g_norm = grad_f32.norm(2)
    g_orth_norm = g_orth.norm(2).add_(eps)
    g_orth_scaled = g_orth * (g_norm / g_orth_norm)

    return g_orth_scaled.to(original_dtype)


def _orthogonalize_gradient_granular(p: torch.Tensor, grad: torch.Tensor, dim: int = 1, eps: float = 1e-30) -> torch.Tensor:
    """
    Projects the gradient `grad` to be orthogonal to the parameter `p` row/col-wise,
    while preserving the original norm of the gradient for each row/col.
    """
    original_dtype = grad.dtype
    p_f32 = p.float()
    grad_f32 = grad.float()

    # Calculate the dot product <p, grad> for each row/col
    dot_prod = torch.sum(p_f32 * grad_f32, dim=dim, keepdim=True)

    # Calculate ||p||^2 for each row/col
    p_norm_sq = torch.sum(p_f32 * p_f32, dim=dim, keepdim=True).add_(eps)

    # Project: g_orth = g - (p * <p, g> / ||p||^2)
    proj = dot_prod / p_norm_sq
    grad_orth = grad_f32 - (proj * p_f32)

    # Magnitude Preservation
    g_norm = torch.norm(grad_f32, p=2, dim=dim, keepdim=True)
    g_orth_norm = torch.norm(grad_orth, p=2, dim=dim, keepdim=True).add_(eps)
    grad_orth_scaled = grad_orth * (g_norm / g_orth_norm)

    return grad_orth_scaled.to(original_dtype)


def _orthogonalize_lora_gradient_pair_granular(p: torch.Tensor, grad: torch.Tensor, pair: torch.Tensor, is_lora_A: bool, eps: float = 1e-30) -> torch.Tensor:
    """
    Projects the LoRA gradient to be orthogonal to the combined weight matrix W = BA,
    applied granularly (row-wise for A, col-wise for B) for each rank dimension.
    """
    original_dtype = grad.dtype
    p_f32 = p.float()
    grad_f32 = grad.float()
    pair_f32 = pair.float()

    if is_lora_A:
        # p is A (rank, d_in), pair is B (d_out, rank) -> Row-wise (dim=1)
        M = pair_f32.mT @ pair_f32  # shape: (rank, rank)

        MA = M @ p_f32
        w_norm_sq = torch.sum(p_f32 * MA, dim=1, keepdim=True).add_(eps)  # shape: (rank, 1)

        Mg = M @ grad_f32
        dot_prod = torch.sum(p_f32 * Mg, dim=1, keepdim=True)  # shape: (rank, 1)

        # Project row-wise
        proj = dot_prod / w_norm_sq
        grad_orth = grad_f32 - (proj * p_f32)

        # Magnitude Preservation (row-wise)
        g_norm = torch.norm(grad_f32, p=2, dim=1, keepdim=True)
        g_orth_norm = torch.norm(grad_orth, p=2, dim=1, keepdim=True).add_(eps)

    else: 
        # p is B (d_out, rank), pair is A (rank, d_in) -> Col-wise (dim=0)
        N = pair_f32 @ pair_f32.mT  # shape: (rank, rank)

        BN = p_f32 @ N
        w_norm_sq = torch.sum(p_f32 * BN, dim=0, keepdim=True).add_(eps)  # shape: (1, rank)

        gN = grad_f32 @ N
        dot_prod = torch.sum(p_f32 * gN, dim=0, keepdim=True)  # shape: (1, rank)

        # Project col-wise
        proj = dot_prod / w_norm_sq
        grad_orth = grad_f32 - (p_f32 * proj)

        # Magnitude Preservation (col-wise)
        g_norm = torch.norm(grad_f32, p=2, dim=0, keepdim=True)
        g_orth_norm = torch.norm(grad_orth, p=2, dim=0, keepdim=True).add_(eps)

    g_orth_scaled = grad_orth * (g_norm / g_orth_norm)
    return g_orth_scaled.to(original_dtype)
