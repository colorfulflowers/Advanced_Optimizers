import torch

import math

@torch.no_grad()
def _lora_newton_schulz_iteration(
    X: torch.Tensor,
    pair: torch.Tensor,
    is_lora_A: bool,
    steps: int = 5,
    eps: float = 1e-7,
    coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    cns: bool = False,
    cns_a_bound: float = 1e-4,
    spectral_normalization: bool = False,
) -> torch.Tensor:
    """
    Computes efficient Newton-Schulz orthogonalization directly in the full combined weight space
    of the two LoRA factors without ever materializing the `d_out * d_in` full product.
    Operates on O(r^2) metrics. 
    """
    dtype = X.dtype
    X_f = X.to(torch.float32)
    pair_f = pair.to(torch.float32)

    if is_lora_A:
        M_L = pair_f.mT @ pair_f # (r, r)
        M_R = X_f @ X_f.mT # (r, r)
    else:
        M_L = X_f.mT @ X_f # (r, r)
        M_R = pair_f @ pair_f.mT # (r, r)

    r = M_L.size(0)

    # Frobenius norm squared of the combined matrix L @ R is exactly Tr(M_L @ M_R)
    frob_norm_sq = torch.sum(M_L * M_R).clamp_min_(1e-12)
    frob_norm = frob_norm_sq.sqrt()

    if spectral_normalization:
        scale = 1.0 / (frob_norm + eps)
    else:
        scale = 1.0 / frob_norm.clamp_min(eps)

    # Scale inner component matrices so the synthesized combined matrix receives `scale`
    M_L.mul_(scale)
    M_R.mul_(scale)

    P = torch.eye(r, device=X.device, dtype=torch.float32)

    if cns:
        lower_bound = cns_a_bound
        upper_bound = 1.0
        for _ in range(steps):
            lb, ub = lower_bound, upper_bound
            lb_ub = lb * ub
            e_sq = (lb**2 + lb_ub + ub**2) / 3.0
            K = 2.0 * e_sq**1.5
            L = lb_ub * (lb + ub)
            denom = K + L
            alpha = 6.0 / denom
            c1 = alpha * e_sq
            c3 = -alpha / 3.0

            # Construct the combined scaled inner projection
            A_k = P @ M_R @ P.mT @ M_L
            P = c1 * P + c3 * (A_k @ P)

            eps_val = (K - L) / denom
            lower_bound, upper_bound = 1.0 - eps_val, 1.0 + eps_val
    else:
        a, b, c = coeffs
        for _ in range(steps):
            A_k = P @ M_R @ P.mT @ M_L
            A_k_P = A_k @ P
            P = a * P + b * A_k_P + c * (A_k @ A_k_P)

    P.mul_(scale)

    # Project factored updates mapping dynamically onto correct active component
    if is_lora_A:
        return (P.to(dtype) @ X)
    else:
        return (X @ P.to(dtype))

@torch.no_grad()
def _compiled_lora_newton_schulz_iteration(
    X: torch.Tensor,
    pair: torch.Tensor,
    is_lora_A: bool,
    steps: int = 5,
    eps: float = 1e-7,
    coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    cns: bool = False,
    cns_a_bound: float = 1e-4,
    spectral_normalization: bool = False,
) -> torch.Tensor:
    """
    Compiled version of LoRA Newton-Schulz orthogonalization voiding in-place mutations.
    """
    dtype = X.dtype
    X_f = X.to(torch.float32)
    pair_f = pair.to(torch.float32)

    if is_lora_A:
        M_L = pair_f.mT @ pair_f
        M_R = X_f @ X_f.mT
    else:
        M_L = X_f.mT @ X_f
        M_R = pair_f @ pair_f.mT

    r = M_L.size(0)

    frob_norm_sq = torch.sum(M_L * M_R).clamp_min(1e-12)
    frob_norm = frob_norm_sq.sqrt()

    if spectral_normalization:
        scale = 1.0 / (frob_norm + eps)
    else:
        scale = 1.0 / frob_norm.clamp_min(eps)
    
    M_L = M_L * scale
    M_R = M_R * scale

    P = torch.eye(r, device=X.device, dtype=torch.float32)

    if cns:
        lower_bound = cns_a_bound
        upper_bound = 1.0
        for _ in range(steps):
            lb, ub = lower_bound, upper_bound
            lb_ub = lb * ub
            e_sq = (lb**2 + lb_ub + ub**2) / 3.0
            K = 2.0 * e_sq**1.5
            L = lb_ub * (lb + ub)
            denom = K + L
            alpha = 6.0 / denom
            c1 = alpha * e_sq
            c3 = -alpha / 3.0
            
            A_k = P @ M_R @ P.mT @ M_L
            P = c1 * P + c3 * (A_k @ P)
            
            eps_val = (K - L) / denom
            lower_bound, upper_bound = 1.0 - eps_val, 1.0 + eps_val
    else:
        a, b, c = coeffs
        for _ in range(steps):
            A_k = P @ M_R @ P.mT @ M_L
            A_k_P = A_k @ P
            P = a * P + b * A_k_P + c * (A_k @ A_k_P)

    P = P * scale

    if is_lora_A:
        return (P.to(dtype) @ X)
    else:
        return (X @ P.to(dtype))

@torch.no_grad()
def _newton_schulz_iteration(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    cns: bool = False,
    cns_a_bound: float = 1e-4,
    spectral_normalization: bool = False,
) -> torch.Tensor:
    """
    Performs the Newton-Schulz iteration to find the nearest orthogonal matrix.
    This is the core computation of the Muon optimizer.

    Some optimizations inspired from:
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/muon.py#L79

    Args:
        G (torch.Tensor): The 2D input matrix (momentum-accumulated gradient).
        steps (int): The number of iterations to run.
        eps (float): Small constant for numerical stability during normalization.
        coeffs (Union[Tuple[float, float, float], List[Tuple[float, float, float]]]):
            The (a, b, c) coefficients for the quintic polynomial update.
        cns (bool): If True, enables Chebyshev-accelerated Newton-Schulz (CANS)
            using an iterative 3rd-order polynomial with optimal coefficients
            derived at each step.
        cns_a_bound (float): The initial lower bound for singular values when
            using CANS. The upper bound is assumed to be 1.0 after normalization.
    Returns:
        torch.Tensor: The orthogonalized matrix.
    """
    assert G.ndim in (2, 3), f"Input must be 2D or 3D, got {G.ndim}D"

    a, b, c = coeffs

    X = G.to(torch.bfloat16)

    # Transpose if needed
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT

    # Normalize spectral norm to at most 1
    if spectral_normalization:
        X.div_(X.norm(dim=(-2, -1), keepdim=True).add_(eps))
    else:
        X.div_(X.norm(dim=(-2, -1), keepdim=True).clamp_min_(eps))

    # Select matrix multiplication function based on dimension (Batched vs Standard)
    mm_fn = torch.baddbmm if X.ndim > 2 else torch.addmm

    # Pre-allocate for performance
    X = X.contiguous()
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    # Allocating B and C for standard NS to avoid loop allocations
    # We also reuse C for CNS updates to be efficient
    C = torch.empty_like(X)
    if not cns:
        B = torch.empty_like(A)

    if cns:
        # Chebyshev-accelerated Newton-Schulz (CANS) from
        # "Accelerating Newton-Schulz Iteration for Orthogonalization via Chebyshev-type Polynomials"
        # This implements the iterative scheme from Algorithm 1, using the
        # closed-form 3rd-order polynomial from Proposition 2.
        # Note: CANS calculates its own coefficients dynamically, ignoring `coeffs`
        lower_bound = cns_a_bound
        upper_bound = 1.0  # Matrix is normalized, so largest singular value is approx 1.

        for _ in range(steps):
            # Calculate optimal 3rd-order coefficients c1, c3 for p(x) = c1*x + c3*x^3
            # based on the current singular value bounds [lower_bound, upper_bound].
            # Formulas are derived from Proposition 2 and its proof in Appendix B of the paper.
            lb, ub = lower_bound, upper_bound
            lb_ub = lb * ub

            # Calculate Mean Square Error term
            e_sq = (lb*lb + lb_ub + ub*ub) / 3.0

            # Calculate components for alpha and bounds update
            # K is the error scaling component
            # L is the bound interaction component
            K = 2.0 * e_sq**1.5
            L = lb_ub * (lb + ub)

            denom = K + L

            # Calculate alpha, which scales the polynomial
            alpha = 6.0 / denom

            c1 = alpha * e_sq
            c3 = -alpha / 3.0

            # Apply the 3rd-order Newton-Schulz update
            # A = X @ X.mT
            mm_fn(A, X, X.mT, beta=0.0, alpha=1.0, out=A)
            # X = c1 * X + c3 * (A @ X)
            mm_fn(X, A, X, beta=c1, alpha=c3, out=C)
            X, C = C, X

            # Update the singular value bounds for the next iteration based on the error
            eps_val = (K - L) / denom
            lower_bound, upper_bound = 1.0 - eps_val, 1.0 + eps_val
    else:
        # Standard Quintic Newton-Schulz
        for _ in range(steps):
            # A = X @ X.mT
            mm_fn(A, X, X.mT, beta=0.0, alpha=1.0, out=A)
            # B = b * A + c * (A @ A)
            mm_fn(A, A, A, beta=b, alpha=c, out=B)
            # X = a * X + B @ X
            mm_fn(X, B, X, beta=a, alpha=1.0, out=C)
            X, C = C, X  # swap refs to avoid copy

    # Transpose back if necessary
    if transposed:
        X = X.mT

    return X.to(G.dtype)

@torch.no_grad()
def _compiled_newton_schulz_iteration(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    cns: bool = False,
    cns_a_bound: float = 1e-4,
    spectral_normalization: bool = False,
) -> torch.Tensor:
    """
    Newton-Schulz iteration refactored for torch.compile compatibility.
    Removes mutable buffers and in-place operations in favor of functional graph construction.
    """
    assert G.ndim in (2, 3), f"Input must be 2D or 3D, got {G.ndim}D"

    a, b, c = coeffs

    X = G.to(torch.bfloat16)

    # Transpose if needed
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT

    # Normalize spectral norm to at most 1
    if spectral_normalization:
        X.div_(X.norm(dim=(-2, -1), keepdim=True).add_(eps))
    else:
        X.div_(X.norm(dim=(-2, -1), keepdim=True).clamp_min_(eps))

    if cns:
        # Chebyshev-accelerated Newton-Schulz (CANS)
        lower_bound = cns_a_bound
        upper_bound = 1.0

        for _ in range(steps):
            lb, ub = lower_bound, upper_bound
            lb_ub = lb * ub
            # Calculate Mean Square Error term
            e_sq = (lb**2 + lb_ub + ub**2) / 3.0

            # Calculate components for alpha and bounds update
            K = 2.0 * e_sq**1.5
            L = lb_ub * (lb + ub)
            denom = K + L
            alpha = 6.0 / denom

            c1 = alpha * e_sq
            c3 = -alpha / 3.0

            # Apply the 3rd-order Newton-Schulz update
            A = X @ X.mT
            X = c1 * X + c3 * (A @ X)

            # Update the singular value bounds for the next iteration based on the error
            eps_val = (K - L) / denom
            lower_bound, upper_bound = 1.0 - eps_val, 1.0 + eps_val

    else:
        # Standard Quintic Newton-Schulz
        # Update: X = a*X + b*(A@X) + c*(A@A@X)
        for _ in range(steps):
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X

    # Transpose back if necessary
    if transposed:
        X = X.mT

    return X.to(G.dtype)

@torch.no_grad()
def newton_schulz(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    cns: bool = False,
    cns_a_bound: float = 1e-4,
    low_rank_ortho: bool = False,
    ortho_rank: int = 128,
    spectral_normalization: bool = False,
    compiled: bool = False,
    p: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Public entry point for Muon orthogonalization.
    Handles either full Newton-Schulz or Low-Rank Orthogonalization via Gaussian Sketching.
    Source: "Low-rank Orthogonalization for Large-scale Matrix Optimization with Applications
    to Foundation Model Training" (https://arxiv.org/abs/2509.11983)

    Args:
        G (torch.Tensor): Input matrix (gradient/update).
        steps (int): NS iterations.
        eps (float): Numerical stability epsilon.
        coeffs (tuple): Polynomial coefficients.
        cns (bool): Use Chebyshev-accelerated Newton-Schulz.
        cns_a_bound (float): CANS lower bound.
        low_rank_ortho (bool): Whether to project to low rank before orthogonalizing.
        ortho_rank (int): Rank for low-rank projection.
    """
    # Guard trigger: Apply pair-aware LoRA Full-Weight Orthogonalization matching if valid parameter is passed.
    if p is not None:
        is_lora_A = getattr(p, '_is_lora_A', False)
        is_lora_B = getattr(p, '_is_lora_B', False)
        if is_lora_A or is_lora_B:
            pair = getattr(p, '_lora_pair', None)
            if pair is not None:
                # Same explicit zero-initialization gating applied natively in scaled_optm.py for stabilization
                lora_ns_fn = _compiled_lora_newton_schulz_iteration if compiled else _lora_newton_schulz_iteration
                return lora_ns_fn(
                    G,
                    pair=pair.detach(),
                    is_lora_A=is_lora_A,
                    steps=steps,
                    eps=eps,
                    coeffs=coeffs,
                    cns=cns,
                    cns_a_bound=cns_a_bound,
                    spectral_normalization=spectral_normalization
                )

    if compiled:
        ns_fn = _compiled_newton_schulz_iteration
    else:
        ns_fn = _newton_schulz_iteration

    if low_rank_ortho:
        # Low-Rank Orthogonalization via Gaussian Sketching
        M = G
        r = min(ortho_rank, M.shape[0], M.shape[1])

        if r > 0:
            # 1. Sketch the matrix
            G_sketch = torch.randn(M.shape[1], r, device=M.device, dtype=M.dtype)
            MG = M @ G_sketch

            # 2. QR decomposition to get orthogonal basis Q
            # Handle dtype mismatch for QR if necessary
            if MG.dtype != torch.float32:
                MG_dtype = M.dtype
                Q, _ = torch.linalg.qr(MG.float())
                Q = Q.to(MG_dtype)
            else:
                Q, _ = torch.linalg.qr(MG)

            # 3. Project M onto the basis
            projected_M = Q.T @ M

            # 4. Orthogonalize the smaller projected matrix
            ortho_projected_M = ns_fn(
                projected_M,
                steps=steps,
                eps=eps,
                coeffs=coeffs,
                cns=cns,
                cns_a_bound=cns_a_bound,
                spectral_normalization = spectral_normalization
            )

            # 5. Project back to the original space
            return Q @ ortho_projected_M

    # Standard Path
    return ns_fn(
        G,
        steps=steps,
        eps=eps,
        coeffs=coeffs,
        cns=cns,
        cns_a_bound=cns_a_bound,
        spectral_normalization=spectral_normalization
    )

def _is_suitable_for_muon(
        param: torch.Tensor,
        min_dim_size: int = 4,
        max_aspect_ratio: float = 128.,
) -> bool:
    """Check if a parameter is suitable for Muon optimization.
    modified from:
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/muon.py#L167
    """

    s = param.shape
    # Must have at least 2 non-unit dimensions
    if param.ndim < 2 or sum(1 for dim_size in s if dim_size > 1) < 2:
        return False

    # Unit dimension in first two positions indicates:
    # - Position embeddings (1, seq, dim)
    # - Depthwise convs (out, 1, h, w)
    # - Other degenerate cases possibly not caught by first rule
    if s[0] == 1 or s[1] == 1:
        return False

    if param.ndim >= 3:
        # For 3D+ tensors, check what dimensions will be AFTER flattening
        # since that's what gets passed to Newton-Schulz iteration
        # Flatten mode: (out, in, *spatial) -> (out, in * spatial_prod)
        out_ch = s[0]
        in_ch_with_spatial = 1
        for d in s[1:]:
            in_ch_with_spatial *= d
        check_dims = (out_ch, in_ch_with_spatial)
    else:
        # For 2D tensors, check as-is
        check_dims = s

    # Both dims should be >= minimum size
    min_size = min(check_dims)
    if min_size < min_dim_size:
        return False

    # Aspect ratio shouldn't be too extreme
    max_size = max(check_dims)
    aspect_ratio = max_size / min_size
    if aspect_ratio > max_aspect_ratio:
        return False

    return True

def approx_mars(current_grad: torch.Tensor, last_grad: torch.Tensor, mars_gamma:float, beta1:float, Simplified_AdEMAMix:bool=False):
    """
    The approximated version of MARS-M, proposed in the paper: "MARS-M: When Variance Reduction
    Meets Matrices" (https://arxiv.org/abs/2510.21800). A variance reduction technique that
    incorporates the changes in gradients into the momentum gradient.
    Formula: c_t = g_t + gamma * beta / (1 - beta) * (g_t - g_{t-1}
    """
    if Simplified_AdEMAMix:
        mars_factor = mars_gamma * beta1
    else:
        mars_factor = mars_gamma * beta1 / (1.0 - beta1)
    # Compute corrected gradient c_t
    # c_t = current_grad + mars_factor * (current_grad - last_grad)
    correction = current_grad.sub(last_grad).mul_(mars_factor).add_(current_grad)
    # Update last_grad to current grad for the next step
    last_grad.copy_(current_grad)
    # Use correction as the gradient for subsequent momentum updates
    return correction

def normuon_update(update: torch.Tensor, v_t: torch.Tensor, beta2, eps):
    """
    The scalar state update of NorMuon variant, proposed in the paper: "NorMuon: Making Muon more
    efficient and scalable" (https://arxiv.org/abs/2510.05491). Implement a row-wise normalization via
    2nd moment estimation to balance parameter utilization and retain Muon conditioning.
    """
    # Update 2nd moment estimate
    mean_squared_update = torch.mean(update.square(), dim=1, dtype=v_t.dtype)
    v_t.lerp_(mean_squared_update, 1 - beta2)
    # Normalize update
    del mean_squared_update
    return update.div_(v_t.sqrt().unsqueeze_(1).add_(eps))

def rms_adjustment(update: torch.Tensor, rms_rescaling: bool, lr):
    if rms_rescaling: # RMS-aligned rescaling
        # This is slower due to norm calculations but it worked the best for t2i models.
        rms_target = 0.2 # default (Adam) value for RMS
        update_norm = torch.linalg.vector_norm(update)
        return update.mul_(lr * rms_target * (math.sqrt(update.numel())) / update_norm.clamp_min_(1e-8))
    else:
        # Original Muon scaling
        r, c = update.size(-2), update.size(-1)
        scaling_factor = math.sqrt(max(1, r / c))
        return update.mul_(lr * scaling_factor)

def _auto_projection_for_adamuon(raw_update: torch.Tensor, kappa_p: float) -> torch.Tensor:
    """
    Inspired from the paper "Lion Secretly Solves Constrained Optimization,
    As Lyapunov Predicts". (https://arxiv.org/abs/2310.05898)

    The core finding of the Lion-K paper is that the optimal "projection"
    depends on the geometry of the parameters:
    - Linear Layers / Transformers (p=1.0): These weights often benefit from
    coordinate-wise uniformity. The "Sign" update (standard Lion/AdaMuon) works
    best here because it treats every neuron/channel as equally important.
    - Convolutional Layers / UNet (p=2.0): These weights often possess rotational
    invariance. A hard "Sign" update distorts the direction of the gradient vector
    in 4D space (Batch, Channel, H, W). A "Spherical" update (p=2) preserves the
    direction while normalizing the magnitude.

    We take those findings and apply it to AdaMuon raw update.
    """
    EPS = 1e-12
    x = raw_update
    p = kappa_p

    # Standard (p=1) - sign update
    if p == 1.0:
        return x.sign_()

    # Spherical (p=2) - rotation invariant
    if p == 2.0:
        # Normalize (L2=1)
        # We skip this, since _newton_schulz_iteration will normalize it.
        # norm = x.norm(p=2).clamp_min_(EPS)
        # x.div_(norm)
        return x

    # General p case - hybrid optimizer
    # Calculate the 'Direction' Numerator: sign(x) * |x|^(p-1)
    num = x.sign() * x.abs().pow_(p - 1)

    # Denominator: ||x||_p^(p-1)
    den = x.norm(p=p).pow_(p - 1).clamp_min_(EPS)
    return num.div_(den)


def get_spectral_scaling(p, shape: torch.Size, n_layers: int):
    """
    From the paper:
    "Hyperparameter Transfer Enables Consistent Gains of Matrix-Preconditioned Optimizers Across Scales"
    Calculates the scaling factors based on the paper's rules.
    Assumes shape is (d_out, d_in).

    Returns:
        ns_eps: Damping for Newton-Schul.
        adaptive_eps: Epsilon for AdaMuon/NorMuon denominator.
        spectral_target: Target spectral norm
        wd_scale: Weight decay scale
    """
    d_out, d_in = shape[0], shape[1]

    # Handle Convolutional/Flattened tensors
    if len(shape) > 2:
        d_in = shape[1:].numel()

    # Overwrite with full layer dimensions if part of a LoRA pair
    pair = getattr(p, '_lora_pair', None)
    if pair is not None:
        if getattr(p, '_is_lora_B', False):
            d_out = p.shape[0]
            d_in  = pair.numel() // pair.shape[0]
        else:  # _is_lora_A
            d_in  = p.numel() // p.shape[0]
            d_out = pair.shape[0]

    # Scaling for Epsilon (Table 2)
    L = max(1, n_layers)

    # A) Newton-Schulz Damping
    # This ensures the matrix orthogonalization is stable across scales.
    # Formula: (1/L) * sqrt(d_in / d_out)
    ns_eps = (1.0 / L) * math.sqrt(d_in / d_out)

    # B) Adaptive Denominator Epsilon
    # This ensures the Adam-style division doesn't explode or vanish.
    # Formula: (1/L) * (1 / sqrt(d_in * d_out))
    adaptive_eps = (1.0 / L) * (1.0 / math.sqrt(d_in * d_out))

    # Spectral Target (Section F) -> sqrt(d_out/d_in)
    spectral_target = math.sqrt(d_out / d_in)

    # Weight Decay (Section 3.4) -> 1/width
    wd_scale = 1.0 / d_in

    return ns_eps, adaptive_eps, spectral_target, wd_scale
