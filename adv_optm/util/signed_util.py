import torch

from . import param_update

def apply_stochastic_sign_(update: torch.Tensor, noise: torch.Tensor | None) -> torch.Tensor:
    """
    Applies the Stochastic Sign operator S_R(v).
    Uses uniform noise injection to compute the stochastic sign
    """
    if update.dim() >= 2:
        update_abs = update.abs()
        # Calculate row and col maximums
        R_col = update_abs.amax(dim=0, keepdim=True) # Shape: (1, cols)
        R_row = update_abs.amax(dim=1, keepdim=True) # Shape: (rows, 1)
        R = torch.minimum(R_row, R_col)
    else:
        # Fallback for 1D tensors (e.g., biases, layernorm)
        R = update.abs().max()

    # Prevent division by zero
    R = R.clamp_min(1e-12)

    if noise is None:
        noise = param_update._get_random_noise_for_sso(update)

    # Chain inplace operations: torch.sign(update / R + noise)
    return update.div_(R).add_(noise).sign_()
