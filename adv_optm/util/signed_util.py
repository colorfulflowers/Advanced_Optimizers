import torch

from . import param_update

def apply_stochastic_sign(update: torch.Tensor, noise: torch.Tensor | None) -> torch.Tensor:
    """
    Applies the Stochastic Sign operator S_R(v).
    Uses uniform noise injection to compute the stochastic sign
    """
    R = update.abs().max().clamp_min(1e-12)

    if noise is None:
        noise = param_update._get_random_noise_for_sso(update)
    return torch.sign(update / R + noise, out=update)
