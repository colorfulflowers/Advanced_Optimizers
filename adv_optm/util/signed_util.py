import torch


def inject_error_feedback(raw_update: torch.Tensor, state: dict, group: dict) -> torch.Tensor:
    """
    Injects the stored error directly into the raw true update, 
    bypassing momentum and directly modulating the error.
    """
    if group.get('error_feedback') and 'error_buffer' in state:
        err = state['error_buffer']
        if err.shape != raw_update.shape:
            err = err.view_as(raw_update)
        return raw_update + err
    return raw_update

def update_error_buffer(true_update: torch.Tensor, quantized_update: torch.Tensor, state: dict, group: dict):
    """
    Calculates and stores the error: the difference between the true continuous update 
    and the projected update.
    """
    if group.get('error_feedback'):
        state['error_buffer'] = true_update - quantized_update.view_as(true_update)

def apply_stochastic_sign(update: torch.Tensor, noise: torch.Tensor | None) -> torch.Tensor:
    """
    Applies the Stochastic Sign operator S_R(v).
    Uses uniform noise injection to compute the stochastic sign
    """
    R = update.abs().max().clamp_min(1e-12)

    if noise is None:
        noise = torch.rand_like(update) * 2.0 - 1.0
    return torch.sign(update / R + noise, out=update)
