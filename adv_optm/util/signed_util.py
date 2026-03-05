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