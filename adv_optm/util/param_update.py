import torch
from torch import Tensor

from typing import Dict, Any

_generators: Dict[torch.device, torch.Generator] = {}


def apply_parameter_update(
    self,
    p: Tensor,
    group: Dict[str, Any],
    update: Tensor,
    lr: float | Tensor,
    wd: float | None = None,
    random_int_tensor: Tensor | None = None,
    decoupled: bool = False,
) -> None:
    """
    Applies decoupled weight decay (standard or cautious) and the final
    parameter update to p in-place.

    Args:
        p: The parameter tensor whose data (p) will be updated.
        group: The parameter group dictionary (must contain "weight_decay").
        update: The pre-calculated update tensor (e.g., scaled gradient or momentum term).
        lr: The current learning rate.
        wd: Optional float value for weight decay, if another value other than group["weight_decay"] is needed.
        random_int_tensor: Optional pre-generated random tensor for stochastic
            rounding. Required for the `torch.compile` path.
        decoupled: Whenever to use the true decoupled weight decay.
    """
    wd = group["weight_decay"] if wd is None else wd
    cautious = group.get('cautious_wd', False)
    if decoupled:
        scaled_wd = wd * (lr / self._init_lr)
    else:
        scaled_wd = wd * lr

    # Compute full update in float32 if using bfloat16 with stochastic rounding
    if p.dtype == torch.bfloat16 and self.stochastic_rounding:
        p_fp32 = p.float()
        update_fp32 = update.float()

        # Apply weight decay if needed
        if wd != 0:
            if cautious:
                # Cautious Weight Decay
                mask = (update_fp32 * p_fp32 >= 0).float()
                p_fp32.addcmul_(p_fp32, mask, value=-scaled_wd)
                del mask
            else:
                # Standard decoupled weight decay
                p_fp32.add_(p_fp32, alpha=-scaled_wd)

        # Apply main update
        p_fp32.add_(-update_fp32)

        # Single stochastic rounding at the end
        if random_int_tensor is not None:
            # Compiled path: use the pre-computed random tensor
            _copy_stochastic_core_(p, p_fp32, random_int_tensor, inplace=True)
            del random_int_tensor
        else:
            # Uncompiled path: generate randoms inside
            copy_stochastic_(p, p_fp32, inplace=True)
        del p_fp32, update_fp32

    else:
        # Standard path for non-bfloat16 or without stochastic rounding
        if wd != 0:
            if cautious:
                # Cautious Weight Decay
                mask = (update * p >= 0).to(p.dtype)
                p.addcmul_(p, mask, value=-scaled_wd)
                del mask
            else:
                # Standard decoupled weight decay
                p.add_(p, alpha=-scaled_wd)

        # Apply main update
        p.add_(-update)

    del update

def set_seed(device: torch.device):
    """
    Initializes or resets the deterministic generator for a specific device.
    This ensures that the sequence of random numbers used for stochastic
    rounding is reproducible.
    """
    global _generators
    if device not in _generators:
        _generators[device] = torch.Generator(device=device)
    _generators[device].manual_seed(42)

def get_generator(device: torch.device) -> torch.Generator:
    """
    Retrieves (and initializes if necessary) the deterministic generator 
    for the specified device.
    """
    if device not in _generators:
        set_seed(device)
    return _generators[device]

def _get_random_int_for_sr(source: Tensor) -> Tensor:
    """
    Generates a random int32 tensor for stochastic rounding.
    This function is not torch.compile-path friendly due to its use of torch.Generator.
    """
    global _generators
    device = source.device

    if device not in _generators:
        set_seed(device)

    # TODO, this is a workaround until torch compile error
    # NotImplementedError: UserDefinedObjectVariable(generator) is fixed
    generator = _generators[device]

    # create a random 16 bit integer
    return torch.randint(
        size=source.shape,
        device=source.device,
        dtype=torch.int32,
        low=0,
        high=(1 << 16),
        generator=generator,
    )

def _get_random_int_for_fp8_sr(source: Tensor) -> Tensor:
    """
    Generates a random int32 tensor for stochastic rounding.
    This function is not torch.compile-path friendly due to its use of torch.Generator.
    """
    global _generators
    device = source.device

    if device not in _generators:
        set_seed(device)

    generator = _generators[device]

    # create a random 20 bit integer (FP32 has 23 mantissa bits, FP8 has 3)
    return torch.randint(
        size=source.shape,
        device=source.device,
        dtype=torch.int32,
        low=0,
        high=(1 << 20),
        generator=generator,
    )

def _copy_stochastic_core_(target: Tensor, source: Tensor, random_int_tensor: Tensor, inplace: bool = False):
    """
    Core logic for BF16 stochastic rounding using a pre-computed random integer tensor.
    This version is designed to be torch.compile-friendly.
    """
    result = random_int_tensor if inplace else random_int_tensor.clone()
    # add the random number to the lower 16 bit of the mantissa
    result.add_(source.view(dtype=torch.int32))

    # mask off the lower 16 bit of the mantissa
    result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

    # copy the higher 16 bit into the target tensor
    target.copy_(result.view(dtype=torch.float32))

def _copy_fp8_stochastic_core_(target: Tensor, source: Tensor, scale: Tensor, random_int_tensor: Tensor, inplace: bool = False):
    """
    Core logic for FP8 stochastic rounding using a pre-computed random integer tensor.
    """
    scaled_source = (source * scale)
    result = random_int_tensor if inplace else random_int_tensor.clone()

    # add the random number to the lower 20 bits of the mantissa
    result.add_(scaled_source.view(dtype=torch.int32))

    # mask off the lower 20 bits of the mantissa
    result.bitwise_and_(-1048576)  # -1048576 = FFF00000 as a signed int32

    # copy the modified tensor into the FP8 target tensor
    target.copy_(result.view(dtype=torch.float32).clamp_(min=-448, max=448).to(torch.float8_e4m3fn))

def copy_stochastic_(target: Tensor, source: Tensor, inplace: bool = False):
    """
    Nerogar's implementation of stochastic rounding in the paper "Revisiting BFloat16 Training"
    (https://arxiv.org/abs/2010.06192). Made deterministic.
    This version is for uncompiled paths; it generates its own random numbers.
    see:
    https://github.com/pytorch/pytorch/issues/120376
    https://github.com/Nerogar/OneTrainer/blob/daae18eaed8c0fa39289b2ff79cc2c1e08577fcb/modules/util/bf16_stochastic_rounding.py

    Args:
        target: the target tensor with dtype=bfloat16
        source: the target tensor with dtype=float32
    """
    random_int_tensor = _get_random_int_for_sr(source)
    _copy_stochastic_core_(target, source, random_int_tensor, inplace)
    del random_int_tensor

def copy_fp8_stochastic_(target: Tensor, source: Tensor, scale: Tensor, inplace: bool = False):
    """
    Stochastic rounding implementation for FP8 e4m3fn states. Made deterministic.
    """
    random_int_tensor = _get_random_int_for_fp8_sr(source)
    _copy_fp8_stochastic_core_(target, source, scale, random_int_tensor, inplace)
    del random_int_tensor

def add_stochastic_(input: Tensor, other: Tensor, alpha: float = 1.0):
    """
    adds other to input using stochastic rounding

    Args:
        input: the input tensor with dtype=bfloat16
        other: the other tensor
        alpha: a multiplier for other
    """
    result = other.clone() if other.dtype == torch.float32 else other.to(dtype=torch.float32)

    result.add_(input, alpha=alpha)
    copy_stochastic_(input, result)
