import torch

def zero_at_t0(r: torch.Tensor, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Forces output = 0 for every t = 0, by construction (u * t).
    for a nonzero or non-constant IC, write your own hard_constraint_fn
    instead (e.g. `u * t + ic_func(r)` for a non-homogeneous IC)."""
    return t * u