"""
Temporal causal weighting for PINN training, per Wang, Sankaran, Wang &
Perdikaris (2023), "An Expert's Guide to Training Physics-Informed Neural
Networks", Sec. 5.1 and Algorithm 1, eqs. (2.10)-(2.11).
 
Motivation (paper, Sec 5.1): conventional PINNs minimize the PDE residual
loss at all times simultaneously, which lets the network start fitting later
timesteps before it has correctly learned earlier ones -- a violation of
the physical causality that information should only propagate forward in
time. This module reweights the residual loss by segment so that a later
segment's loss only "counts" once earlier segments are already small.
 
    L_r(theta) = (1/M) * sum_i  w_i * L_r^i(theta),                  (2.10)
    w_i = exp( -eps * sum_{k=1}^{i-1} L_r^k(theta) ),  i = 2..M      (2.11)
    w_1 = 1  (empty sum)
 
w_i is detached from the autograd graph (the paper's lax.stop_gradient):
the causal weights are treated as fixed coefficients at each step, not as
something theta is optimized against directly.
"""

from __future__ import annotations
import torch

def causal_weighted_residual(residual_terms: dict[str, torch.Tensor], weights: dict[str, torch.Tensor], t: torch.Tensor, t_bounds: tuple[float, float], n_chunks: int, eps: float, verbose: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    """Bins already-computed PDE residuals by time and combines them with
    causal weights.
 
    Bins by time rather than recomputing residuals.pde() per chunk: since
    autograd differentiates per-point, slicing a batch of residuals by a
    boolean mask after the fact is equivalent to (and cheaper than) calling
    residuals.pde() separately on each temporal slice.
 
    Args:
        residual_terms: e.g. {'pde0': res0, 'pde1': res1, 'pde2': res2},
            each an (N, 1) tensor of raw (unsquared) residuals with grad
            history intact -- same tensors residuals.pde() returns.
        weights: static per-term weights, e.g. {'pde0': w0, 'pde1': w1, 'pde2': w2},
            matching the w['pde0'] etc. used elsewhere in mse_loss so the
            causal and non-causal paths stay on equal footing.
        t: (N, 1) time coordinate for each collocation point (same ordering
            as the residual tensors).
        t_bounds: (t_min, t_max) of the domain to partition into chunks.
        n_chunks: M in the paper -- number of equal-length temporal segments.
        eps: the causality strength hyper-parameter (paper recommends a
            "moderately large" eps such that all weights converge to 1 by
            the end of training; too small under-enforces causality, too
            large can stall training entirely if early chunks can't be
            minimized to a small enough value -- see Sec 5.1).
 
    Returns:
        total_loss: scalar, ready to be added into the overall loss.
        chunk_losses: (n_chunks,) tensor, detached raw per-chunk loss --
            for plotting L_r(t, theta) (paper Fig 3, top-right / Fig 19 left).
        chunk_weights: (n_chunks,) tensor, detached w_i's -- for the
            min_t w(t) convergence diagnostic (paper Fig 19, right panel);
            training is behaving well once min(chunk_weights) -> 1.
    """

    t_lo, t_hi = t_bounds
    edges = torch.linspace(t_lo, t_hi, n_chunks + 1, device=t.device)
    t_flat = t.squeeze(-1)

    per_chunk_loss_list = []
    for i in range(n_chunks):
        lo = edges[i];  hi = edges[i + 1]
        mask = (t_flat >= lo) & (t_flat < hi) if i < n_chunks - 1 else (t_flat >= lo) & (t_flat <= hi)

        if not mask.any():
            per_chunk_loss_list.append(torch.zeros((), device=t.device, dtype=t.dtype))
            continue

        loss_i = sum(
            weights[name] * res[mask].pow(2).mean() for name, res in residual_terms.items()
            )

        per_chunk_loss_list.append(loss_i)
        if verbose: print(f'per-chunk-loss [chunk: {i}] - [{lo} - {hi}]', per_chunk_loss_list)

    chunk_losses = torch.stack(per_chunk_loss_list)

    with torch.no_grad():
        cum_prev = torch.cumsum(chunk_losses, dim=0) - chunk_losses
        chunk_weights = torch.exp(-eps * cum_prev)

    total_loss = (chunk_weights * chunk_losses).mean()

    return total_loss, chunk_losses.detach(), chunk_weights
