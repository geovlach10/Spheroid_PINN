"""Trastuzumab-specific evaluation helpers: builds a generic
pinnpy.evaluation.Evaluator from a raw FDM solve, and a quick-look
plot of the free-antibody spatial distribution over time.
"""

from __future__ import annotations
import numpy as np
import torch

from pinnpy.evaluation import Evaluator
from pinnpy.pinns import PINN
from . import constants as _CST

SPECIES = ('c0', 'c1', 'c2')


def build_evaluator(fdm_x: np.ndarray, fdm_sol, n_t: int = 101, device: str = 'cpu') -> Evaluator:
    """Builds an Evaluator for trastuzumab's 3-species uptake/clearance
    model from a raw FDM solve.

    Args:
        fdm_x: spatial nodes the FDM solver used, shape (N,) -- from
            `fdm.get_diffusion_differential_operator`'s return.
        fdm_sol: the FDM solve's `scipy.integrate.OdeSolution` (dense
            output), as returned by `fdm.run_fdm`.
        n_t: how many timestamps to sample from `fdm_sol` across [0, 1]
            (normalized time) for comparison.
        device: where to evaluate the PINN's network.

    Returns:
        A generic Evaluator, pre-configured with species=('c0','c1','c2')
        and the FDM solution split into per-species (N, n_t) arrays.
    """
    N = fdm_x.shape[0]
    t_grid = np.linspace(0.0, 1.0, n_t)
    Y = fdm_sol(t_grid)   # (3N, n_t)

    dfm_sol = {
        'c0': Y[:N, :],
        'c1': Y[N:2*N, :],
        'c2': Y[2*N:, :],
    }

    return Evaluator(
        species=SPECIES,
        reference_sol=dfm_sol,
        r=fdm_x,
        t_grid=t_grid,
        primary_species='c0',
        device=device,
    )


def get_spatial_antibody_distribution(t_exp, pinn: PINN, L):
    """Quick-look plot: free-antibody (c0) concentration vs. r, at five
    evenly spaced timesteps from 0 to `t_exp`.

    Args:
        t_exp: end time of the experiment (normalized units -- plotted
            timestamps are shown in hours, `time * 24`).
        pinn: a trained PINN. Its `.net` is called directly and its
            output sliced for column 0 -- works whether `.net` is a
            plain backbone or a ConstrainedNet, since both return a
            single (N, 3) tensor.
        L: receptor-load scaling factor, used only in the plot title
            (`L * R_T`) -- not passed to the network.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 9))
    r = torch.linspace(0, 1, 100).reshape(-1, 1)
    for time in [0, 0.25 * t_exp, 0.5 * t_exp, 0.75 * t_exp, t_exp]:
        t = torch.full_like(r, time)
        with torch.no_grad():
            pred = pinn.net(r, t)
        c0 = pred[:, 0:1]
        plt.plot(r, c0, label=f't={time * 24}h')
    plt.title(f'spatial distribution of TRM || time: [0 - {t_exp * 24}h] - R_T: {L * _CST.R_T}')
    plt.xlabel(f'r')
    plt.ylabel(f'[Ab_I]/[C_reference]')
    plt.legend()
    plt.show()