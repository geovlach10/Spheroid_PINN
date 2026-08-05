import numpy as np
import torch
import matplotlib.pyplot as plt
from ..evaluation import Evaluator
from ..neural_nets import FCNN
from ..constants import *

def get_spatial_antibody_distribution(t_exp, model_name: str, net: FCNN, L):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 9))
    r = torch.linspace(0, 1, 100).reshape(-1, 1)
    for time in [0, 0.25 * t_exp, 0.5 * t_exp, 0.75 * t_exp, t_exp]:
        t = torch.full_like(r, time)
        with torch.no_grad():
            c0, _, _ = net(r, t)
        plt.plot(r, c0, label=f't={time * 24}h')
    plt.title(f'{model_name}\nspatial distribution of TRM || time: [0 - {t_exp * 24}h] - R_T: {L * R_T}')
    plt.xlabel(f'r')
    plt.ylabel(f'[Ab_I]/[C_reference]')
    plt.legend()
    plt.show()

def plot_comparison(results: dict, evaluator: Evaluator, models: dict, species='c0'):
    """results: from Evaluator.compare. Rows = models, cols = [overlay, error heatmap]."""
    labels = list(results.keys())
    fig, axes = plt.subplots(len(labels), 2, figsize=(14, 5*len(labels)), squeeze=False)

    snap_js = [int(round(ts*(len(evaluator.t_grid)-1))) for ts in evaluator.SNAPSHOTS]

    for row, label in enumerate(labels):
        res = results[label]
        r = res['r']
        fdm_field = evaluator.fdm[species]                  # (N, n_t)
        pinn_field = fdm_field + res['error_field'][species]  # reconstruct pinn = fdm + err

        # left: overlay at snapshots
        ax = axes[row][0]
        for j, ts in zip(snap_js, evaluator.SNAPSHOTS):
            line, = ax.plot(r, fdm_field[:, j], '--', alpha=0.6)
            ax.plot(r, pinn_field[:, j], color=line.get_color(),
                    label=f't={ts:.2f}')
        ax.set_title(f'{label}: PINN (solid) vs FDM (dashed) — {species}')
        ax.set_xlabel('r'); ax.set_ylabel(species); ax.legend(fontsize=8)

        # right: error heatmap
        ax = axes[row][1]
        im = ax.pcolormesh(evaluator.t_grid, r, res['error_field'][species],
                           shading='auto', cmap='RdBu_r')
        ax.set_title(f"{label}: error (global L2={res['global_l2']:.2e})")
        ax.set_xlabel('t'); ax.set_ylabel('r')
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()