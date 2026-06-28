import numpy as np
import matplotlib.pyplot as plt
from ..evaluation import Evaluator

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