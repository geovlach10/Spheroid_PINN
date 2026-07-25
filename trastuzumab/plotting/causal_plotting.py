"""Diagnostic plots for temporal causal weighting (Wang, Sankaran, Wang &
Perdikaris 2023, Sec 5.1). Two views:
 
- plot_chunk_diagnostics: a single-snapshot look at L_r^i(theta) and w_i
  across temporal chunks, right now, from whatever's currently sitting in
  pinn.meta['chunk_losses'] / ['chunk_weights'] (populated each time
  mse_loss() runs with pinn.causal=True).
 
- plot_causal_history: the full-training-run version (paper Fig 19 style)
  -- per-chunk loss evolution as a heatmap, and the min_t w(t) convergence
  trace -- built from a history you accumulate during training (see the
  usage note in plot_causal_history's docstring for how to collect it).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import torch

from ..pinns import BasePinn

def plot_chunk_diagnostics(pinn: BasePinn, title: str | None):

    """Plot per-time-chunk PDE residual loss and causal weight at the
    current training step.
 
    Requires pinn.causal=True and at least one prior call to pinn.mse_loss()
    so that pinn.meta['chunk_losses'] / ['chunk_weights'] are populated.
    """

    if ('chunk_losses' not in pinn.meta) or ('chunk_weights' not in pinn.meta):
        raise RuntimeError(
            "No causal chunk diagnostics found in pinn.meta -- make sure "
            "pinn.causal=True and mse_loss() has been called at least once."
        )

    chunk_losses = pinn.meta['chunk_losses'].detach().cpu().numpy()
    chunk_weights = pinn.meta['chunk_weights'].detach().cpu().numpy()
    n_chunks = len(chunk_losses)
 
    t_lo, t_hi = pinn.lower_bounds[1], pinn.upper_bounds[1]
    edges = np.linspace(t_lo, t_hi, n_chunks + 1)
    centers = (edges[:-1] + edges[1:]) / 2
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
 
    axes[0].plot(centers, chunk_losses, marker='o')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('t (normalized)')
    axes[0].set_ylabel(r'per-chunk PDE residual loss $L_r^i(\theta)$')
    axes[0].set_title('Residual loss per temporal chunk')
    axes[0].grid(True, linestyle='--', alpha=0.4)
 
    axes[1].plot(centers, chunk_weights, marker='o', color='tab:orange')
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].axhline(1.0, color='gray', linestyle=':', linewidth=1)
    axes[1].set_xlabel('t (normalized)')
    axes[1].set_ylabel(r'causal weight $w_i$')
    axes[1].set_title('Causal weight per temporal chunk')
    axes[1].grid(True, linestyle='--', alpha=0.4)
 
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()