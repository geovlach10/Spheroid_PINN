"""Generic PINN-vs-reference-solution evaluation.

Evaluator scores a trained PINN's network against a reference solution
(e.g. an FDM solve, or any other ground truth sampled on the same
(r, t) grid) via relative L2 error -- globally, per species, and at
fixed time snapshots. Species-count-agnostic: works for any number of
solution species, not a fixed 3.

This module knows nothing about any specific PDE or solver -- the
reference data (`fdm`) and grid (`r`, `t_grid`) are supplied by the
caller, already computed. See trastuzumab/evaluation.py for a
trastuzumab-specific convenience constructor that builds an Evaluator
from a raw FDM solve.

Example:
    evaluator = Evaluator(
        species=('u',),
        fdm={'u': fdm_solution_array},   # shape (N, n_t)
        r=spatial_nodes,                  # shape (N,)
        t_grid=time_grid,                 # shape (n_t,)
    )
    scores = evaluator.score(my_pinn)
"""

from __future__ import annotations
import numpy as np
import torch
from .pinns import PINN


class Evaluator:
    """Scores a trained PINN's network against reference data, at
    matching (r, t) grid points, via relative L2 error -- globally, per
    species, and at fixed time snapshots.

    Args:
        species: names of the solution's output channels/species, in
            the same column order the network's forward pass produces
            them (e.g. `('c0', 'c1', 'c2')`). Drives every per-species
            comparison below -- any number of species works.
        fdm: reference solution, one array per species in `species`,
            each shape `(N, n_t)` -- `N` matching `len(r)`, `n_t`
            matching `len(t_grid)`. Named `fdm` for historical reasons
            (finite-difference-method oracle), but any reference
            solution sampled on the same grid works.
        r: spatial nodes the reference solution was computed at, shape
            `(N,)`.
        t_grid: time points the reference solution was computed at
            (normalized, typically `[0, 1]`), shape `(n_t,)`.
        snapshots: normalized time fractions (within `[0, 1]`, matching
            `t_grid`'s range) to report per-snapshot error at. Default:
            5 evenly spaced points across `[0, 1]`, including both
            endpoints.
        primary_species: which entry of `species` `per_snapshot` scores
            (per-snapshot error is reported for one species at a time,
            not all of them). Default: `species[0]`.
        device: where to evaluate the PINN's network.
    """
    
    def __init__(self, species: tuple[str, ...], reference_sol: dict[str, np.ndarray], 
                 r: np.ndarray, t_grid: np.ndarray, snapshots: tuple[float, ...] | None = None, 
                 primary_species: str | None = None, device: str = 'cpu'):
        '''
        fdm_x   : np.ndarray(N,) spatial nodes tetracted from get_diffucion_differential_operator
        fdm_sol : scipy OdeSolution 
        '''
        self.species = species
        self.fdm = reference_sol
        self.device = device
        self.r = torch.tensor(r, dtype=torch.float32).view(-1, 1)
        self.t_grid = t_grid
        self.snapshots = snapshots if snapshots is not None else tuple(np.linspace(0.0, 1.0, 5))
        self.primary_species = primary_species if primary_species is not None else species[0]

        for s in self.species:
            assert s in self.fdm, f"species {s!r} missing from fdm dict (got keys: {list(self.fdm)})"
            assert self.fdm[s].shape == (len(r), len(t_grid)), f"fdm[{s!r}].shape {self.fdm[s].shape} != (len(r), len(t_grid)) = ({len(r)}, {len(t_grid)})"


    def _pinn_field(self, model: PINN) -> dict[str, np.ndarray]:
        """Evaluates `model.net` at every (r, t) point in `self.t_grid`
        x `self.r`, returning each species as an (N, n_t) array shaped
        to match `self.fdm`'s layout exactly.

        Assumes `model.net`'s forward pass returns a single tensor
        whose columns are ordered exactly like `self.species`."""
        pred = {s: [] for s in self.species}
        net = model.net
        for time in self.t_grid:
            t = torch.full_like(self.r, float(time)).to(self.device)
            with torch.no_grad():
                out = net(self.r.to(self.device), t)
            for i, s in enumerate(self.species):
                pred[s].append(out[:, i:i+1].cpu().numpy().ravel())
        return  {s: np.stack(v, axis=1) for s, v in pred.items()}
    
    @staticmethod
    def _rel_l2(pred, true):
        """Relative L2: ||pred-true|| / ||true||, guarded against zero norm."""
        denom = np.linalg.norm(true)
        return float(np.linalg.norm(pred - true) / denom) if denom > 0 else float('nan')

    def score(self, model: PINN) -> dict:
        """Scores `model` against the reference solution.

        Args:
            model: a trained PINN (anything with a `.net` whose forward
                pass matches `self.species`' column order).

        Returns:
            dict with:
                global_l2: relative L2 error across all species/times, combined.
                per_species: {species: relative L2} for each entry in `self.species`.
                per_snapshot: {snapshot fraction: relative L2 of `self.primary_species` at that time}.
                error_field: {species: (N, n_t) pred-minus-true array}.
                r, t: the spatial nodes / time grid used, as numpy arrays.
        """
        pred = self._pinn_field(model)

        for s in self.species:
            assert pred[s].shape == self.fdm[s].shape, f'{s}: {pred[s].shape} vs {self.fdm[s].shape}'
        pred_all = np.concatenate([pred[s] for s in self.species], axis=0)
        true_all = np.concatenate([self.fdm[s] for s in self.species], axis=0)

        per_species = {s: self._rel_l2(pred[s], self.fdm[s]) for s in self.species}

        per_snapshot = {}
        for ts in self.snapshots:
            idx = int(round(ts * (len(self.t_grid) - 1)))
            per_snapshot[ts] = self._rel_l2(
                pred[self.primary_species][:, idx],
                self.fdm[self.primary_species][:, idx]
                )

        return {
            'global_l2'      :self._rel_l2(pred_all, true_all),
            'per_species'    :per_species,
            'per_snapshot'   :per_snapshot,
            'error_field'    :{s: pred[s] - self.fdm[s] for s in self.species},
            'r'              :self.r.cpu().numpy().ravel(),
            't'              :self.t_grid
        }

    def compare(self, models: dict):
        """Scores multiple models at once -- the bake-off.

        Args:
            models: {label: model}, e.g. {'FCNN': pinn_a, 'ModifiedMLP': pinn_b}.

        Returns:
            {label: score_dict}, one `score()` result per model.
        """
        return {label: self.score(m) for label, m in models.items()}
    

