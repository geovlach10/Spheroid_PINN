"""

Residual-based Adaptive Refinement with greed (RAR-G) -- Algorithm 1 of
Wu et al., Comput. Methods Appl. Mech. Engrg. 403 (2023) 115671.
 
RAR-G is a *refinement* scheme, not a resampling one: it never throws the
collocation set away, it only grows it. Each round a fresh dense candidate
cloud is scored by its PDE residual, the worst `m_add` points are appended to
the collocation set, and training resumes. The set is therefore monotonically
increasing and biased toward wherever the network currently violates the PDE
most.

"""

from __future__ import annotations
import torch
from ..trastuzumab import residuals
from .datasets import Dataset
from . import pinns


def _score(net, canditates: Dataset, L:float) -> torch.Tensor:
    """Per-point scalar residual used to rank candidates.
 
    Returns a 1-D tensor of length len(candidates): the Euclidean norm
    ||(res0, res1, res2)|| at each point."""
    res0, res1, res2 = residuals.pde(net=net, dataset=canditates, L=L)
    norm = torch.sqrt(res0 ** 2 + res1 ** 2 + res2 ** 2)
    return norm.detach().flatten()

def _select(scores: torch.Tensor, canditates: Dataset, m_add: int) -> Dataset:
    '''Greedy top-m selection'''
    m = min(m_add, scores.shape[0])
    _, idx = torch.topk(scores, k=m)
    selected = canditates.data[idx].detach()
    return Dataset(data=selected, name='rar-g', n_points=selected.shape[0])

def rar_g(
    pinn: pinns.BasePinn, 
    optimizer: torch.optim.Adam,
    *,
    n_rounds: int = 10,
    n_dense: int = 20000,
    m_add: int = 500,
    rnd_epochs: int = 1000,
    point_budget: int | None = 5000,
    warmup_epochs: int = 20000,
    L: float = 1.0,
    log: bool = True,
) -> list[dict]:
    """Algorithm 1 (RAR-G), driven against an existing Pinn.
 
    Parameters
    ----------
    pinn          : a constructed Pinn. Its `collocation_training_dataset` is
                    grown in place; the initial/center/surface sets are left
                    untouched (RAR refines the interior PDE residual, not the BCs).
    optimizer     : an Adam optimizer created ONCE by the caller and reused for
                    every segment, so momentum survives across rounds. Pinn's
                    train_adam takes the optimizer from outside precisely to make
                    this possible.
    n_rounds      : maximum refinement rounds (line 9, iteration limit).
    n_dense       : size of the fresh candidate cloud S0 each round.
    m_add         : points appended per round (line 6).
    rnd_epochs    : Adam epochs per refinement segment (line 8).
    warmup_epochs : Adam epochs before the first refinement (line 2, pre-train).
    point_budget  : stop once the collocation set reaches this size (line 9,
                    point-count limit). Pass None to disable and stop on rounds only.
    L             : receptor-load scaling, threaded identically into BOTH training
                    and scoring so the two can never disagree.
    lbfgs_epochs  : if > 0, run a single L-BFGS polish after all rounds.
                    train_lbfgs builds a fresh optimizer each call (its history
                    cannot survive segmentation), so it is run once, at the end.
 
    Returns
    -------
    list[dict] : per-round log with round index, collocation size, and the max /
                 mean residual on that round's candidate cloud. Useful to confirm
                 the refinement is actually chasing (and reducing) the worst residual.
    """
    history: list[dict] = []

    pinn.train_adam(optimizer, epochs=warmup_epochs, L=L) if warmup_epochs else None
    pinns.get_spatial_antibody_distribution(pinn.upper_bounds[1], pinn, L=1.0) if log else None

    for rnd in range(1, n_rounds + 1):
        canditates = pinn.sampler.sample_collocation_points(
            n_points=n_dense,
            l_bounds=pinn.lower_bounds,
            u_bounds=pinn.upper_bounds,
            seed_offset=rnd,
        )                            

        scores = _score(net=pinn.net, canditates=canditates, L=L)
        selected = _select(scores, canditates, m_add)

        pinn.collocation_training_dataset += selected
        pinn.collocation_training_dataset.plotme() if log else None   ##plot new dataset...

        n_now = len(pinn.collocation_training_dataset)
        if log:
            print(
                f'[RAR-G] round {rnd:>2} | points={n_now:>5} | '
                f'max_res={scores.max().item():.3e} | '
                f'mean_res={scores.mean().item():.3e}'
            )
        history.append({
            'round': rnd,
            'n_points': n_now,
            'max_score': scores.max().item(),
            'mean_score': scores.mean().item(),
        })
 
        # Line 8: resume training on the grown set (momentum preserved).
        pinn.train_adam(optimizer, epochs=rnd_epochs, L=L)
        pinns.get_spatial_antibody_distribution(pinn.upper_bounds[1], pinn, L=1.0) if log else None

        # Line 9: point-count stop.
        if point_budget is not None and n_now >= point_budget:
            if log:
                print(f'[RAR-G] point budget {point_budget} reached at round {rnd}; stopping.')
            break

    return history