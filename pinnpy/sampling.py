"""Residual-based Adaptive Refinement with greed (RAR-G) -- Algorithm 1 of
Wu et al., Comput. Methods Appl. Mech. Engrg. 403 (2023) 115671.

RAR-G is a *refinement* scheme, not a resampling one: it never throws the
collocation set away, it only grows it. Each round a fresh dense candidate
cloud is scored by its PDE residual, the worst `m_add` points are appended to
the collocation set, and training resumes. The set is therefore monotonically
increasing and biased toward wherever the network currently violates the PDE
most.

Species-count-agnostic: scoring sums squared residuals across however
many keys pinn.pde.residual_fn returns, rather than assuming a fixed
number of species."""

from __future__ import annotations
import torch

from .datasets import Dataset
from .pinns import PINN
from .trainer import Trainer


def _score(pinn: PINN, canditates: Dataset, **pde_kwargs) -> torch.Tensor:
    """Per-point scalar residual used to rank candidates.

    Returns a 1-D tensor of length len(candidates): the Euclidean norm
    of the residual vector at each point, summed across every species
    key pinn.pde.residual_fn returns (however many there are)."""
    res = pinn.pde.residual_fn(net=pinn.net, dataset=canditates, **pde_kwargs, **pinn.pde.kwargs)
    squared_sum = torch.stack([v**2 for v in res.values()], dim=0).sum(dim=0)
    norm = torch.sqrt(squared_sum)
    return norm.detach().flatten()

def _select(scores: torch.Tensor, canditates: Dataset, m_add: int) -> Dataset:
    '''Greedy top-m selection'''
    m = min(m_add, scores.shape[0])
    _, idx = torch.topk(scores, k=m)
    selected = canditates.data[idx].detach()
    return Dataset(data=selected, name='rar-g', n_points=selected.shape[0])

def rar_g(
    pinn: PINN,
    trainer: Trainer,
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
    """Algorithm 1 (RAR-G), driven against an existing PINN.

    Parameters
    ----------
    pinn          : a constructed PINN. Its `pinn.pde.dataset` is grown in
                    place; IC/BC datasets are left untouched (RAR refines
                    the interior PDE residual, not IC/BC).
    trainer       : a Trainer wrapping `pinn`, used to actually run each
                    training segment (`trainer.train(optimizer, epochs=...)`)
                    -- keeps gradnorm/logging/history working through
                    refinement rounds, not just before/after them.
    optimizer     : an Adam optimizer created ONCE by the caller and reused
                    for every segment, so momentum survives across rounds.
    n_rounds      : maximum refinement rounds (line 9, iteration limit).
    n_dense       : size of the fresh candidate cloud S0 each round.
    m_add         : points appended per round (line 6).
    rnd_epochs    : epochs per refinement segment (line 8).
    warmup_epochs : epochs before the first refinement (line 2, pre-train).
    point_budget  : stop once the collocation set reaches this size (line 9,
                    point-count limit). Pass None to disable and stop on
                    rounds only.
    L             : forwarded as a pde_kwarg to both training
                    (`trainer.train(..., L=L)`) and scoring
                    (`_score(..., L=L)`), so the two can never disagree.
                    Only meaningful if pinn.pde.residual_fn actually
                    accepts an `L` kwarg -- omit/adjust if your PDE
                    doesn't use one.

    Returns
    -------
    list[dict] : per-round log with round index, collocation size, and the
                 max/mean residual on that round's candidate cloud. Useful
                 to confirm the refinement is actually chasing (and
                 reducing) the worst residual.
    """
    history: list[dict] = []

    if warmup_epochs:
        trainer.train(optimizer, epochs=warmup_epochs, L=L)
    if log:
        pinn.check_concentration_profile(epoch=0, title='RAR-G warmup')

    for rnd in range(1, n_rounds + 1):
        canditates = pinn.sampler.sample_collocation_points(
            n_points=n_dense,
            l_bounds=pinn.lower_bounds,
            u_bounds=pinn.upper_bounds,
            seed_offset=rnd,
        )                            

        scores = _score(pinn, canditates=canditates, L=L)
        selected = _select(scores, canditates, m_add)

        pinn.pde.dataset += selected
        if log:
            pinn.pde.dataset.plotme()

        n_now = len(pinn.pde.dataset)
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
            'mean_score': scores.mean().item()
        })
 
        # resume training on the grown set (momentum preserved).
        trainer.train(optimizer, epochs=rnd_epochs, L=L)
        if log:
            pinn.check_concentration_profile(epoch=trainer.current_iter, title=f'RAR-G round {rnd}')

        # Line 9: point-count stop.
        if point_budget is not None and n_now >= point_budget:
            if log:
                print(f'[RAR-G] point budget {point_budget} reached at round {rnd}; stopping.')
            break

    return history