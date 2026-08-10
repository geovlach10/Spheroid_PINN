"""Tests for PINN.loss_fn's handling of multiple InitialConditions and
BoundaryConditions.

Uses a synthetic 2-species toy PDE (species 'a', 'b') rather than
trastuzumab's real residuals -- the point here is to confirm loss_fn's
own bookkeeping (loss-term keys, per-condition dict routing, hard-
constrained zeroing, causal aggregation) is correct in isolation,
independent of any real physics. trastuzumab/residuals.py gets its
own tests once it's rewritten to this contract.
"""

from __future__ import annotations
import pytest
import torch

from pinnpy.pinns import PDE, InitialCondition, BoundaryCondition, ForwardPinn
from pinnpy.datasets import DatasetSampler


# --- toy residual_fn callables -----------------------------------------

def toy_pde_residual(net, dataset, **kwargs) -> dict[str, torch.Tensor]:
    """da/dt = a, db/dt = 2b -- arbitrary, just needs autograd to flow."""
    r = dataset.data.clone()[:, 0:1].requires_grad_(True)
    t = dataset.data.clone()[:, 1:2].requires_grad_(True)
    pred = net(r, t)
    a = pred[:, 0:1]; b = pred[:, 1:2]
    a_t = torch.autograd.grad(a, t, torch.ones_like(a), create_graph=True)[0]
    b_t = torch.autograd.grad(b, t, torch.ones_like(b), create_graph=True)[0]
    return {'a': a_t - a, 'b': b_t - 2 * b}


def toy_ic_residual(net, dataset, ic_func, **kwargs) -> dict[str, torch.Tensor]:
    """Both species should equal ic_func(r) at this IC's dataset."""
    r = dataset.r
    t = dataset.t
    pred = net(r, t)
    a = pred[:, 0:1]; b = pred[:, 1:2]
    target = ic_func(r)
    return {'a': a - target, 'b': b - target}


def toy_bc_residual(net, dataset, target: float = 0.0, **kwargs) -> torch.Tensor:
    """da/dr = target at this BC's dataset (only constrains species 'a')."""
    r = dataset.data.clone()[:, 0:1].requires_grad_(True)
    t = dataset.data.clone()[:, 1:2]
    a = net(r, t)[:, 0:1]
    a_r = torch.autograd.grad(a, r, torch.ones_like(a), create_graph=True)[0]
    return a_r - target


# --- test fixture builder ------------------------------------------------

def _build_pinn(hard_conditions: tuple[str, ...] = (), causal: bool = False, n_chunks: int = 4):
    """Builds a ForwardPinn with 2 species, 2 ICs ('ic1', 'ic2'), and
    2 BCs ('bc_left', 'bc_right'). Point counts are small and fixed so
    n_col/n_initial/n_boundary are easy to assert against."""
    sampler = DatasetSampler(seed=7)

    pde = PDE(
        dataset=sampler.sample_collocation_points(n_points=64),
        residual_fn=toy_pde_residual,
    )
    ic1 = InitialCondition(
        name='ic1',
        dataset=sampler.sample_initial_points(n_points=16),
        ic_func=lambda r: torch.zeros_like(r),
        residual_fn=toy_ic_residual,
    )
    ic2 = InitialCondition(
        name='ic2',
        dataset=sampler.sample_initial_points(n_points=8),
        ic_func=lambda r: torch.ones_like(r),
        residual_fn=toy_ic_residual,
    )
    bc_left = BoundaryCondition(
        name='bc_left',
        dataset=sampler.sample_center_points(n_points=16),
        residual_fn=toy_bc_residual,
        kwargs={'target': 0.0},
    )
    bc_right = BoundaryCondition(
        name='bc_right',
        dataset=sampler.sample_surface_points(n_points=8),
        residual_fn=toy_bc_residual,
        kwargs={'target': 1.0},
    )

    pinn = ForwardPinn(
        pde=pde,
        n_species=2,
        initial_conditions=[ic1, ic2],
        boundary_conditions=[bc_left, bc_right],
        hard_conditions=hard_conditions,
        causal=causal,
        n_chunks=n_chunks,
        seed=7,
    )
    return pinn


_FULL_WEIGHTS = {
    'pde_a': 1.0, 'pde_b': 1.0,
    'ic1_a': 1.0, 'ic1_b': 1.0,
    'ic2_a': 1.0, 'ic2_b': 1.0,
    'bc_left': 1.0, 'bc_right': 1.0,
}


# --- tests -----------------------------------------------------------

def test_loss_term_keys_cover_every_pde_ic_bc_combination():
    pinn = _build_pinn()
    total_loss, terms = pinn.loss_fn(w=_FULL_WEIGHTS)

    expected_keys = {
        'pde_a', 'pde_b',
        'ic1_a', 'ic1_b',
        'ic2_a', 'ic2_b',
        'bc_left', 'bc_right',
    }
    assert set(terms.keys()) == expected_keys


def test_total_loss_equals_sum_of_individual_terms():
    pinn = _build_pinn()
    total_loss, terms = pinn.loss_fn(w=_FULL_WEIGHTS)
    assert torch.isclose(total_loss, torch.stack(list(terms.values())).sum())


def test_raw_loss_terms_match_manual_computation_per_ic():
    """Confirms loss_fn actually routes each IC's own ic_func/dataset
    through its own residual_fn -- not, say, silently reusing ic1's
    target for ic2, or mixing up which dataset belongs to which IC."""
    pinn = _build_pinn()
    pinn.loss_fn(w=_FULL_WEIGHTS)
    raw = pinn.meta['raw_loss_terms']

    ic1, ic2 = pinn.initial_conditions
    for ic in (ic1, ic2):
        pred = pinn.net(ic.dataset.r, ic.dataset.t)
        target = ic.ic_func(ic.dataset.r)
        expected_a = (pred[:, 0:1] - target).pow(2).mean()
        expected_b = (pred[:, 1:2] - target).pow(2).mean()
        assert torch.isclose(raw[f'{ic.name}_a'], expected_a)
        assert torch.isclose(raw[f'{ic.name}_b'], expected_b)


def test_hard_constrained_conditions_are_exactly_zero():
    pinn = _build_pinn(hard_conditions=('ic2', 'bc_right'))
    # weights for hard-constrained terms are never looked up -- omit them
    # to also confirm loss_fn doesn't KeyError trying to read them.
    weights = {
        'pde_a': 1.0, 'pde_b': 1.0,
        'ic1_a': 1.0, 'ic1_b': 1.0,
        'bc_left': 1.0,
    }
    total_loss, terms = pinn.loss_fn(w=weights)

    for key in ('ic2_a', 'ic2_b', 'bc_right'):
        assert terms[key].item() == 0.0
        assert pinn.meta['raw_loss_terms'][key].item() == 0.0

    # non-hard terms should NOT be trivially zero (sanity check that the
    # hard-constrained branch didn't accidentally zero everything).
    assert terms['ic1_a'].item() != 0.0 or terms['ic1_b'].item() != 0.0


def test_missing_weight_key_raises_keyerror():
    pinn = _build_pinn()
    incomplete_weights = dict(_FULL_WEIGHTS)
    del incomplete_weights['bc_right']
    with pytest.raises(KeyError):
        pinn.loss_fn(w=incomplete_weights)


def test_n_col_n_initial_n_boundary_properties():
    pinn = _build_pinn()
    assert pinn.n_col == 64
    assert pinn.n_initial == 16 + 8       # ic1 + ic2
    assert pinn.n_boundary == 16 + 8      # bc_left + bc_right


def test_causal_mode_aggregates_pde_species_into_single_key():
    pinn = _build_pinn(causal=True, n_chunks=4)
    total_loss, terms = pinn.loss_fn(w=_FULL_WEIGHTS)

    assert 'pde' in terms
    assert 'pde_a' not in terms and 'pde_b' not in terms

    expected_keys = {'pde', 'ic1_a', 'ic1_b', 'ic2_a', 'ic2_b', 'bc_left', 'bc_right'}
    assert set(terms.keys()) == expected_keys

    assert pinn.meta['chunk_losses'].shape == (4,)
    assert pinn.meta['chunk_weights'].shape == (4,)