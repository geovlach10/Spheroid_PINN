"""Concrete PDE/IC/BC residual functions for the trastuzumab uptake/
clearance model -- the trastuzumab-specific implementation of pinnpy's
PDE / InitialCondition / BoundaryCondition contract (see pinnpy.pinns).

Every function here is a plain callable, matching PDEResidualFn /
ICResidualFn / BCResidualFn's Protocol signatures -- there's no class
to subclass, just wrap each one in the matching pinnpy dataclass.

Species keys used throughout: 'c0' (free antibody), 'c1' (bound),
'c2' (internalized).

Note on hard constraints: this module's initial_residual is a SOFT IC
term (used when 'ic' is NOT in hard_conditions). trastuzumab's usual
setup hard-constrains IC/BC via ConstrainedNet (ic/neumann/robin) --
see pinnpy.constrained_net -- rather than pinnpy.hard_constraints'
zero_at_t0, since ConstrainedNet's subtractive-form IC/BC enforcement
is more accurate than an architectural t-multiply for this problem's
Neumann/Robin boundary conditions specifically."""

from __future__ import annotations
from typing import Callable

import torch 
from pinnpy.neural_nets import MLP
from pinnpy.datasets import Dataset
from . import constants as _CON

## porosity profiles.
def phi(r: torch.Tensor):
    """Tumor porosity profile: phi(r) = 0.44 r^3.2 + 0.56."""
    return 0.44 * r ** 3.2 + 0.56

def dphi_dr(r: torch.Tensor):
    """Analytic derivative phi'(r) = 0.44 * 3.2 * r^2.2."""
    return 0.44 * 3.2 * r ** 2.2

## residual operators.
def pde_residual(net: MLP, dataset: Dataset, L=1.0, scaled: bool = False, **kwargs) -> dict[str, torch.Tensor]:
    """Interior PDE residual for trastuzumab's 3-species uptake/
    clearance system. Use as `PDE(residual_fn=pde_residual, dataset=...)`.

    Args:
        net: the network being trained. Its output must have (at
            least) 3 columns: c0 (free), c1 (bound), c2 (internalized).
        dataset: interior collocation points, columns (r, t).
        L: receptor-load scaling factor (applied to `L * R_T_STAR`).
        scaled: if True, divide each raw residual by its characteristic
            scale (constants.SCALE0/1/2).

    Returns:
        {'c0': res0, 'c1': res1, 'c2': res2}.
    """
    # Slice the dataset and conenct the vectors to the graph
    r = dataset.data.clone()[:,0:1].requires_grad_(True)
    t = dataset.data.clone()[:, 1:2].requires_grad_(True)

    pred = net(r, t)
    c0 = pred[:, 0:1];  c1 = pred[:, 1:2];  c2 = pred[:, 2:3]
    u0 = c0 / phi(r)

    # calculate time derivatives
    u0_t = torch.autograd.grad(u0, t, torch.ones_like(u0), create_graph=True)[0]
    c1_t = torch.autograd.grad(c1, t, torch.ones_like(c1), create_graph=True)[0]
    c2_t = torch.autograd.grad(c2, t, torch.ones_like(c2), create_graph=True)[0]

    # Calculate space derivatives
    u0_r = torch.autograd.grad(u0, r, grad_outputs=torch.ones_like(u0), create_graph=True)[0]
    K =  phi(r) * r**2 * u0_r
    K_r = torch.autograd.grad(K, r, torch.ones_like(K), create_graph=True)[0]

    diffusion = _CON.D_STAR * K_r
    reaction = _CON.K_ON_STAR * u0 * (L * _CON.R_T_STAR - c1) - _CON.K_OFF_STAR * c1
    internalization = _CON.K_INT_STAR * c1
    
    res0 = phi(r) * u0_t * (r**2) - diffusion + reaction * (r**2)
    res1 = c1_t - reaction + internalization
    res2 = c2_t - internalization
    
    if scaled:
        res0, res1, res2 = res0 / _CON.SCALE0, res1 / _CON.SCALE1(L), res2 / _CON.SCALE2

    return {'c0': res0, 'c1': res1, 'c2': res2}

def initial_residual(net: MLP, dataset: Dataset, ic_func: Callable[[torch.Tensor], torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
    """Homogeneous IC: all three species should equal ic_func(r) at t=0.
    Use as `InitialCondition(name='ic', ..., residual_fn=initial_residual)`.
    Only evaluated when 'ic' is NOT in hard_conditions."""
    r = dataset.data.clone()[:,0:1].requires_grad_(True)
    t = dataset.data.clone()[:,1:2].requires_grad_(True)
    
    pred = net(r, t)
    c0 = pred[:, 0:1];  c1 = pred[:, 1:2];  c2 = pred[:, 2:3]
    ic0 = ic1 = ic2 = ic_func(r)

    return {'c0': c0 - ic0, 'c1': c1 - ic1, 'c2': c2 - ic2}

def center_neumann_residual(net: MLP, dataset: Dataset, target: float = 0.0, **kwargs) -> torch.Tensor:
    """Neumann BC at r=0: dc0/dr = target. Use as
    `BoundaryCondition(name='center', ..., residual_fn=center_neumann_residual, kwargs={'target': 0.0})`."""
    r = dataset.data.clone()[:,0:1].requires_grad_(True)
    t = dataset.data.clone()[:,1:2].requires_grad_(True)
    
    c0 = net(r, t)[:, 0:1]
    c0_r = torch.autograd.grad(c0, r, torch.ones_like(c0), create_graph=True)[0]
    return c0_r - target
  

def surface_robin_residual(net: MLP, dataset: Dataset, **kwargs) -> torch.Tensor:
    """Robin BC at r=1: surface flux balances bath mass transfer. Use
    as `BoundaryCondition(name='surface', ..., residual_fn=surface_robin_residual)`."""
    r = dataset.data.clone()[:,0:1].requires_grad_(True)
    t = dataset.data.clone()[:,1:2].requires_grad_(True)

    c0 = net(r, t)[:, 0:1]
    u0 = c0 / phi(r)
    u0_r = torch.autograd.grad(u0, r, torch.ones_like(u0), create_graph=True)[0]

    return  phi(r) * u0_r - (_CON.P_STAR / _CON.D_STAR) * (_CON.C_SOL_STAR - u0)



