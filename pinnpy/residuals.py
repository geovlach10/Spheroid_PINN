from __future__ import annotations
from typing import Callable

import torch 
from .neural_nets import BaseMLP
from .constrained_net import ConstrainedNet
from .datasets import Dataset
from . import constants as _CON

## porosity profiles.
def phi(r: torch.Tensor):
    """Porosity profile phi(r) = 0.44 r^3.2 + 0.56.
    Works on numpy arrays (FDM) and torch tensors (PINN)
    """
    return 0.44 * r ** 3.2 + 0.56

def dphi_dr(r: torch.Tensor):
    """Analytic derivative phi'(r) = 0.44 * 3.2 * r^2.2."""
    return 0.44 * 3.2 * r ** 2.2

## residual operators.
def pde(net: BaseMLP | ConstrainedNet, dataset: Dataset, L=1.0, scaled: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ''' Everything is normilized r=r_hat, t=t_hat, C=C_hat...'''
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
    
    return (res0 / _CON.SCALE0, res1 / _CON.SCALE1(L), res2 / _CON.SCALE2 ) if scaled else (res0, res1, res2)

def center_neumann(net: BaseMLP | ConstrainedNet, dataset: Dataset, target=0.0) -> torch.Tensor:
    r = dataset.data.clone()[:,0:1].requires_grad_(True)
    t = dataset.data.clone()[:,1:2].requires_grad_(True)
    
    c0 = net(r, t)[:, 0:1]
    c0_r = torch.autograd.grad(c0, r, torch.ones_like(c0), create_graph=True)[0]
    return c0_r - target
  

def surface_robin(net: BaseMLP | ConstrainedNet, dataset: Dataset) -> torch.Tensor:
    r = dataset.data.clone()[:,0:1].requires_grad_(True)
    t = dataset.data.clone()[:,1:2].requires_grad_(True)

    c0 = net(r, t)[:, 0:1]
    u0 = c0 / phi(r)
    u0_r = torch.autograd.grad(u0, r, torch.ones_like(u0), create_graph=True)[0]

    return  phi(r) * u0_r - (_CON.P_STAR / _CON.D_STAR) * (_CON.C_SOL_STAR - u0)


def initial(net: BaseMLP | ConstrainedNet, dataset: Dataset, ic_func: Callable[[torch.Tensor], torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r = dataset.data.clone()[:,0:1].requires_grad_(True)
    t = dataset.data.clone()[:,1:2].requires_grad_(True)
    
    pred = net(r, t)
    c0 = pred[:, 0:1];  c1 = pred[:, 1:2];  c2 = pred[:, 2:3]
    ic0 = ic1 = ic2 = ic_func(r)

    return c0 - ic0, c1 - ic1, c2 - ic2
