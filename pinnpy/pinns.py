from __future__ import annotations
from typing import Callable

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt

import torch

from . import residuals
from .datasets import DatasetSampler
from .neural_nets import BaseMLP, FCNN
from .constrained_net import ConstrainedNet
from . import constants as _CST
from .causal import causal_weighted_residual


class BasePinn(ABC):

    '''Model + physics + persistence. Owns the network and its training points,
    and knows how to *score itself* — but knows nothing about optimizers.'''

    BETA = _CST.P_STAR / _CST.D_STAR

    def __init__(self, n_col: int, n_initial: int, n_center: int, n_surface: int, initial_fn: Callable, net: BaseMLP | ConstrainedNet | None = None, layers: int = 4, neurons: int = 16, l_bounds: tuple[float, float] = (0, 0), u_bounds: tuple[float, float] = (1.0, 1.0), device: str = 'cpu', seed: int = 42, dtype: torch.dtype = torch.float32, hard_conditions: tuple[str, ...] = ('ic', 'neumann', 'robin'), causal: bool = False, n_chunks: int = 24, causal_eps: float = 1.0):
        
        self.seed = seed
        self.device = device
        self.dtype = dtype
        self.meta: dict[str, Any] = {}
        # ---
        self.causal = causal
        self.n_chunks = n_chunks
        self.causal_eps = causal_eps
        # ---
        self.lower_bounds = list(l_bounds)
        self.upper_bounds = list(u_bounds)
        # ---
        self.n_col = n_col
        self.n_initial = n_initial
        self.n_center = n_center
        self.n_surface = n_surface
        # ---
        self.initial_fn = initial_fn
        self.hard_conditions = hard_conditions
        # ---
        self.net = net if net is not None else FCNN(in_dim=2, out_dim=3, n_layers=layers, n_neurons=neurons, initialization='xavier_normal', seed=self.seed)
        self.net.to(self.device)

        # Dataset atttributes
        self.sampler = DatasetSampler(seed=self.seed)
        self.collocation_training_dataset = self.sampler.sample_collocation_points(n_points=self.n_col, l_bounds=l_bounds, u_bounds=u_bounds)
        self.initial_training_dataset = self.sampler.sample_initial_points(n_points=self.n_initial, l_bounds=l_bounds, u_bounds=u_bounds)
        self.center_training_dataset = self.sampler.sample_center_points(n_points=self.n_center, l_bounds=l_bounds, u_bounds=u_bounds)
        self.surface_training_dataset = self.sampler.sample_surface_points(n_points=self.n_surface, l_bounds=l_bounds, u_bounds=u_bounds)
        self.sensor_training_dataset = None
        
        self._DATASET = self.collocation_training_dataset + self.center_training_dataset + self.surface_training_dataset + self.initial_training_dataset

    def forward(self, r: torch.Tensor, t: torch.Tensor):
        '''forward pass throught the net: (r, t) -> (c0, c1, c2)'''
        return self.net(r, t)
    
    def mse_loss(self, w: dict[str, float], L: float = 1.0, scaled: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        '''Compute the mean squared error loss for the PDE, IC, and BC residuals. Returns (total_loss, individual_loss_terms).'''
        pde0, pde1, pde2 = residuals.pde(net=self.net, dataset=self.collocation_training_dataset, L=L, scaled=scaled)
        ic0, ic1, ic2 = residuals.initial(net=self.net, dataset=self.initial_training_dataset, ic_func=self.initial_fn) if 'ic' not in self.hard_conditions else (torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))
        center = residuals.center_neumann(net=self.net, dataset=self.center_training_dataset, target=0.0) if 'neumann' not in self.hard_conditions else torch.tensor(0.0)
        surface = residuals.surface_robin(net=self.net, dataset=self.surface_training_dataset) if 'robin' not in self.hard_conditions else torch.tensor(0.0)

        raw_loss_terms = {
            'pde0': pde0.pow(2).mean(), 'pde1': pde1.pow(2).mean(), 'pde2': pde2.pow(2).mean(),
            'ic0': ic0.pow(2).mean(), 'ic1': ic1.pow(2).mean(), 'ic2': ic2.pow(2).mean(),
            'center': center.pow(2).mean(), 'surface': surface.pow(2).mean(),
        }
        self.meta['raw_loss_terms'] = raw_loss_terms

        if self.causal:
            pde_loss, chunk_losses, chunk_weights = causal_weighted_residual(
                residual_terms={'pde0': pde0, 'pde1': pde1, 'pde2': pde2},
                weights={'pde0': w['pde0'], 'pde1': w['pde1'], 'pde2': w['pde2']},
                t=self.collocation_training_dataset.t,
                t_bounds=(self.lower_bounds[1], self.upper_bounds[1]),
                n_chunks=self.n_chunks,
                eps=self.causal_eps
            )
            self.meta['chunk_losses'] = chunk_losses
            self.meta['chunk_weights'] = chunk_weights
            raw_loss_terms['pde'] = raw_loss_terms['pde0'] + raw_loss_terms['pde1'] + raw_loss_terms['pde2']

            individual_weighted_loss_terms = {
                'pde': pde_loss,
                'ic0': w['ic0'] * ic0.pow(2).mean(),
                'ic1': w['ic1'] * ic1.pow(2).mean(),
                'ic2': w['ic2'] * ic2.pow(2).mean(),
                'center': w['center'] * center.pow(2).mean(),
                'surface': w['surface'] * surface.pow(2).mean()
            }

        else:
            individual_weighted_loss_terms = {
                'pde0': w['pde0'] * pde0.pow(2).mean(),
                'pde1': w['pde1'] * pde1.pow(2).mean(),
                'pde2': w['pde2'] * pde2.pow(2).mean(),
                'ic0': w['ic0'] * ic0.pow(2).mean(),
                'ic1': w['ic1'] * ic1.pow(2).mean(),
                'ic2': w['ic2'] * ic2.pow(2).mean(),
                'center': w['center'] * center.pow(2).mean(),
                'surface': w['surface'] * surface.pow(2).mean()
            }

        total_loss = torch.stack(list(individual_weighted_loss_terms.values())).sum()

        individual_weighted_loss_terms.update(self._extra_loss_term(L))      # Adds data loss term at inverse problems.
        return total_loss, individual_weighted_loss_terms
    
    @abstractmethod
    def _extra_loss_term(self, L: float) -> dict[str, torch.Tensor]:
        ...


    def resample_datasets(self, n_col: int, n_initial: int, n_center: int, n_surface: int, lower_bounds: tuple[float, float] = (0.0, 0.0), upper_bounds: tuple[float, float] = (1.0, 1.0) ) -> 'BasePinn':
        """Redraw the training point sets in place (RAR-G hook). Returns self."""
        self.collocation_training_dataset += self.sampler.sample_collocation_points(n_points=n_col, l_bounds=lower_bounds, u_bounds=upper_bounds)
        self.initial_training_dataset += self.sampler.sample_initial_points(n_points=n_initial, l_bounds=lower_bounds, u_bounds=upper_bounds)
        self.center_training_dataset += self.sampler.sample_center_points(n_points=n_center, l_bounds=lower_bounds, u_bounds=upper_bounds)
        self.surface_training_dataset += self.sampler.sample_surface_points(n_points=n_surface, l_bounds=lower_bounds, u_bounds=upper_bounds)
        self._DATASET = self.collocation_training_dataset + self.center_training_dataset + self.surface_training_dataset + self.initial_training_dataset
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        print(f'points resampled...\n-new total: {self._DATASET.n_points}\n-upper_bounds: {self.upper_bounds}')
        return self
    
    def to_checkpoint(self) -> dict:
        """The model's serialization format lives here, next to `load` that reads it
        back — single source of truth for the schema."""
        checkpoint =  {
            'arch': {'n_layers': self.net.n_layers,
                    'n_neurons': self.net.n_neurons,
                    'seed': self.net.seed},
            'state_dict': self.net.state_dict(),
        }
        return checkpoint
    
    @classmethod
    def load(cls, path: str | Path, device: str = 'cpu', dtype=torch.float32):

        """Rebuild a (concrete) Pinn for evaluation. Factory via `cls.__new__`:
        skips `__init__` so we don't re-sample datasets we don't need to score
        a trained net. Call on a concrete subclass (e.g. `ForwardPinn.load`),
        not on BasePinn (abstract — `__new__` would refuse)."""

        checkpoint: dict[str, Any] = torch.load(path, map_location=device)
        arch = checkpoint['arch']
    
        net = FCNN(2, 3, n_layers=arch['n_layers'], n_neurons=arch['n_neurons'], initialization='xavier_normal', seed=arch['seed']).to(device=device)
        net.load_state_dict(checkpoint['state_dict'])
        net.eval()

        new_pinn = cls.__new__(cls)        # bypasses the __init__() constructor.
        new_pinn.net = ConstrainedNet(inner_net=net, beta=cls.BETA, c_sol_star=_CST.C_SOL_STAR, eps=0.01, enforce=('ic', 'neumann', 'robin'))
        new_pinn.device = device
        new_pinn.dtype = dtype
        new_pinn.seed = arch['seed']
        new_pinn.meta = checkpoint.get('meta', {}) 
        return new_pinn
    
    def check_concentration_profile(self, epoch=None, L=1.0):
        t_exp = self.upper_bounds[1]
        fig = plt.figure(figsize=(6, 4))
        r = torch.linspace(0, 1, 100).reshape(-1, 1)
        for time in [0, 0.25 * t_exp, 0.5 * t_exp, 0.75 * t_exp, t_exp]:
            t = torch.full_like(r, time)
            with torch.no_grad():
                c0 = self.forward(r, t)[:, 0:1]
            plt.plot(r, c0, label=f't={time * 24}h')
        plt.title(f'spatial distribution of TRM || time: [0 - {t_exp * 24}h] - R_T: {L * _CST.R_T} || epoch: {epoch}')
        plt.xlabel(f'r')
        plt.ylabel(f'[Ab_I]/[C_reference]')
        plt.legend()
        plt.show()

    def __repr__(self) -> str:
        return f'{type(self).__name__}()'
            


class ForwardPinn(BasePinn):

    """Forward solver: all PDE parameters are known constants from
    `constants.py`. No data term."""

    def _extra_loss_term(self, L: float) -> dict[str, torch.Tensor]:
        return {}
    

class InversePinn(BasePinn):
     
     '''inverse solver.'''

     def _extra_loss_term(self, L: float) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            'InversePinn is a documented seam; build it when an inverse '
            'problem with observational data is specified.')


    