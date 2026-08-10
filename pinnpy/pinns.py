"""Generic PINN orchestration: PINN and the physics-injection contract.
 
PINN owns training-point sampling, loss aggregation, and checkpoint
persistence for a physics-informed neural network -- but knows nothing
about any specific PDE. Every piece of actual physics (the interior PDE
residual, initial conditions, boundary conditions, and any hard-constraint
parameters) is supplied by the caller as plain data: PDE / InitialCondition
/ BoundaryCondition objects, each wrapping a callable that does the real
computation.
 
Minimal example (a 1-species, 1-BC toy problem):
 
    import torch
    from pinnpy.pinns import ForwardPinn, PDE, BoundaryCondition
    from pinnpy.datasets import DatasetSampler
 
    def my_pde_residual(net, dataset, **kwargs) -> dict[str, torch.Tensor]:
        r = dataset.data.clone()[:, 0:1].requires_grad_(True)
        t = dataset.data.clone()[:, 1:2].requires_grad_(True)
        u = net(r, t)[:, 0:1]
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        return {'u': u_t - u}   # e.g. du/dt = u
 
    def my_boundary_residual(net, dataset, **kwargs) -> torch.Tensor:
        r = dataset.data.clone()[:, 0:1].requires_grad_(True)
        t = dataset.data.clone()[:, 1:2]
        u = net(r, t)[:, 0:1]
        return u   # e.g. u(r=1, t) = 0
 
    sampler = DatasetSampler(seed=42)
    boundary_ds = sampler.sample_surface_points(n_points=200)
 
    pinn = ForwardPinn(
        pde=PDE(residual_fn=my_pde_residual),
        boundary_conditions=[
            BoundaryCondition(name='boundary', dataset=boundary_ds, residual_fn=my_boundary_residual),
        ],
        n_col=5_000,
        n_species=1,   # must match how many keys my_pde_residual returns
    )
 
See PDE, InitialCondition, and BoundaryCondition below for the full
signature each residual_fn callable must satisfy.
"""

from __future__ import annotations
from typing import Callable, Any, Protocol
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from .datasets import DatasetSampler, Dataset
from .neural_nets import MLP, FCNN
from .constrained_net import ConstrainedNet
from .causal import causal_weighted_residual

class PDEResidualFn(Protocol):
    """Call signature every `PDE.residual_fn` callable must satisfy.
 
    Called by `PINN.loss_fn` as `residual_fn(net, dataset, **kwargs)`,
    where `**kwargs` is the union of whatever
    `loss_fn(w=..., **pde_kwargs)` was called with, plus this PDE's own
    fixed `.kwargs`. Must return one raw (unsquared) residual tensor
    per solution species, keyed by whatever species names you choose.
    """
    def __call__(self, net: MLP, dataset: Dataset, **kwargs) -> dict[str, torch.Tensor]: ...

class ICResidualFn(Protocol):
    """Call signature every `InitialCondition.residual_fn` callable must
    satisfy.
 
    Called as `residual_fn(net, dataset, ic_func=ic.ic_func, **ic.kwargs)`.
    Must return the same species keys as the `PDE.residual_fn` it's
    paired with -- an IC constrains the whole solution state at a fixed
    time, not just one channel.
    """
    def __call__(self, net: MLP, dataset: Dataset, ic_func: Callable[[torch.Tensor], torch.Tensor], **kwargs) -> dict[str, torch.Tensor]: ...

class BCResidualFn(Protocol):
    """Call signature every `BoundaryCondition.residual_fn` callable
    must satisfy.
 
    Called as `residual_fn(net, dataset, **bc.kwargs)`. Returns a
    single residual tensor -- a BC typically constrains one channel
    (see `ICResidualFn`'s docstring for why ICs return a dict instead).
    """
    def __call__(self, net: MLP, dataset: Dataset, **kwargs) -> torch.Tensor: ...

@dataclass
class PDE:

    """Wraps your PDE's interior residual function AND the collocation
    dataset it's evaluated on.
 
    A PINN has exactly one PDE, evaluated once per training step (inside
    PINN.loss_fn). Unlike InitialCondition/BoundaryCondition, PDE is NOT
    frozen -- RAR-G-style adaptive refinement (see sampling.py) grows
    `pde.dataset` in place over training
    (`pde.dataset = pde.dataset + more_points`), so this needs to stay
    mutable.
 
    Args:
        dataset: interior collocation points -- build via
            `DatasetSampler.sample_collocation_points(...)`, or supply
            your own. `PINN.n_col` reads this dataset's length directly
            (`len(pde.dataset)`) rather than tracking a separate count,
            so it can never drift out of sync after growth.
        residual_fn: called as `residual_fn(net, dataset, **kwargs)`,
            where `**kwargs` is the union of whatever
            `loss_fn(w=..., **pde_kwargs)` was called with, plus this
            PDE's own `.kwargs` below. Must return a
            `dict[str, torch.Tensor]` -- one raw (unsquared) residual
            tensor per solution species, keyed by a name you choose
            (e.g. `{'c0': res0, 'c1': res1}`).
 
            The number of keys you return here must equal `net`'s
            output width. If you're letting PINN build a default
            network for you, pass that same count as `n_species` when
            constructing the PINN (see `PINN.__init__`) -- otherwise
            the default network's output shape won't match your
            residual's shape and you'll get a runtime tensor-shape
            error the first time `loss_fn` runs.
 
            This key set also drives every downstream loss-term name
            (`'pde_c0'`, `'pde_c1'`, ...) and is what
            `InitialCondition.residual_fn` must match if you use one.
        kwargs: extra keyword arguments always forwarded to
            `residual_fn`, on every call, in addition to whatever
            per-call `pde_kwargs` `loss_fn` forwards. Use this for
            problem constants that don't change call-to-call (e.g. a
            diffusion coefficient), as opposed to things like a
            receptor-load scaling factor you might want to vary from
            one `loss_fn` call to the next via `pde_kwargs`.
 
            CAUTION: if a key appears in BOTH this dict and the
            `pde_kwargs` a caller passes to `loss_fn`, calling
            `residual_fn` raises `TypeError: got multiple values for
            keyword argument` -- keep the two disjoint.
 
    Example -- a 1-species heat equation, du/dt = diffusivity * d2u/dr2,
    with `diffusivity` fixed at construction time (via `kwargs`) rather
    than passed per-call:
 
        def heat_residual(net, dataset, diffusivity):
            r = dataset.data.clone()[:, 0:1].requires_grad_(True)
            t = dataset.data.clone()[:, 1:2].requires_grad_(True)
            u = net(r, t)[:, 0:1]
            u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
            u_r = torch.autograd.grad(u, r, torch.ones_like(u), create_graph=True)[0]
            u_rr = torch.autograd.grad(u_r, r, torch.ones_like(u_r), create_graph=True)[0]
            return {'u': u_t - diffusivity * u_rr}
 
        pde = PDE(
            residual_fn=heat_residual,
            dataset=sampler.sample_collocation_points(n_points=10_000),
            kwargs={'diffusivity': 0.01},
        )
        # net's output width must be 1 here -- pass n_species=1 to PINN
        # if you want PINN to build the default network for you.
    """

    dataset: Dataset
    residual_fn: PDEResidualFn
    kwargs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class InitialCondition:

    """One initial condition -- a constraint on the solution at a fixed
    time (typically t=0).
 
    PINN accepts a *list* of these (`initial_conditions=[...]`), not a
    single one: a steady-state problem that doesn't need an IC at all
    passes an empty list; a problem needing more than one IC-like
    constraint (e.g. matching two different snapshots) passes more than
    one, each with a distinct `name`.
 
    Whether this IC is actually evaluated as a soft loss term depends on
    `PINN.hard_conditions`: if `ic.name` appears in `hard_conditions`,
    this IC is assumed to be enforced architecturally instead (e.g. via
    ConstrainedNet), and its loss contribution is fixed at zero rather
    than computed.
 
    Frozen (unlike PDE): an IC's `.name` is used as a loss-term-key
    namespace inside `loss_fn`, so it should never be silently
    reassigned mid-training. If you need a different IC, construct a
    new InitialCondition and reassign `pinn.initial_conditions`.
 
    Args:
        name: unique identifier for this IC. Used to (a) check against
            `hard_conditions`, and (b) namespace this IC's loss-term
            keys as `'{name}_{species}'` (e.g. `'ic_c0'`, `'ic_c1'`).
            Must be unique across all of a PINN's InitialConditions.
        dataset: the points this IC is evaluated on -- typically
            `sampler.sample_initial_points(...)`, but any Dataset works.
        ic_func: the target function this IC compares the network
            against, e.g. `lambda r: torch.zeros_like(r)` for a
            homogeneous IC. Passed straight through to `residual_fn` as
            its `ic_func` argument -- what you do with it is entirely up
            to your `residual_fn` implementation.
        residual_fn: called as
            `residual_fn(net, dataset, ic_func=ic_func, **kwargs)`.
            Must return a `dict[str, torch.Tensor]` keyed identically to
            your `PDE.residual_fn`'s species keys.
        kwargs: extra fixed keyword arguments forwarded to `residual_fn`
            on every call, in addition to `net`/`dataset`/`ic_func`.
 
    Example:
        def initial_residual(net, dataset, ic_func):
            r = dataset.r.requires_grad_(True)
            t = dataset.t
            u = net(r, t)[:, 0:1]
            return {'u': u - ic_func(r)}
 
        ic = InitialCondition(
            name='ic',
            dataset=sampler.sample_initial_points(n_points=200),
            ic_func=lambda r: torch.zeros_like(r),
            residual_fn=initial_residual,
        )
    """

    name: str
    dataset: Dataset
    ic_func: Callable[[torch.Tensor], torch.Tensor]
    residual_fn: ICResidualFn
    kwargs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BoundaryCondition:
    """One boundary condition -- a constraint on the solution at a fixed
    spatial location (or region), evaluated across the time domain.
 
    PINN accepts a *list* of these, exactly like InitialCondition -- a
    problem with more or fewer BCs than trastuzumab's (center, surface)
    just supplies a different-length list; nothing in PINN hardcodes BC
    names or count.
 
    Same `hard_conditions` behavior as InitialCondition: if `bc.name`
    appears in `hard_conditions`, this BC's loss contribution is fixed
    at zero (assumed enforced architecturally instead).
 
    Frozen, for the same reason as InitialCondition: `.name` is a
    loss-key namespace, not something to reassign mid-training.
 
    Args:
        name: unique identifier for this BC. Used to (a) check against
            `hard_conditions`, and (b) as this BC's loss-term key
            directly (e.g. `'center'`, `'surface'`). Must be unique
            across all of a PINN's BoundaryConditions.
        dataset: the points this BC is evaluated on -- typically
            `sampler.sample_center_points(...)` /
            `sample_surface_points(...)`, but any Dataset works.
        residual_fn: called as `residual_fn(net, dataset, **kwargs)`.
            Must return a single `torch.Tensor`.
        kwargs: extra fixed keyword arguments forwarded to `residual_fn`
            on every call, in addition to `net`/`dataset`. Example:
            `BoundaryCondition(..., kwargs={'target': 0.0})` for a
            Neumann BC with a configurable target flux.
 
    Example:
        def neumann_residual(net, dataset, target=0.0):
            r = dataset.r.requires_grad_(True)
            t = dataset.t
            u = net(r, t)[:, 0:1]
            u_r = torch.autograd.grad(u, r, torch.ones_like(u), create_graph=True)[0]
            return u_r - target
 
        bc = BoundaryCondition(
            name='center',
            dataset=sampler.sample_center_points(n_points=200),
            residual_fn=neumann_residual,
            kwargs={'target': 0.0},
        )
    """
     
    name: str
    dataset: Dataset
    residual_fn: BCResidualFn
    kwargs: dict = field(default_factory=dict)


class PINN(ABC):

    """Model + physics + persistence for a physics-informed neural network.
 
    Owns the network and loss aggregation across the PDE residual and
    any injected initial/boundary conditions -- and knows how to score
    itself. Knows nothing about optimizers (see Trainer for that) and
    nothing about any specific PDE (see PDE / InitialCondition /
    BoundaryCondition above for how physics gets injected).
 
    This class is abstract -- instantiate a concrete subclass instead:
    ForwardPinn (no data term) or InversePinn (data-fitting term, not
    yet implemented).
 
    Args:
        pde: the interior PDE residual and its dataset. See `PDE`.
        n_species: how many output channels the network needs. Only
            used if `net` is not supplied (PINN builds a default `FCNN`
            sized to this). Must equal the number of keys your
            `pde.residual_fn` returns -- get this wrong and training
            will fail the first time `loss_fn` runs, with a tensor-shape
            mismatch between the network's output and the residual
            dict. Ignored if you pass your own `net` (its output width
            is already fixed by however you built it).
        initial_conditions: list of `InitialCondition`s. Default `[]`
            -- a problem with no IC (e.g. steady-state) can omit this.
        boundary_conditions: list of `BoundaryCondition`s. Default `[]`.
        net: a pre-built `MLP`-family network to train. If omitted, a
            default `FCNN(in_dim=2, out_dim=n_species, n_layers=layers, n_neurons=neurons, ...)`
            is constructed -- override `layers`/`neurons` to size it, or
            pass your own `net` (e.g. a `ModifiedMLP`, or one with a
            Fourier `input_transformation`) for anything more specific.
        layers, neurons: architecture size for the default `net`, if
            `net` is not supplied. Ignored if `net` is supplied.
        l_bounds, u_bounds: (r, t) domain bounds, used only for binning
            causal-weighting time chunks when `causal=True`. Not used
            to build any dataset -- every Dataset (`pde.dataset`, IC/BC
            datasets) is built by you, before constructing the PINN.
        device, seed, dtype: standard training/reproducibility knobs.
        hard_conditions: names of `InitialCondition`s/`BoundaryCondition`s
            (matched against their `.name`) that are enforced
            architecturally (e.g. via a `ConstrainedNet`-wrapped `net`)
            rather than as soft loss terms. Their loss contribution is
            fixed at zero in `loss_fn` rather than computed.
        causal: if `True`, weight the PDE loss by time-chunk per
            temporal causal weighting (see `causal.py`) instead of
            averaging uniformly over the whole domain.
        n_chunks, causal_eps: causal-weighting hyperparameters, used
            only if `causal=True`.
        constrained_net_kwargs: keyword arguments forwarded to
            `ConstrainedNet(inner_net=net, **constrained_net_kwargs)`
            when a checkpoint is reloaded via `load()`. Only relevant if
            you plan to `to_checkpoint()`/`load()` a hard-constrained
            model -- see `load()`'s docstring.
 
    Example:
        pinn = ForwardPinn(
            pde=my_pde,                                 # already carries its own dataset
            n_species=3,                                # however many keys my_pde.residual_fn returns
            initial_conditions=[my_ic],
            boundary_conditions=[my_bc_1, my_bc_2],
            hard_conditions=('ic', 'my_bc_1'),          # my_bc_2 stays a soft loss term
        )
        total_loss, loss_terms = pinn.loss_fn(w=my_weights_dict, L=1.0, scaled=True)
        # L/scaled above are just example pde_kwargs -- entirely up to
        # what my_pde.residual_fn actually accepts.
    """

    def __init__(self, pde: PDE, 
                 n_species: int | None = None, net: MLP | None = None, layers: int = 4, neurons: int = 16,
                 initial_conditions: list[InitialCondition] | None = None, boundary_conditions: list[BoundaryCondition] | None = None,
                 l_bounds: tuple[float, float] = (0, 0), u_bounds: tuple[float, float] = (1.0, 1.0), 
                 device: str = 'cpu', seed: int = 42, dtype: torch.dtype = torch.float32, 
                 hard_conditions: tuple[str, ...] = (), constrained_net_kwargs: dict[str, Any] | None = None,
                 causal: bool = False, n_chunks: int = 24, causal_eps: float = 1.0):
        
        self.seed = seed
        self.device = device
        self.dtype = dtype
        self.meta: dict[str, Any] = {}

        self.pde = pde
        self.initial_conditions = initial_conditions or []
        self.boundary_conditions = boundary_conditions or []
        self.constrained_net_kwargs = constrained_net_kwargs or {}

        # cauasal-weighting hyperparameters
        self.causal = causal
        self.n_chunks = n_chunks
        self.causal_eps = causal_eps

        self.lower_bounds = list(l_bounds)
        self.upper_bounds = list(u_bounds)

        self.hard_conditions = hard_conditions

        if net is not None:
            self.net = net
        else:
            if n_species is None:
                raise ValueError(
                    "PINN needs to know how many output channels to build for "
                    "the default network. Either pass n_species (must match the "
                    "number of keys your pde.residual_fn returns), or construct "
                    "and pass your own net explicitly."
                )
            self.net = FCNN(in_dim=2, out_dim=n_species, n_layers=layers, n_neurons=neurons, initialization='xavier_normal', seed=self.seed)
        self.net.to(self.device)

        # Dataset atttributes
        self.sampler = DatasetSampler(seed=self.seed)
        self.sensor_training_dataset = None

        self._DATASET = self.pde.dataset
        for ic in self.initial_conditions:
            self._DATASET += ic.dataset
        for bc in self.boundary_conditions:
            self._DATASET += bc.dataset
        

    @property
    def n_col(self) -> int: return len(self.pde.dataset)

    @property
    def n_initial(self) -> int: return sum(len(ic.dataset) for ic in self.initial_conditions)

    @property
    def n_boundary(self) -> int: return sum(len(bc.dataset) for bc in self.boundary_conditions)

    def forward(self, r: torch.Tensor, t: torch.Tensor):
        '''forward pass throught the net: (r, t) -> (c0, c1, c2)'''
        return self.net(r, t)

    
    def loss_fn(self, w: dict[str, float], **pde_kwargs) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the total weighted MSE loss across the PDE residual and
        every injected initial/boundary condition.
 
        Args:
            w: per-term weights, keyed exactly like the returned
                `individual_loss_terms` dict -- one float weight per
                `'pde_{species}'` key (or `'pde'` if `causal=True`), per
                `'{ic.name}_{species}'` key, and per `bc.name`. Must
                supply a weight for every non-hard-constrained term or
                `loss_fn` raises a `KeyError`.
            **pde_kwargs: forwarded straight through to
                `self.pde.residual_fn(net=..., dataset=self.pde.dataset, **pde_kwargs, **self.pde.kwargs)`.
                Entirely up to what your `pde.residual_fn` accepts --
                e.g. trastuzumab's PDE expects `L=...`/`scaled=...`; a
                different problem might expect nothing here at all, or
                something else entirely. See `PDE`'s docstring for the
                collision caveat if a key appears in both `pde_kwargs`
                and `self.pde.kwargs`.
 
        Returns:
            (total_loss, individual_loss_terms):
                total_loss: scalar Tensor, ready for `.backward()`.
                individual_loss_terms: dict[str, Tensor] of every
                    weighted loss term that was summed into total_loss,
                    plus whatever `self._extra_loss_term(...)` contributes
                    (see `ForwardPinn`/`InversePinn`).
 
        Loss-term keys:
            - `'pde_{species}'` for each key in
              `self.pde.residual_fn(...)`'s return dict (or a single
              `'pde'` key if `causal=True` -- see `causal.py`).
            - `'{ic.name}_{species}'` for each `InitialCondition`, one
              per species key its `residual_fn` returns.
            - `bc.name` for each `BoundaryCondition`.
        """
        # 1. compute pde residuual
        pde_res = self.pde.residual_fn(net=self.net, dataset=self.pde.dataset, **pde_kwargs, **self.pde.kwargs)

        # 2. compute raw loss terms (unweighted MSE) for PDE, ICs, BCs
        raw_loss_terms = {f'pde_{key}': residual.pow(2).mean() for key, residual in pde_res.items()}

        ic_res_by_name: dict[str, dict[str, torch.Tensor]] = {}
        for ic in self.initial_conditions:
            if ic.name not in self.hard_conditions:
                ic_res = ic.residual_fn(net=self.net, dataset=ic.dataset, ic_func=ic.ic_func, **ic.kwargs)
            else:
                ic_res = {k: torch.tensor(0.0) for k in pde_res.keys()}
            ic_res_by_name[ic.name] = ic_res
            raw_loss_terms.update({f'{ic.name}_{k}': res.pow(2).mean() for k, res in ic_res.items()})

        for bc in self.boundary_conditions:
            if bc.name not in self.hard_conditions:
                bc_res = bc.residual_fn(net=self.net, dataset=bc.dataset, **bc.kwargs)
                raw_loss_terms[bc.name] = bc_res.pow(2).mean()
            else:
                raw_loss_terms[bc.name] = torch.tensor(0.0)

        self.meta['raw_loss_terms'] = raw_loss_terms

        # 3. compute weighted loss terms
        if self.causal:
            pde_loss, chunk_losses, chunk_weights = causal_weighted_residual(
                residual_terms={f'pde_{key}': res for key, res in pde_res.items()},
                weights={f'pde_{key}': w[f'pde_{key}'] for key in pde_res.keys()},
                t=self.pde.dataset.t,
                t_bounds=(self.lower_bounds[1], self.upper_bounds[1]),
                n_chunks=self.n_chunks,
                eps=self.causal_eps
            )
            self.meta['chunk_losses'] = chunk_losses
            self.meta['chunk_weights'] = chunk_weights
            raw_loss_terms['pde'] = torch.stack([raw_loss_terms[f'pde_{key}'] for key in pde_res.keys()]).sum()
            individual_weighted_loss_terms = {'pde': pde_loss}
        else:
            individual_weighted_loss_terms = {f'pde_{key}': w[f'pde_{key}'] * residual.pow(2).mean() for key, residual in pde_res.items()}


        for ic in self.initial_conditions:
            for key, residual in ic_res_by_name[ic.name].items():
                if ic.name not in self.hard_conditions:
                    individual_weighted_loss_terms[f'{ic.name}_{key}'] = w[f'{ic.name}_{key}'] * residual.pow(2).mean()
                else:
                    individual_weighted_loss_terms[f'{ic.name}_{key}'] = torch.tensor(0.0)

        for bc in self.boundary_conditions:
            if bc.name not in self.hard_conditions:
                individual_weighted_loss_terms[bc.name] = w[bc.name] * raw_loss_terms[bc.name]
            else:
                individual_weighted_loss_terms[bc.name] = torch.tensor(0.0)

        total_loss = torch.stack(list(individual_weighted_loss_terms.values())).sum()
        individual_weighted_loss_terms.update(self._extra_loss_term(**pde_kwargs))      # Adds data loss term at inverse problems.

        return total_loss, individual_weighted_loss_terms
    
    @abstractmethod
    def _extra_loss_term(self, **pde_kwargs) -> dict[str, torch.Tensor]:
        """Extra loss term(s) a concrete subclass contributes on top of
        the PDE/IC/BC terms loss_fn already computes -- e.g. a
        data-fitting term for InversePinn. Return `{}` if there's
        nothing extra (see ForwardPinn). Receives the same `pde_kwargs`
        loss_fn was called with, in case the extra term needs them."""
        ...


    def resample_datasets(self, n_collocation: int, n_initial: int, n_left: int, n_right: int, lower_bounds: tuple[float, float] = (0.0, 0.0), upper_bounds: tuple[float, float] = (1.0, 1.0) ) -> 'PINN':
        """Redraw the interior collocation dataset in place (the RAR-G
        hook -- see `sampling.py`) and update the domain bounds. Returns
        self, so this chains: `pinn.resample_datasets(...).loss_fn(...)`.
 
        Note: `InitialCondition`/`BoundaryCondition` datasets are NOT
        resampled here -- if your IC/BC points need to change too,
        resample them yourself and reassign
        `pinn.initial_conditions`/`pinn.boundary_conditions` before
        calling this. PINN doesn't know how a given IC/BC's dataset
        should be regenerated (that's problem-specific).
 
        Args:
            n_collocation: number of NEW collocation points to draw and append
                (existing points are kept -- this grows the set, it
                doesn't replace it).
            lower_bounds, upper_bounds: new (r, t) domain bounds.
        """
        self.pde.dataset = self.pde.dataset + self.sampler.sample_collocation_points(n_points=n_collocation, l_bounds=lower_bounds, u_bounds=upper_bounds)
        self._DATASET = self.pde.dataset
        for ic in self.initial_conditions:
            self._DATASET = self._DATASET + ic.dataset
        for bc in self.boundary_conditions:
            self._DATASET = self._DATASET + bc.dataset
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        print(f'points resampled...\n-new total: {self._DATASET.n_points}\n-upper_bounds: {self.upper_bounds}')
        return self
    
    def to_checkpoint(self) -> dict:
        """Serialize this PINN's network to a checkpoint dict, ready for
        `torch.save`. Pairs with `load()`, which reads this exact
        schema back -- if you change what's saved here, update `load()`
        to match.
 
        Returns:
            {'arch': {n_layers, n_neurons, seed}, 'constrained_net_kwargs': {...}, 'state_dict': ...}
        """
        checkpoint = {
            'arch': {'n_layers': self.net.n_layers,
                    'n_neurons': self.net.n_neurons,
                    'seed': self.net.seed},
            'constrained_net_kwargs': self.constrained_net_kwargs,
            'state_dict': self.net.state_dict(),
        }
        return checkpoint
    
    @classmethod
    def load(cls, path: str | Path, device: str = 'cpu', dtype=torch.float32):

        """Rebuild a (concrete) Pinn for evaluation. Factory via `cls.__new__`:
        skips `__init__` so we don't re-sample datasets we don't need to score
        a trained net. Call on a concrete subclass (e.g. `ForwardPinn.load`),
        not on PINN (abstract — `__new__` would refuse)."""

        checkpoint: dict[str, Any] = torch.load(path, map_location=device)
        arch: dict[str, Any] = checkpoint['arch']
        c_net_kwargs = checkpoint.get('constrained_net_kwargs', {})

        net = FCNN(in_dim=2, out_dim=3, n_layers=arch['n_layers'], n_neurons=arch['n_neurons'], initialization='xavier_normal', seed=arch['seed']).to(device=device)
        net.load_state_dict(checkpoint['state_dict'])
        net.eval()

        new_pinn = cls.__new__(cls)        # bypasses the __init__() constructor.
        new_pinn.net = ConstrainedNet(inner_net=net, **c_net_kwargs)
        new_pinn.device = device
        new_pinn.dtype = dtype
        new_pinn.seed = arch['seed']
        new_pinn.constrained_net_kwargs = c_net_kwargs
        new_pinn.meta = checkpoint.get('meta', {}) 
        return new_pinn
    
    def check_concentration_profile(self, epoch=None, title=''):
        """Quick-look plot: channel-0 prediction vs. r, at five evenly
        spaced timesteps from 0 to `self.upper_bounds[1]`. Useful as a
        `Trainer` callback (`profile_every=...`) to eyeball training
        progress without a full evaluation pass.
 
        Args:
            epoch: shown in the plot title, if provided.
            title: prefix for the plot title -- pass something
                describing your problem/run, since this method itself
                has no domain knowledge to generate one.
        """
        t_exp = self.upper_bounds[1]
        fig = plt.figure(figsize=(6, 4))
        r = torch.linspace(0, 1, 100).reshape(-1, 1)
        for time in [0, 0.25 * t_exp, 0.5 * t_exp, 0.75 * t_exp, t_exp]:
            t = torch.full_like(r, time)
            with torch.no_grad():
                c0 = self.forward(r, t)[:, 0:1]
            plt.plot(r, c0, label=f't={time * 24}h')
        plt.title(f'{title} || epoch: {epoch}')
        plt.xlabel(f'r')
        plt.ylabel(f'channel 0')
        plt.legend()
        plt.show()

    def __repr__(self) -> str:
        return f'{type(self).__name__}()'
            


class ForwardPinn(PINN):

    """Forward solver: all PDE parameters are known constants, supplied
    via your `PDE`/`InitialCondition`/`BoundaryCondition` objects. No
    data-fitting term -- `_extra_loss_term` always returns `{}`."""

    def _extra_loss_term(self, **pde_kwargs) -> dict[str, torch.Tensor]:
        return {}
    

class InversePinn(PINN):
     
    """Inverse solver: for problems with observational data to fit
    alongside the PDE/IC/BC residuals (e.g. inferring an unknown
    physical parameter from sparse measurements).

    Not yet implemented -- this is a documented seam. Build it (i.e.
    override `_extra_loss_term` to compute a data-fitting term against
    your observations) when an inverse problem is actually specified.
    """

    def _extra_loss_term(self, **pde_kwargs) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            'InversePinn is a documented seam; build it when an inverse '
            'problem with observational data is specified.')


    