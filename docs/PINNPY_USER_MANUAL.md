# pinnpy User Manual

A task-oriented walkthrough of `pinnpy` -- the generic PINN library. For
full parameter reference, see each module's docstrings (`pinnpy/pinns.py`,
`pinnpy/neural_nets.py`, etc.); this manual is about *how* to do things,
with worked examples, not an exhaustive Args: listing.

`pinnpy` knows nothing about any specific physics problem. Every example
below is deliberately generic (a 1-species heat-equation-style toy
problem) so you can see exactly what's yours to write versus what the
library provides. For a full real-world example, see `trastuzumab/`,
which is built entirely on the patterns in this manual.

## Table of contents

1. [Installation](#1-installation)
2. [The five things you write for a new problem](#2-the-five-things-you-write-for-a-new-problem)
3. [Building a network backbone](#3-building-a-network-backbone)
4. [Defining your physics: PDE, InitialCondition, BoundaryCondition](#4-defining-your-physics)
5. [Constructing and training a PINN](#5-constructing-and-training-a-pinn)
6. [Hard-constraining IC/BC instead of soft loss terms](#6-hard-constraining-icbc)
7. [Optional training techniques](#7-optional-training-techniques)
8. [Checkpointing](#8-checkpointing)
9. [Evaluating against a reference solution](#9-evaluating-against-a-reference-solution)
10. [Full worked example, start to finish](#10-full-worked-example)

---

## 1. Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

This installs both `pinnpy` and (if present in the same repo) any
domain-specific sibling package, e.g. `trastuzumab`, as editable installs.

```python
import pinnpy
from pinnpy.pinns import ForwardPinn, PDE, InitialCondition, BoundaryCondition
```

---

## 2. The five things you write for a new problem

`pinnpy` provides the orchestration (loss aggregation, training loop,
checkpointing, optional advanced training techniques). For *your*
physics problem, you write:

1. A **PDE residual function** -- given a network and a batch of
   interior points, return the PDE residual(s).
2. Zero or more **initial condition residual functions**.
3. Zero or more **boundary condition residual functions**.
4. The **Datasets** each of the above is evaluated on (via
   `DatasetSampler`, or your own point-generation code).
5. A **weights dict** telling the trainer how to balance the loss
   terms against each other.

Everything else -- the network, the optimizer loop, checkpointing,
optional causal weighting / gradient-norm balancing / adaptive
sampling -- `pinnpy` already provides.

---

## 3. Building a network backbone

The simplest backbone is `FCNN`, a standard feedforward network:

```python
from pinnpy.neural_nets import FCNN

net = FCNN(
    in_dim=2,        # (r, t) -- or however many inputs your problem has
    out_dim=1,        # number of solution species
    n_layers=4,
    n_neurons=64,
    initialization='xavier_normal',
)
```

`ModifiedMLP` is a drop-in alternative with a gated-encoder architecture
that costs more compute but tends to reduce PDE residuals further:

```python
from pinnpy.neural_nets import ModifiedMLP

net = ModifiedMLP(in_dim=2, out_dim=1, n_layers=4, n_neurons=64, initialization='xavier_normal')
```

### Optional: Fourier feature embedding, for high-frequency solutions

```python
from pinnpy.embeddings.fourier import FourierFeatures

net = FCNN(
    in_dim=2, out_dim=1, n_layers=4, n_neurons=64,
    initialization='xavier_normal',
    input_transformation=FourierFeatures(input_dim=2, mapping_size=64, sigma=5.0),
)
```

### Optional: Random Weight Factorization

```python
net = FCNN(in_dim=2, out_dim=1, n_layers=4, n_neurons=64, initialization='xavier_normal', use_rwf=True)
```

### Optional: an architectural hard constraint

By default, `net(r, t)` returns the raw network output -- nothing is
enforced architecturally. If your problem has, say, a homogeneous
initial condition (`u(r, 0) = 0` for every `r`), you can bake that in
so it's *never* violated, rather than trained toward as a soft loss
term:

```python
from pinnpy.hard_constraints import zero_at_t0

net = FCNN(..., hard_constraint_fn=zero_at_t0)
```

`zero_at_t0` is just `lambda r, t, u: t * u` -- write your own
`(r, t, u) -> u` callable for anything more specific (e.g. a non-zero
or non-constant IC).

You will normally let `ForwardPinn` build this net for you (see
Section 5) rather than construct it by hand -- the manual construction
above is shown so you know what's available to customize.

---

## 4. Defining your physics

### 4.1 The PDE residual

A `PDE` wraps a callable and the interior collocation dataset it's
evaluated on:

```python
import torch
from pinnpy.pinns import PDE
from pinnpy.datasets import DatasetSampler

def heat_residual(net, dataset, diffusivity, **kwargs) -> dict[str, torch.Tensor]:
    """du/dt = diffusivity * d2u/dr2"""
    r = dataset.data.clone()[:, 0:1].requires_grad_(True)
    t = dataset.data.clone()[:, 1:2].requires_grad_(True)
    u = net(r, t)[:, 0:1]
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_r = torch.autograd.grad(u, r, torch.ones_like(u), create_graph=True)[0]
    u_rr = torch.autograd.grad(u_r, r, torch.ones_like(u_r), create_graph=True)[0]
    return {'u': u_t - diffusivity * u_rr}

sampler = DatasetSampler(seed=42)

pde = PDE(
    residual_fn=heat_residual,
    dataset=sampler.sample_collocation_points(n_points=10_000),
    kwargs={'diffusivity': 0.01},   # fixed extras, always forwarded
)
```

**Rules for a `residual_fn`:**
- Signature: `(net, dataset, **kwargs) -> dict[str, Tensor]`.
- Return one tensor per solution species, keyed by a name you choose.
  **The number of keys must match `net`'s output width** -- get this
  wrong and you'll hit a tensor-shape error the first time you train.
- `**kwargs` is required even if unused, so `pinnpy` can forward
  per-call extras (see `loss_fn`'s `**pde_kwargs`, Section 5) without
  every `residual_fn` needing to declare every possible keyword.

### 4.2 Initial conditions

```python
from pinnpy.pinns import InitialCondition

def initial_residual(net, dataset, ic_func, **kwargs) -> dict[str, torch.Tensor]:
    r = dataset.r
    t = dataset.t
    u = net(r, t)[:, 0:1]
    return {'u': u - ic_func(r)}

ic = InitialCondition(
    name='ic',
    dataset=sampler.sample_initial_points(n_points=200),
    ic_func=lambda r: torch.zeros_like(r),
    residual_fn=initial_residual,
)
```

Pass a **list** of `InitialCondition`s to your `PINN` -- zero if your
problem is steady-state, more than one if you need to match multiple
snapshots. Each needs a unique `name`.

### 4.3 Boundary conditions

```python
from pinnpy.pinns import BoundaryCondition

def dirichlet_residual(net, dataset, target=0.0, **kwargs) -> torch.Tensor:
    r = dataset.r
    t = dataset.t
    u = net(r, t)[:, 0:1]
    return u - target

bc = BoundaryCondition(
    name='right_edge',
    dataset=sampler.sample_surface_points(n_points=200),
    residual_fn=dirichlet_residual,
    kwargs={'target': 0.0},
)
```

Same list pattern as ICs. A `BoundaryCondition.residual_fn` returns a
single `Tensor` (not a dict) -- a BC typically constrains one channel.

---

## 5. Constructing and training a PINN

```python
from pinnpy.pinns import ForwardPinn
from pinnpy.trainer import Trainer
import torch

pinn = ForwardPinn(
    pde=pde,
    n_species=1,                          # must match heat_residual's dict key count
    initial_conditions=[ic],
    boundary_conditions=[bc],
    layers=4, neurons=64,                  # sizes the default net, since we didn't pass net=...
)

weights = {'pde_u': 1.0, 'ic_u': 1.0, 'right_edge': 1.0}

trainer = Trainer(pinn, weights=weights)
optimizer = torch.optim.Adam(pinn.net.parameters(), lr=1e-3)
trainer.train(optimizer, epochs=20_000, log_every=200, diffusivity=0.01)
```

Notice `diffusivity=0.01` is passed to `trainer.train(...)` directly --
it's forwarded as `**pde_kwargs` all the way through to
`heat_residual`. Anything your `residual_fn` needs per-call (not fixed
at `PDE(kwargs=...)` construction time) goes here.

**Weight-dict keys**, generated automatically by `loss_fn` based on
what you supplied:
- `'pde_{species}'` for each key your PDE's `residual_fn` returns
  (or a single `'pde'` if `causal=True` -- see Section 7)
- `'{ic.name}_{species}'` for each IC, per species key
- `bc.name` for each BC

If you supply your own network (`net=...`) instead of letting
`ForwardPinn` build one, `n_species` is ignored.

---

## 6. Hard-constraining IC/BC

Two ways to make an IC/BC architecturally guaranteed rather than a
soft loss term:

**Option A -- `hard_constraint_fn` on the network** (Section 3), for
simple, purely-`t`-shaped constraints.

**Option B -- `ConstrainedNet`**, for anything involving spatial
derivatives too (e.g. Neumann/Robin BCs), using a subtractive-form
Theory of Functional Connections (TFC) correction:

```python
from pinnpy.constrained_net import ConstrainedNet

constrained = ConstrainedNet(
    inner_net=pinn.net,
    beta=..., c_sol_star=..., eps=0.01,
    enforce=('ic', 'neumann', 'robin'),
)
```

Either way, once a condition is hard-constrained, tell `PINN` to skip
computing it as a loss term:

```python
pinn = ForwardPinn(..., hard_conditions=('ic', 'right_edge'))
```

`hard_conditions` names must match your `InitialCondition`/
`BoundaryCondition` `.name`s exactly. Their loss contribution is fixed
at `0.0` -- `loss_fn` never calls their `residual_fn`, and you don't
need to supply weights for them.

---

## 7. Optional training techniques

All three live in `pinnpy.training`, are fully optional, and compose
with each other.

### Temporal causal weighting

Fixes: a PINN learning later timesteps before it's correctly learned
earlier ones.

```python
pinn = ForwardPinn(..., causal=True, n_chunks=24, causal_eps=1.0)
```

Once enabled, `loss_fn` reports a single `'pde'` key instead of
per-species `'pde_{species}'` keys -- update your weights dict
accordingly.

### Gradient-norm loss balancing

Fixes: one loss term (often the PDE residual) dominating training
because its gradients are naturally larger than another term's.

```python
trainer = Trainer(pinn, weights=weights, use_gradnorm=True, gradnorm_update_every=1000)
```

`trainer.weights` is rebalanced automatically every
`gradnorm_update_every` steps -- you don't touch it yourself.

### RAR-G (residual-based adaptive sampling)

Fixes: a fixed collocation set that never concentrates points where
the PDE is currently violated most.

```python
from pinnpy.training.sampling import rar_g

history = rar_g(
    pinn, trainer, optimizer,
    n_rounds=10, n_dense=20_000, m_add=500,
    rnd_epochs=1000, warmup_epochs=20_000,
    diffusivity=0.01,   # forwarded as a pde_kwarg, same as trainer.train's
)
```

---

## 8. Checkpointing

```python
trainer.save('models/my_run.pt')
```

```python
reloaded = ForwardPinn.load('models/my_run.pt')
prediction = reloaded.forward(r, t)
```

`load()` reconstructs whichever `MLP` subclass (`FCNN`, `ModifiedMLP`)
actually trained the checkpoint, and wraps it in a `ConstrainedNet`
using whatever `constrained_net_kwargs` were saved -- you don't need
to remember or re-specify them. **Known limitation:** input/output
transformations (e.g. a Fourier embedding) and RWF settings are not
yet serialized -- if your original net used either, `load()` won't
reconstruct them.

`load()` skips `__init__` entirely (it's for evaluation, not further
training) -- no PDE/IC/BC objects are reattached to the reloaded PINN.

---

## 9. Evaluating against a reference solution

If you have a reference solution (e.g. from a classical solver)
sampled on the same `(r, t)` grid your PINN was trained on:

```python
from pinnpy.evaluation import Evaluator

evaluator = Evaluator(
    species=('u',),
    reference_sol={'u': reference_array},   # shape (N, n_t)
    r=spatial_nodes,                         # shape (N,)
    t_grid=time_grid,                        # shape (n_t,)
)
scores = evaluator.score(pinn)
print(scores['global_l2'])       # overall relative L2 error
print(scores['per_species'])     # {'u': relative L2}
print(scores['per_snapshot'])    # {snapshot fraction: relative L2}
```

Comparing several models at once:

```python
results = evaluator.compare({'FCNN run': pinn_a, 'ModifiedMLP run': pinn_b})
```

If you're wiring up a *specific* problem's reference solver (like
`trastuzumab/fdm.py`), write a small convenience constructor that does
your solver's array-splitting and calls `Evaluator(...)` for you --
see `trastuzumab/evaluation.py::build_evaluator` as a worked example
of this pattern.

---

## 10. Full worked example

Putting Sections 3-8 together, start to finish, for the 1-species heat
equation used throughout this manual:

```python
import torch
from pinnpy.pinns import ForwardPinn, PDE, InitialCondition, BoundaryCondition
from pinnpy.trainer import Trainer
from pinnpy.datasets import DatasetSampler


def heat_residual(net, dataset, diffusivity, **kwargs):
    r = dataset.data.clone()[:, 0:1].requires_grad_(True)
    t = dataset.data.clone()[:, 1:2].requires_grad_(True)
    u = net(r, t)[:, 0:1]
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_r = torch.autograd.grad(u, r, torch.ones_like(u), create_graph=True)[0]
    u_rr = torch.autograd.grad(u_r, r, torch.ones_like(u_r), create_graph=True)[0]
    return {'u': u_t - diffusivity * u_rr}


def initial_residual(net, dataset, ic_func, **kwargs):
    u = net(dataset.r, dataset.t)[:, 0:1]
    return {'u': u - ic_func(dataset.r)}


def dirichlet_residual(net, dataset, target=0.0, **kwargs):
    u = net(dataset.r, dataset.t)[:, 0:1]
    return u - target


sampler = DatasetSampler(seed=42)

pde = PDE(residual_fn=heat_residual, dataset=sampler.sample_collocation_points(n_points=10_000), kwargs={'diffusivity': 0.01})
ic = InitialCondition(name='ic', dataset=sampler.sample_initial_points(n_points=200), ic_func=lambda r: torch.zeros_like(r), residual_fn=initial_residual)
left = BoundaryCondition(name='left', dataset=sampler.sample_center_points(n_points=200), residual_fn=dirichlet_residual, kwargs={'target': 0.0})
right = BoundaryCondition(name='right', dataset=sampler.sample_surface_points(n_points=200), residual_fn=dirichlet_residual, kwargs={'target': 1.0})

pinn = ForwardPinn(pde=pde, n_species=1, initial_conditions=[ic], boundary_conditions=[left, right], layers=4, neurons=64)

weights = {'pde_u': 1.0, 'ic_u': 1.0, 'left': 1.0, 'right': 1.0}
trainer = Trainer(pinn, weights=weights, use_gradnorm=True)
optimizer = torch.optim.Adam(pinn.net.parameters(), lr=1e-3)
trainer.train(optimizer, epochs=20_000, log_every=500)
trainer.save('models/heat_equation.pt')
```

For the full real-world version of this pattern -- a 3-species coupled
system, Neumann/Robin BCs, `ConstrainedNet` hard constraints, causal
weighting, and RAR-G all composed together -- read `trastuzumab/`
end to end, starting from `trastuzumab/residuals.py`.