# 🧬 Antibody Delivery PINN Digital Twin

A Physics-Informed Neural Network (PINN) that models the spatiotemporal penetration of the antibody trastuzumab into tumor spheroids, coupled to a finite-difference reference solver for validation.

The codebase is split into two packages:

- **`pinnpy`** — a generic, reusable PINN library: network backbones, an injectable PDE/initial-condition/boundary-condition training contract, and a library of optional training techniques (temporal causal weighting, gradient-norm loss balancing, residual-based adaptive sampling). Nothing in `pinnpy` knows what a "trastuzumab" is.
- **`trastuzumab`** — the domain-specific application built on top of `pinnpy`: the physical constants, the actual PDE/IC/BC residual functions, the finite-difference reference solver, and evaluation helpers specific to this problem.

## 🎯 Scope & Motivation

**The biological problem.** Targeted alpha-particle therapy (TAT) delivers cytotoxic α-emitting isotopes to tumor cells via antibody-radioconjugates. α-particles are extremely effective, but only within ~4–5 cell lengths of where they're deposited — so a cancer cell that the antibody never physically reaches goes unkilled regardless of how potent the isotope is. High-affinity antibodies like trastuzumab bind so avidly to the first receptors they encounter that they stall near the tumor surface and never penetrate to the avascular core; deeper cells survive. This transport-limited, not affinity-limited, resistance mechanism is the actual problem being modeled: how far antibody actually gets, and how much of it stays bound versus internalized, over time and radius, sets the ceiling on how much of a spheroid TAT can kill (Kavousanakis, Macher, Kevrekidis & Sofou, 2025).

**Why solve this system with a PINN instead of purely FDM.** The finite-difference solver (`trastuzumab/fdm.py`) is kept in this project deliberately, as the ground-truth oracle `Evaluator` scores the PINN against — FDM is not being replaced because it's wrong, but because it's expensive to reuse for every point in a parameter sweep (receptor density, diffusivity, affinity, spheroid radius). A trained PINN is a differentiable, mesh-free surrogate for the solution field — once trained, it evaluates any `(r, t)` in a single forward pass, with exact derivatives available via autograd. That combination is what makes PINNs attractive for parameter sweeps, sensitivity analysis, and eventually inverse fitting, none of which are natural fits for a solver that has to be rebuilt from scratch for every parameter combination (Raissi, Perdikaris & Karniadakis, 2019).

**Goal of this codebase.** Build a PINN that reproduces the reaction–diffusion–internalization dynamics of the digital-twin model above accurately enough (validated against FDM) to eventually stand in for repeated FDM solves across the receptor-density / affinity / spheroid-size sweeps the underlying research explores.

## 🏗️ Architecture

```
pinnpy/                      # generic PINN library -- no trastuzumab-specific code
├── neural_nets.py             # MLP (abstract), FCNN, ModifiedMLP, RWFLinear
├── hard_constraints.py        # optional hard_constraint_fn helpers (e.g. zero_at_t0)
├── embeddings/                 # input embeddings (FourierFeatures, ...)
├── constrained_net.py         # ConstrainedNet -- hard IC/BC (TFC) decorator
├── datasets.py                 # Dataset, DatasetSampler (LHS collocation/IC/BC sampling)
├── pinns.py                    # PDE, InitialCondition, BoundaryCondition, PINN, ForwardPinn, InversePinn
├── trainer.py                   # Trainer -- optimization loop, gradnorm, logging, checkpointing
├── evaluation.py                # Evaluator -- generic, species-count-agnostic scoring
├── training/                    # optional training-technique library
│   ├── causal.py                  # temporal causal weighting
│   ├── weighting.py                # gradient-norm adaptive loss balancing
│   └── sampling.py                  # RAR-G residual-based adaptive refinement
└── plotting/                    # causal_plotting.py -- training diagnostics

trastuzumab/                   # domain-specific application, built on pinnpy
├── constants.py                 # single source of truth for physical parameters
├── fdm.py                        # finite-difference reference solver (the "oracle")
├── residuals.py                  # concrete PDE/IC/BC residual_fn implementations
└── evaluation.py                  # build_evaluator(...) + spatial-distribution plotting
```

### The PDE / InitialCondition / BoundaryCondition contract

`PINN` (in `pinnpy/pinns.py`) owns loss aggregation, checkpointing, and nothing about physics. Every piece of actual physics is supplied by the caller as plain data:

| Object | Holds | Frozen? |
|---|---|---|
| `PDE` | `residual_fn` + its own collocation `dataset` | No — RAR-G grows `pde.dataset` in place |
| `InitialCondition` | `name`, `dataset`, `ic_func`, `residual_fn` | Yes — `.name` is a loss-key namespace |
| `BoundaryCondition` | `name`, `dataset`, `residual_fn` | Yes — same reason |

`PINN` accepts a **list** of `InitialCondition`s and `BoundaryCondition`s — any count, not a fixed structure — and works for any number of solution species, driven entirely by however many keys a given `residual_fn` returns. Each `residual_fn`'s expected call signature is typed via a `Protocol` (`PDEResidualFn`/`ICResidualFn`/`BCResidualFn`), so a mismatched signature is flagged by your editor at the point you write it, not discovered as a runtime error three calls deep inside `loss_fn`.

```python
pinn.loss_fn(w=weights, **pde_kwargs)
```

`w` is a per-term weight dict, keyed `'pde_{species}'`, `'{ic.name}_{species}'`, and `bc.name`. `**pde_kwargs` forwards straight through to your PDE's `residual_fn` — e.g. trastuzumab's PDE expects `L=...` (a receptor-load scaling factor); a different problem might need nothing at all.

### Network backbones

`MLP` (abstract) owns everything shared between backbones: construction bookkeeping, the RWF-vs-plain `nn.Linear` layer factory, weight init, and the input/output transformation hooks. Concrete backbones implement only `_build_layers`/`_compute_hidden`:

| Backbone | Description |
|---|---|
| `FCNN` | Standard feedforward network. |
| `ModifiedMLP` | Gated-encoder architecture (paper eqs. 6.7–6.11) — two encoders computed once and re-injected at every hidden layer via a learned gate. Costs more compute; tends to lower PDE residuals. |

Both compose with `FourierFeatures` (`pinnpy/embeddings/`) for high-frequency solution components, and `use_rwf=True` for Random Weight Factorization.

### Hard constraints

Two independent mechanisms, at different levels:

- **`ConstrainedNet`** wraps a trained backbone and enforces IC + Neumann(r=0)/Robin(r=1) BCs architecturally (subtractive-form TFC), rather than as soft loss terms. Trastuzumab's usual setup uses this for all three.
- **`hard_constraint_fn`** — an optional hook on `MLP` itself (`hard_constraint_fn(r, t, u) -> u`, called last in `forward`). Not applied by default. `pinnpy/hard_constraints.py` provides `zero_at_t0` (`u * t`) for problems with a homogeneous IC at `t=0`; write your own for anything else (e.g. a non-zero or non-constant IC).

### Training techniques (`pinnpy/training/`)

Optional, composable, opt-in — a PINN trains fine without any of them:

- **Temporal causal weighting** (`causal.py`) — reweights the PDE loss by time-chunk so later timesteps only "count" once earlier ones are already well-fit. `PINN(..., causal=True, n_chunks=...)`.
- **Gradient-norm balancing** (`weighting.py`) — periodically rescales the PDE/IC/BC loss groups so their backpropagated gradient norms stay balanced. `Trainer(..., use_gradnorm=True)`.
- **RAR-G** (`sampling.py`) — periodically scores a dense candidate cloud by PDE residual and appends the worst-violating points to the collocation set. `training.sampling.rar_g(pinn, trainer, optimizer, ...)`.

## 🚀 Getting Started

```bash
# environment
python -m venv venv
source venv/bin/activate

# install (editable, both packages discovered via pyproject.toml)
pip install -e .
```

### Quickstart

```python
import torch
from pinnpy.pinns import ForwardPinn, PDE, InitialCondition, BoundaryCondition
from pinnpy.trainer import Trainer
from pinnpy.datasets import DatasetSampler
from trastuzumab.residuals import (
    pde_residual, initial_residual, center_neumann_residual, surface_robin_residual,
)

sampler = DatasetSampler(seed=42)

pde = PDE(
    residual_fn=pde_residual,
    dataset=sampler.sample_collocation_points(n_points=10_000),
)

ic = InitialCondition(
    name='ic',
    dataset=sampler.sample_initial_points(n_points=200),
    ic_func=lambda r: torch.zeros_like(r),
    residual_fn=initial_residual,
)
center = BoundaryCondition(
    name='center', dataset=sampler.sample_center_points(n_points=200),
    residual_fn=center_neumann_residual, kwargs={'target': 0.0},
)
surface = BoundaryCondition(
    name='surface', dataset=sampler.sample_surface_points(n_points=200),
    residual_fn=surface_robin_residual,
)

pinn = ForwardPinn(
    pde=pde, n_species=3,
    initial_conditions=[ic], boundary_conditions=[center, surface],
    hard_conditions=(),   # soft-enforced -- everything above is a real loss term
)

weights = {
    'pde_c0': 1.0, 'pde_c1': 1.0, 'pde_c2': 1.0,
    'ic_c0': 1.0, 'ic_c1': 1.0, 'ic_c2': 1.0,
    'center': 1.0, 'surface': 1.0,
}

trainer = Trainer(pinn, weights=weights, use_gradnorm=True)
optimizer = torch.optim.Adam(pinn.net.parameters(), lr=1e-3)
trainer.train(optimizer, epochs=20_000, log_every=200, L=1.0)
trainer.save('models/model.pt')
```

## 📊 Evaluation

`build_evaluator` wraps a raw FDM solve into a generic `Evaluator` (relative L2 error, globally, per-species, and at fixed time snapshots):

```python
from trastuzumab.fdm import run_fdm
from trastuzumab.evaluation import build_evaluator

x, sol = run_fdm(m=100, phase_name='uptake', y0=..., t_final=1.0)
evaluator = build_evaluator(fdm_x=x, fdm_sol=sol)
scores = evaluator.score(pinn)
```

## 🗺️ Roadmap / Known Issues

- [ ] `DatasetSampler.sample_collocation_points` uses a fixed seed, so collocation points are static across training rather than resampled each iteration (paper Sec 6.2 recommendation).
- [ ] Add exponential LR decay to the Adam training loop (paper Sec 6.1).
- [ ] Add Chebyshev-node sampling as an alternative to Latin Hypercube.
- [ ] `__post_init__` fail-fast signature validation for `PDE`/`InitialCondition`/`BoundaryCondition` — Protocol-based editor-time checking exists, but nothing raises at construction time for a genuinely mismatched `residual_fn`.
- [ ] `resample_datasets` only grows `pde.dataset`; IC/BC growth needs an opt-in `resample_fn` field on those dataclasses.
- [ ] `Trainer.pde_normalized` is currently dead state — nothing reads it since `loss_fn` takes generic `**pde_kwargs`.
- [ ] `tests/test_constrained_net.py::test_checkpoint_schema_unchanged` fails — a checkpoint built from one architecture is loaded into a differently-sized fresh network; pre-existing, unrelated to the PDE/IC/BC refactor.
- [ ] `causal_plotting.py`, `plotting.py`, and `weighting.py`'s docstring example still reference stale names or the pre-split package layout.
- [ ] Project venv is on Python 3.9.6, producing `PyparsingDeprecationWarning` noise from matplotlib's dependency chain on every test run — consider upgrading.

## 📚 References

**Biological / clinical motivation**
- Kavousanakis, M., Macher, R., Kevrekidis, Y., & Sofou, S. (2025). *A Digital Twin to Optimize Treatment Efficacy of Targeted Alpha-particle Therapies by Antibody-Radioconjugate Cocktails Against Solid Tumors.* bioRxiv. https://doi.org/10.64898/2025.12.19.695379

**PINN foundations**
- Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 378, 686–707.
- Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2017). *Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations.* arXiv:1711.10561.

**Training techniques implemented in this codebase**
- Wang, S., Sankaran, S., Wang, H., & Perdikaris, P. (2023). *An Expert's Guide to Training Physics-Informed Neural Networks.* arXiv:2308.08468. (Fourier feature embeddings, Random Weight Factorization, temporal causal weighting, the Modified MLP architecture, and gradient-norm loss balancing.)
- Wang, S., Sankaran, S., & Perdikaris, P. (2022). *Respecting causality is all you need for training physics-informed neural networks.* arXiv:2203.07404.
- Wang, S., Wang, H., Seidman, J.H., & Perdikaris, P. (2022). *Random Weight Factorization Improves the Training of Continuous Neural Representations.*
- Tancik, M., Srinivasan, P.P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R., Barron, J.T., & Ng, R. (2020). *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains.* arXiv:2006.10739.
- Wang, S., Yu, X., & Perdikaris, P. (2022). *When and why PINNs fail to train: A neural tangent kernel perspective.* Journal of Computational Physics, 449, 110768.
- Wu, C., Zhu, M., Tan, Q., Kartha, Y., & Lu, L. (2023). *A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks.* Computer Methods in Applied Mechanics and Engineering, 403, 115671. (RAR-G.)
- Lu, L., Pestourie, R., Yao, W., Wang, Z., Verdugo, F., & Johnson, S.G. (2021). *Physics-informed neural networks with hard constraints for inverse design.* arXiv:2102.04626. (Hard-constrained boundary/initial condition enforcement, related to `ConstrainedNet`'s TFC approach.)