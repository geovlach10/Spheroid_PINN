# 🧬 Trastuzumab Spheroid PINN

A Physics-Informed Neural Network (PINN) that models the spatiotemporal penetration of the antibody trastuzumab into tumor spheroids, coupled to a finite-difference reference solver for validation.

The PDE system (diffusion–reaction–internalization of free, bound, and internalized antibody, in spherical coordinates) is solved with a family of modern PINN training techniques from Wang, Sankaran, Wang & Perdikaris (2023), *"An Expert's Guide to Training Physics-Informed Neural Networks"* — Fourier feature embeddings, Random Weight Factorization, temporal causal weighting, gradient-norm loss balancing, and a gated "Modified MLP" architecture — alongside a hard-constrained (TFC) variant that enforces initial/boundary conditions architecturally rather than as soft loss penalties.

## 🎯 Scope & Motivation

**The biological problem.** Targeted alpha-particle therapy (TAT) delivers cytotoxic α-emitting isotopes to tumor cells via antibody-radioconjugates. α-particles are extremely effective, but only within ~4–5 cell lengths of where they're deposited — so a cancer cell that the antibody never physically reaches goes unkilled regardless of how potent the isotope is. High-affinity antibodies like trastuzumab bind so avidly to the first receptors they encounter that they stall near the tumor surface and never penetrate to the avascular core; deeper cells survive. This transport-limited, not affinity-limited, resistance mechanism is the actual problem being modeled: how far antibody actually gets, and how much of it stays bound versus internalized, over time and radius, sets the ceiling on how much of a spheroid TAT can kill (Kavousanakis, Macher, Kevrekidis & Sofou, 2025).

**Why solve this system with a PINN instead of purely FDM.** The finite-difference solver (`fdm.py`) is kept in this project deliberately, as the ground-truth oracle `Evaluator` scores the PINN against — FDM is not being replaced because it's wrong, but because it's expensive to reuse. A method-of-lines FDM solve is fast for *one* fixed set of physical parameters (receptor density, diffusivity, affinity, spheroid radius), but this research question is inherently a many-query problem: exploring how the optimal antibody-cocktail split ratio shifts across spheroid sizes, receptor expression levels, and affinity combinations means solving the PDE system again for every point in that parameter sweep. A trained PINN is a differentiable, mesh-free surrogate for the solution field — once trained, it evaluates any `(r, t)` (and, in an inverse/parametric extension, any physical-parameter combination) in a single forward pass, with exact derivatives available via autograd rather than finite-difference stencils. That combination — cheap repeated evaluation plus built-in differentiability — is what makes PINNs attractive for parameter sweeps, sensitivity analysis, and eventually inverse fitting (inferring transport parameters from sparse experimental data), none of which are natural fits for a solver that has to be rebuilt from scratch for every parameter combination (Raissi, Perdikaris & Karniadakis, 2019).

The tradeoff is real and openly acknowledged in the literature this project builds on: PINN training is typically slower than a single classical solve, and its forward-problem accuracy doesn't (yet) enjoy the same convergence guarantees mesh refinement gives FDM/FEM. That's precisely why `fdm.py` and `Evaluator` exist here — the FDM solution is the reference truth the PINN is continually checked against, not a discarded baseline.

**Goal of this codebase.** Build a PINN that reproduces the reaction–diffusion–internalization dynamics of the digital-twin model above accurately enough (validated against FDM) to eventually stand in for repeated FDM solves across the receptor-density / affinity / spheroid-size sweeps the underlying research explores — while tracking, and where possible closing, the training-reliability gaps (spectral bias, causality violation, loss-term imbalance, stiff-IC instability) that the PINN training literature has identified as the reasons naive PINNs often fail to match a classical solver's reliability in the first place.

## 📐 Mathematical Model

The system solves three coupled PDEs in normalized spherical coordinates (r̂ ∈ [0,1], t̂ ∈ [0,1]) for the free (Ĉf), bound (Ĉb), and internalized (Ĉi) antibody concentrations.

**PDE 1 — free antibody (diffusion–reaction):**

$$
res_1 = A \cdot \varphi \frac{\partial}{\partial \hat t}\left(\frac{\hat C_f}{\varphi}\right) - B \cdot \frac{\partial}{\partial \hat r}\left(\hat r^2 \varphi \frac{\partial}{\partial \hat r}\left(\frac{\hat C_f}{\varphi}\right)\right) + C \cdot \frac{\hat C_f}{\varphi}\hat r^2 - D \cdot \hat C_b \hat r^2
$$

**PDE 2 — bound antibody (reaction–internalization):**

$$
res_2 = E \cdot \frac{\partial \hat C_b}{\partial \hat t} - F \cdot \frac{\hat C_f}{\varphi} + G \cdot \frac{\hat C_f}{\varphi}\hat C_b - H \cdot \hat C_b
$$

**PDE 3 — internalized antibody:**

$$
res_3 = Q \cdot \frac{\partial \hat C_i}{\partial \hat t} - T \cdot \hat C_b
$$

**Boundary conditions:** Neumann (symmetry) at r̂=0, Robin (surface flux) at r̂=1.
**Initial conditions:** all three species are zero at t̂=0.

φ(r) = 0.44 r^3.2 + 0.56 is the tumor porosity profile. Coefficients A–H, Q, T and the physical constants (D, K_off, K_D, K_int, R_t, P) are defined once in `constants.py` and consumed identically by both the PINN and the FDM reference solver, so the two cannot disagree on physics by construction.

## 🏗️ Architecture

```
pinnpy/
├── neural_nets.py       # BaseMLP hierarchy: FCNN, ModifiedMLP, RWFLinear
├── embeddings/          # Input embeddings (FourierFeatures, ...)
├── constrained_net.py   # ConstrainedNet — hard IC/BC (TFC) decorator
├── datasets.py          # Dataset, DatasetSampler (LHS collocation/IC/BC sampling)
├── residuals.py         # PDE / IC / BC residual operators (autograd-based)
├── causal.py            # Temporal causal weighting (Algorithm 1)
├── weighting.py         # Gradient-norm adaptive loss balancing
├── sampling.py           # RAR-G residual-based adaptive refinement
├── pinns.py              # BasePinn / ForwardPinn / InversePinn — model + physics + persistence
├── trainer.py            # Trainer — optimization loop, logging, checkpointing
├── evaluation.py         # Evaluator — PINN vs. FDM scoring (relative L2, per-species/snapshot)
├── fdm.py                # Finite-difference reference solver (the "oracle")
├── constants.py          # Single source of truth for physical parameters
├── plotting/              # Result & training-diagnostic visualization
└── preprocessing/          # normalization.py — z-score feature/target scaling
```

### Network backbones

`BaseMLP` (abstract) owns everything shared between PINN backbones: constructor bookkeeping, the RWF-vs-plain `nn.Linear` layer factory, weight initialization, the input/output transformation hooks, and the hard-IC output convention (`C(r,0) = 0`, enforced via a final `t * u` multiply). Concrete backbones implement only two hooks (`_build_layers`, `_compute_hidden`):

| Backbone | Description |
|---|---|
| `FCNN` | Standard feedforward network. |
| `ModifiedMLP` | Gated-encoder architecture (paper eqs. 6.7–6.11) — two encoders U, V computed once and re-injected at every hidden layer via a learned gate. Costs more compute; tends to lower PDE residuals. |

Both compose with:
- **`FourierFeatures`** (`embeddings/`) — random Fourier feature input embedding (eq. 4.3), for high-frequency solution components.
- **`RWFLinear`** — Random Weight Factorization (`use_rwf=True`), a drop-in `nn.Linear` replacement giving each layer a self-adaptive effective learning rate (eq. 4.4–4.5).

### Hard constraints

`ConstrainedNet` wraps a trained backbone and enforces the initial condition plus Neumann(r=0)/Robin(r=1) boundary conditions **by construction** (Theory of Functional Connections), rather than as soft loss terms — via subtractive-IC and carrier-function corrections.

> **Known gap:** `ConstrainedNet.forward` currently returns a `(c0, c1, c2)` tuple, while `BaseMLP.forward` returns a single `(N, 3)` tensor. These two contracts don't yet match — `ConstrainedNet` is not currently a `BaseMLP` subclass, and callers (`evaluation.py`, `plotting.py`) unpack it as a tuple. Unifying this is tracked as an open follow-up.

### Loss weighting strategies

- **Temporal causal weighting** (`causal.py`) — reweights the PDE residual loss by time-chunk so later timesteps only "count" once earlier ones are already well-fit, respecting the physical causality that information propagates forward in time (eqs. 2.10–2.11).
- **Gradient-norm balancing** (`weighting.py`) — periodically rescales the IC/BC/PDE loss groups so their backpropagated gradient norms stay balanced, avoiding one term dominating training (eqs. 2.12–2.15).

### Adaptive sampling

- **`DatasetSampler`** — Latin Hypercube sampling for collocation points; structured sampling for IC/center/surface points.
- **RAR-G** (`sampling.py`) — Residual-based Adaptive Refinement with greed: periodically scores a dense candidate cloud by PDE residual and appends the worst-violating points to the collocation set, growing it toward wherever the network currently fits worst.

## 🚀 Getting Started

```bash
# environment
python -m venv venv
source venv/bin/activate

# install
pip install -r requirements.txt
```

### Quickstart

```python
from pinnpy.pinns import ForwardPinn
from pinnpy.neural_nets import FCNN
from pinnpy.embeddings.fourier_features import FourierFeatures
from pinnpy.trainer import Trainer
import torch

net = FCNN(
    in_dim=2, out_dim=3, n_layers=4, n_neurons=64,
    initialization='xavier_normal',
    input_transformation=FourierFeatures(mapping_size=64, sigma=5.0),
    use_rwf=True,
)

pinn = ForwardPinn(
    n_col=10_000, n_initial=200, n_center=200, n_surface=200,
    initial_fn=lambda r: torch.zeros_like(r),
    net=net,
    causal=True, n_chunks=24, causal_eps=1.0,
)

weights = {'pde0': 1.0, 'pde1': 1.0, 'pde2': 1.0,
           'ic0': 1.0, 'ic1': 1.0, 'ic2': 1.0,
           'center': 1.0, 'surface': 1.0}

trainer = Trainer(pinn, weights=weights, use_gradnorm=True)
optimizer = torch.optim.Adam(pinn.net.parameters(), lr=1e-3)
trainer.train(optimizer, epochs=20_000, log_every=200)
trainer.save('models/model.pt')
```

## 📊 Evaluation

`Evaluator` scores a trained PINN against the FDM reference solution (relative L2 error, globally, per-species, and at fixed time snapshots):

```python
from trastuzumab.fdm import run_fdm
from trastuzumab.evaluation import Evaluator

x, sol = run_fdm(m=100, phase_name='uptake', y0=..., t_final=1.0)
evaluator = Evaluator(fdm_x=x, fdm_sol=sol)
scores = evaluator.score(pinn)
```

## 🗺️ Roadmap / Known Issues

- [ ] Unify `ConstrainedNet` under the `BaseMLP` contract (single-tensor `forward`, matching `FCNN`/`ModifiedMLP`).
- [ ] Checkpoint (de)serialization doesn't yet persist backbone type, Fourier, or RWF hyperparameters — a `FCNN`-only checkpoint format currently.
- [ ] `DatasetSampler.sample_collocation_points` uses a fixed seed, so collocation points are static across training rather than resampled each iteration (paper Sec 6.2 recommendation).
- [ ] Add exponential LR decay to the Adam training loop (paper Sec 6.1).
- [ ] Add Chebyshev-node sampling as an alternative to Latin Hypercube.
- [ ] `individual_loss_terms` dict has different keys in causal (`'pde'`) vs. non-causal (`'pde0'/'pde1'/'pde2'`) mode — a schema inconsistency for downstream loggers.

## 📚 References

**Biological / clinical motivation**
- Kavousanakis, M., Macher, R., Kevrekidis, Y., & Sofou, S. (2025). *A Digital Twin to Optimize Treatment Efficacy of Targeted Alpha-particle Therapies by Antibody-Radioconjugate Cocktails Against Solid Tumors.* bioRxiv. https://doi.org/10.64898/2025.12.19.695379

**PINN foundations**
- Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 378, 686–707.
- Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2017). *Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations.* arXiv:1711.10561.

**Training techniques implemented in this codebase**
- Wang, S., Sankaran, S., Wang, H., & Perdikaris, P. (2023). *An Expert's Guide to Training Physics-Informed Neural Networks.* arXiv:2308.08468. (Fourier feature embeddings, Random Weight Factorization, temporal causal weighting, the Modified MLP architecture, and gradient-norm loss balancing, as implemented in `neural_nets.py`, `embeddings/`, `causal.py`, and `weighting.py`.)
- Wang, S., Sankaran, S., & Perdikaris, P. (2022). *Respecting causality is all you need for training physics-informed neural networks.* arXiv:2203.07404.
- Wang, S., Wang, H., Seidman, J.H., & Perdikaris, P. (2022). *Random Weight Factorization Improves the Training of Continuous Neural Representations.*
- Tancik, M., Srinivasan, P.P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R., Barron, J.T., & Ng, R. (2020). *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains.* arXiv:2006.10739.
- Wang, S., Yu, X., & Perdikaris, P. (2022). *When and why PINNs fail to train: A neural tangent kernel perspective.* Journal of Computational Physics, 449, 110768.
- Wu, C., Zhu, M., Tan, Q., Kartha, Y., & Lu, L. (2023). *A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks.* Computer Methods in Applied Mechanics and Engineering, 403, 115671. (RAR-G, `sampling.py`.)
- Lu, L., Pestourie, R., Yao, W., Wang, Z., Verdugo, F., & Johnson, S.G. (2021). *Physics-informed neural networks with hard constraints for inverse design.* arXiv:2102.04626. (Hard-constrained boundary/initial condition enforcement, related to `constrained_net.py`'s TFC approach.)