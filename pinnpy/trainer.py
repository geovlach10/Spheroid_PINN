"""Optimization loop, loss weighting, and logging for PINN training.

Trainer wraps a constructed PINN and owns everything about *how* it's
trained: running optimizer steps against `pinn.loss_fn`, optionally
rebalancing loss-term weights via gradient-norm balancing (eq
2.12-2.15), recording per-epoch/per-chunk history for later plotting,
and checkpointing. Trainer knows nothing about the physics itself --
that's entirely `pinn`'s responsibility (see pinns.py).

Example:
    trainer = Trainer(pinn, weights=my_weights, use_gradnorm=True)
    optimizer = torch.optim.Adam(pinn.net.parameters(), lr=1e-3)
    trainer.train(optimizer, epochs=20_000, log_every=200)
    trainer.save('models/my_run.pt')
"""

from __future__ import annotations
from pathlib import Path
import torch
from .pinns import PINN
from .weighting import GradNormWeighter
from datetime import datetime

_MODELS_DIR = Path(__file__).parent.parent / 'models'

class Trainer:

    """Runs the optimization loop for a PINN, with optional gradient-norm
    loss-term rebalancing, and records training history for later
    inspection/plotting.

    Args:
        pinn: a constructed PINN (ForwardPinn/InversePinn) to train.
        weights: initial per-loss-term weights, keyed exactly like
            `pinn.loss_fn`'s `w` argument (`'pde_{species}'`,
            `'{ic.name}_{species}'`, `bc.name`, or `'pde'` if
            `pinn.causal=True`). Copied, not referenced -- mutating
            the dict you passed in afterward has no effect; use
            `trainer.weights` instead. Mutated in place by
            `_apply_gradnorm` if `use_gradnorm=True`.
        pde_normalized: forwarded as `loss_fn`'s `scaled` kwarg... no
            longer applicable directly (loss_fn takes **pde_kwargs now)
            -- kept here as a Trainer-level convenience flag; thread it
            into your own `pde_kwargs` at the `train()` call site if
            your PDE uses a `scaled`-style kwarg.
        use_gradnorm: if True, rebalance `self.weights` every
            `gradnorm_update_every` iterations so the PDE/IC/BC loss
            groups' backpropagated gradient norms stay balanced (see
            weighting.py).
        gradnorm_alpha, gradnorm_update_every: passed to
            `GradNormWeighter`. Ignored if `use_gradnorm=False`.
    """

    def __init__(self, pinn: PINN, weights: dict[str, float], pde_normalized: bool = False,
                 use_gradnorm: bool = False, gradnorm_alpha: float = 0.9, gradnorm_update_every: int = 1000):
        
        self.pinn = pinn
        self.weights: dict[str, float] = dict(weights)
        self.history: dict[str, list[float]] = {}
        self.current_iter: int = 0
        self.stages: list[dict] = []
        self.pde_normalized: bool = pde_normalized

        # --- causal diagnostics history (only populated when pinn.causal=True) ---
        self.chunk_loss_history: list[torch.Tensor] = []
        self.chunk_weight_history: list[torch.Tensor] = []
        self.chunk_iter: list[int] = []

        # --- gradient-norm loss balancing (eq 2.12-2.15) ---
        self.use_gradnorm = use_gradnorm
        self.gradnorm_weighter = GradNormWeighter(alpha=gradnorm_alpha, update_every=gradnorm_update_every) if use_gradnorm else None
        self.gradnorm_history: list = []

    def train(self, optimizer: torch.optim.Optimizer, epochs: int, log_every: int = 200, profile_every: int=0, **pde_kwargs) -> 'Trainer':
        """Runs `epochs` optimizer steps against `self.pinn.loss_fn`.

        Args:
            optimizer: e.g. `torch.optim.Adam(pinn.net.parameters(), ...)`.
            epochs: number of steps to run.
            log_every, profile_every: logging/profiling cadence, in epochs.
            **pde_kwargs: forwarded to `pinn.loss_fn(w=self.weights, **pde_kwargs)`
                every step -- e.g. `L=1.0` for trastuzumab's PDE, or
                nothing at all for a PDE that doesn't need any.
        """
        self.stages.append({
            'opt': type(optimizer).__name__,
            'epochs': epochs,
            'lr': optimizer.param_groups[0].get('lr'),
            'pde_kwargs': dict(pde_kwargs),
            't_end': self.pinn.upper_bounds[1]
        })

        def closure():
            optimizer.zero_grad()
            loss, _ = self.pinn.loss_fn(w=self.weights, **pde_kwargs)
            loss.backward()
            return loss
        
        for epoch in range(epochs):
            optimizer.step(closure)         # type: ignore[arg-type]
            total_loss, individual_loss_terms = self.pinn.loss_fn(w=self.weights, **pde_kwargs)
            self._record(epoch, total_loss, individual_loss_terms)
            self._record_causal(log_every)
            self._apply_gradnorm()
            self._log(epoch, total_loss, individual_loss_terms, log_every)
            self._check_outpout_profile(epoch, profile_every)
            self.current_iter += 1

        return self

    
    def _apply_gradnorm(self) -> None:
        """Updates self.weights via gradient-norm loss balancing (eq
        2.12-2.15), sourced from self.pinn.meta['raw_loss_terms'] -- the
        *unweighted* per-term losses loss_fn() exposes alongside its
        normal (weighted) return value.

        Group membership is derived from self.pinn's actual
        initial_conditions/boundary_conditions rather than assumed --
        so this works for any number of ICs/BCs/species, not just
        trastuzumab's fixed ic0/ic1/ic2, center/surface, pde0/pde1/pde2.
        Hard-constrained terms are included in raw (loss_fn always
        populates them, at value 0.0) -- including them here is
        harmless, since loss_fn ignores self.weights entirely for any
        hard-constrained key when computing the actual loss.
        """
        if self.gradnorm_weighter is None:
            return
        raw: dict[str, torch.Tensor] | None = self.pinn.meta.get('raw_loss_terms')
        if raw is None:
            return

        pde_terms = [v for k, v in raw.items() if k =='pde' or k.startswith('pde_')]
        ic_terms = [v for ic in self.pinn.initial_conditions for k, v in raw.items() if k.startswith(f'{ic.name}_')]
        bc_terms = [raw[bc.name] for bc in self.pinn.boundary_conditions if bc.name in raw]
        
        group_losses = {
            'r': torch.stack(pde_terms).sum(dim=0) if pde_terms else torch.tensor(0.0),
            'ic': torch.stack(ic_terms).sum(dim=0) if ic_terms else torch.tensor(0.0),
            'bc': torch.stack(bc_terms).sum(dim=0) if bc_terms else torch.tensor(0.0),
        }     

        lambdas = self.gradnorm_weighter.step(group_losses, params=list(self.pinn.net.parameters()), iteration=self.current_iter)
        

        if 'pde' in raw and getattr(self.pinn, 'causal', False):
            self.weights['pde'] = lambdas.r
        else:
            for k in raw:
                if k.startswith('pde_'):
                    self.weights[k] = lambdas.r

        for ic in self.pinn.initial_conditions:
            for k in raw:
                if k.startswith(f'{ic.name}_'):
                    self.weights[k] = lambdas.ic

        for bc in self.pinn.boundary_conditions:
            if bc.name in raw:
                self.weights[bc.name] = lambdas.bc

        if self.current_iter % self.gradnorm_weighter.update_every == 0:
            self.gradnorm_history.append(lambdas)


    def save(self, path: str | Path | None = None) -> None:
        path = _MODELS_DIR / 'model.pt' if path is None else Path(path)
        if not path.is_absolute() and path.parent == Path('.'):
            path = _MODELS_DIR / path
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = self.pinn.to_checkpoint()
        checkpoint['meta'] = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'total_iters': self.current_iter,
            'stages': self.stages,
            'final_total_loss': self.history['total'][-1] if self.history.get('total') else None,
        }
        torch.save(checkpoint, path)
        print(f'checkpoint saved -> {path}')

    def _record(self, epoch: int, total: torch.Tensor, terms: dict[str, torch.Tensor]) -> None:
        self.history.setdefault('total', []).append(total.item())
        for name, value in terms.items():
            self.history.setdefault(name, []).append(value.item())

    def _record_causal(self, every: int) -> None:
        if not getattr(self.pinn, 'causal', False):
            return
        if self.current_iter % every != 0:
            return
        chunk_losses = self.pinn.meta.get('chunk_losses')
        chunk_weights = self.pinn.meta.get('chunk_weights')
        if chunk_losses is None or chunk_weights is None:       
            return 
        self.chunk_loss_history.append(chunk_losses.detach().clone())
        self.chunk_weight_history.append(chunk_weights.detach().clone())
        self.chunk_iter.append(self.current_iter)

    
    def _log(self, epoch: int, total: torch.Tensor, terms: dict[str, torch.Tensor], every: int) -> None:
        if epoch == 0:
            cols = ['epoch', 'total', *terms.keys()]
            header = ' | '.join(f'{c:^11}' for c in cols)
            print('-' * len(header))
            print(header)
            print('-' * len(header))
        if epoch % every == 0:
            cells = [f'{epoch:^11d}', f'{total.item():^11.3e}',
                     *(f'{v.item():^11.3e}' for v in terms.values())]
            print(' | '.join(cells))
    
    def _check_outpout_profile(self, epoch:int, every: int):
        if every == 0:
            return
        if epoch % every == 0:
            self.pinn.check_concentration_profile(epoch=epoch)