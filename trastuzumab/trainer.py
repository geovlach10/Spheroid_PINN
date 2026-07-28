from __future__ import annotations
from pathlib import Path
import torch
from .pinns import BasePinn
from .weighting import GradNormWeighter
from datetime import datetime

_MODELS_DIR = Path(__file__).parent.parent / 'models'

class Trainer:

    """it is responsible for optimization, loss weighting and logging"""

    def __init__(self, pinn: BasePinn, weights: dict[str, float], pde_normalized: bool = False,
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

    def train(self, optimizer: torch.optim.Optimizer, epochs: int, L: float = 1.0, log_every: int = 200, profile_every: int=0) -> 'Trainer':
        self.stages.append({
            'opt': type(optimizer).__name__,
            'epochs': epochs,
            'lr': optimizer.param_groups[0].get('lr'),
            'L': L,
            't_end': self.pinn.upper_bounds[1]
        })

        def closure():
            optimizer.zero_grad()
            loss, _ = self.pinn.mse_loss(w=self.weights, L=L, scaled=self.pde_normalized)
            loss.backward()
            return loss
        
        for epoch in range(epochs):
            optimizer.step(closure)         # type: ignore[arg-type]
            total_loss, individual_loss_terms = self.pinn.mse_loss(w=self.weights, L=L, scaled=self.pde_normalized)
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
        *unweighted* per-term losses mse_loss() exposes alongside its
        normal (weighted) return value."""

        if self.gradnorm_weighter is None:
            return
        raw: dict[str, torch.Tensor] | None = self.pinn.meta.get('raw_loss_terms')
        if raw is None:
            return
        
        group_losses = {
            'ic': raw['ic0'] + raw['ic1'] + raw['ic2'],
            'bc': raw['center'] + raw['surface'],
            'r': raw.get('pde', raw['pde0'] + raw['pde1'] + raw['pde2']),
        }       

        lambdas = self.gradnorm_weighter.step(group_losses, params=list(self.pinn.net.parameters()), iteration=self.current_iter)
        self.weights['ic0'] = self.weights['ic1'] = self.weights['ic2'] = lambdas.ic
        self.weights['center'] = self.weights['surface'] = lambdas.bc

        if 'pde' in raw and getattr(self.pinn, 'causal', False):
            self.weights['pde'] = lambdas.r
        else:
            self.weights['pde0'] = self.weights['pde1'] = self.weights['pde2'] = lambdas.r

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
        if epoch % every == 0:
            self.pinn.check_concentration_profile(epoch=epoch, L=1.0)