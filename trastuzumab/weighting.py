"""
Gradient-norm-based adaptive loss weighting, per Wang, Sankaran, Wang &
Perdikaris (2023), "An Expert's Guide to Training Physics-Informed Neural
Networks", Sec 5.2, eq (2.12)-(2.15).
 
The paper decomposes the total loss into three groups -- initial condition
(ic), boundary condition (bc), and PDE residual (r) -- and computes one
lambda scalar per group such that, after weighting, the *norm* of each
group's backpropagated gradient is equal across all three:
 
    ||lambda_ic * grad_theta(L_ic)|| = ||lambda_bc * grad_theta(L_bc)||
                                      = ||lambda_r  * grad_theta(L_r)||
                                      = ||grad(L_ic)|| + ||grad(L_bc)|| + ||grad(L_r)||   (5.3)
 
which is achieved by:
 
    lambda_hat_g = (||grad L_ic|| + ||grad L_bc|| + ||grad L_r||) / ||grad L_g||   (2.12)-(2.14)
 
Since the optimal weights aren't knowable in advance (no validation set
exists for a forward PDE problem, per Sec 3/5.2's motivation), lambda_hat
is recomputed periodically from the network's *current* gradients and
folded into a running average rather than fixed once:
 
    lambda_new = alpha * lambda_old + (1 - alpha) * lambda_hat_new         (2.15)
 
This groups at the same granularity as the paper (ic / bc / r), which is
coarser than this codebase's mse_loss keys (ic0/ic1/ic2, center/surface,
pde0/pde1/pde2 or pde). The caller is expected to sum sub-terms into these
three groups before calling gradient_norm_lambdas / GradNormWeighter.step,
then broadcast the resulting lambda_ic/lambda_bc/lambda_r back onto the
finer w dict keys mse_loss expects.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import torch

@dataclass(frozen=True)
class GroupLambdas:
    ic: float
    bc: float
    r: float

    def as_dict(self):
        return asdict(self)
    
    
def _grad_norm(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> float:

    """L2 norm of d(loss)/d(params), flattened and concatenated across all
    parameter tensors, returned as a plain float rather than a 0-dim Tensor."""

    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    flat_list = [g.flatten() for g in grads if g is not None]
    if not flat_list:
        return 0.0
    return torch.cat(flat_list).norm().item()


def gradient_norm_lambdas(group_losses: dict[str, torch.Tensor], params: list[torch.nn.Parameter], previous: GroupLambdas | None) -> GroupLambdas:

    """Computes lambda_hat_g for each group, per eq (2.12)-(2.14).
 
    Args:
        group_losses: {'ic': L_ic, 'bc': L_bc, 'r': L_r} -- each a scalar
            tensor already summed across whatever sub-terms make up that
            group, with grad history intact (do NOT .detach() these).
        params: the network's trainable parameters, e.g.
            list(pinn.net.parameters()).
 
    Returns:
        GroupLambdas(ic=..., bc=..., r=...) -- already detached (plain
        python floats), since these become fixed coefficients for the next
        optimizer step, not something to backprop through themselves.
    """

    norms: dict[str, float] = {name: _grad_norm(loss, params) for name, loss in group_losses.items()}
    total = sum(norms.values())

    result: dict[str, float] = {}
    for name, n in norms.items():
        if n == 0.0:
            fallback = getattr(previous, name) if previous is not None else 1.0
            result[name] = fallback
        else:
            result[name] = total / n

    return GroupLambdas(**result)


class GradNormWeighter:

    """Stateful wrapper: owns the running lambda values and the moving-average
    update (eq 2.15), so the training loop doesn't have to thread
    lambda_old through by hand.
 
    Usage:
        weighter = GradNormWeighter(alpha=0.9, update_every=1000)
        ...
        ## inside the training loop, after computing individual_loss_terms
        ## (or a forward pass that produces per-group losses with grad intact):
        group_losses = {
            'ic': individual_loss_terms['ic0'] + individual_loss_terms['ic1'] + individual_loss_terms['ic2'],
            'bc': individual_loss_terms['center'] + individual_loss_terms['surface'],
            'r':  individual_loss_terms.get('pde', individual_loss_terms['pde0'] + individual_loss_terms['pde1'] + individual_loss_terms['pde2']),
        }
        lambdas = weighter.step(group_losses, params=list(pinn.net.parameters()), iteration=trainer.current_iter)
        ## lambdas is a GroupLambdas -- lambdas.ic / lambdas.bc / lambdas.r,
        ## or lambdas.as_dict() to broadcast onto the finer w dict:
        w['ic0'] = w['ic1'] = w['ic2'] = lambdas.ic
        w['center'] = w['surface'] = lambdas.bc
        w['pde0'] = w['pde1'] = w['pde2'] = lambdas.r   # or w['pde'] if causal
    """

    def __init__(self, alpha: float = 0.9, update_every: int = 1000, init_lambdas: GroupLambdas | None = None):
        self.alpha = alpha
        self.update_every = update_every
        self.lambdas = init_lambdas if init_lambdas is not None else GroupLambdas(ic=1.0, bc=1.0, r=1.0)

    def step(self, group_losses: dict[str, torch.Tensor], params: list[torch.nn.Parameter], iteration: int) -> GroupLambdas:
        if iteration % self.update_every == 0:
            lambda_hat = gradient_norm_lambdas(group_losses, params, previous=self.lambdas)
            self.lambdas = GroupLambdas(
                ic=self.alpha * self.lambdas.ic + (1 - self.alpha) * lambda_hat.ic,
                bc=self.alpha * self.lambdas.bc + (1 - self.alpha) * lambda_hat.bc,
                r=self.alpha * self.lambdas.r + (1 - self.alpha) * lambda_hat.r
            )
        return self.lambdas
