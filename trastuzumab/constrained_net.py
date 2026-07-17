from __future__ import annotations
import torch
import torch.nn as nn
from .neural_nets import FCNN

def phi(r: torch.Tensor):
    """Porosity profile phi(r) = 0.44 r^3.2 + 0.56.
    Works on numpy arrays (FDM) and torch tensors (PINN)
    """
    return 0.44 * r ** 3.2 + 0.56

class ConstrainedNet(nn.Module):
    """Decorator over a raw FCNN enforcing IC + Neumann(r=0) + Robin(r=1) on
    channel 0 by construction (TFC constrained expression), and the homogeneous
    IC on channels 1,2 (subtractive form).

    Channel 0 is returned as PHYSICAL c0 = phi(r) * u0, where the CE is built in
    u0 = c0/phi space (the variable the BC operators act on). Downstream `pde`
    recovers u0 = c0/phi exactly (phi > 0), so no residual code changes.

    enforce: subset of {'ic','neumann','robin'}. IC is always applied (the
    subtractive base); 'neumann'/'robin' toggle their correction terms. Pass-1
    behaviour = enforce=('ic',).
    """

    def __init__(self, inner_net: FCNN, beta: float, c_sol_star: float, eps: float, n=6,  enforce: tuple[str, ...] = ('ic', 'neumann', 'robin')):
        super().__init__()
        self.inner_net = inner_net
        self.beta = beta
        self.C_SOL_STAR = c_sol_star
        self.eps = eps
        self.n = n                      # // robin carrier localization exponent.
        self.enforce = enforce

    def forward(self, r: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t0 = torch.zeros_like(t)
        pred_t = self.inner_net(r, t)
        g0_t = pred_t[:, 0:1]; g1_t = pred_t[:, 1:2]; g2_t =  pred_t[:, 2:3]            # // inner FCNN output at (r, t)
        
        pred_t0 = self.inner_net(r, t0)
        g0_0 = pred_t0[:, 0:1]; g1_0 = pred_t0[:, 1:2]; g2_0 = pred_t0[:, 2:3]                                        # // inner FCNN output at (r, 0) -> IC footprint

        u0 = g0_t - g0_0                                                                 # // subtractive IC: u0(r, 0) = 0

        if 'neumann' in self.enforce:
            _, Ng_t = self._get_inner_net_pred_and_grad(0.0, t)
            _, Ng_0 = self._get_inner_net_pred_and_grad(0.0, t0)
            u0 = u0 - self._phi_N(r) * (Ng_t - Ng_0)                     # // Neumann(r=0) correction
        
        if 'robin' in self.enforce:
            g0_1t, dg0_1t_dr = self._get_inner_net_pred_and_grad(1.0, t)
            g0_10, dg0_10_dr = self._get_inner_net_pred_and_grad(1.0, t0)
            Rg_t = dg0_1t_dr + self.beta * g0_1t
            Rg_0 = dg0_10_dr + self.beta * g0_10

            gamma = self.beta * self.C_SOL_STAR * (1.0 - torch.exp(-t / self.eps))
            A = gamma * self._psi(r)
            localized_robin_correction = self._psi(r) * (Rg_t - Rg_0)
            
            u0 = u0 + A - localized_robin_correction
    

        c0 = phi(r) * u0
        c1 = g1_t - g1_0                                                 # // subtractive IC: c1(r, 0) = 0
        c2 = g2_t - g2_0                                                 # // subtractive IC: c2(r, 0) = 0
        return c0, c1, c2
    
    def _psi(self, r):
        '''Localized robin carrier psi(r) = r**n / (n + beta)'''
        return r**self.n / (self.n + self.beta)
    
    def _phi_N(self, r: torch.Tensor):
        '''Neumann carrier on r=0'''
        return  r - (1 + self.beta) / self.beta
    
    def _get_inner_net_pred_and_grad(self, r_val: float, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        '''returns tuple(g0, dg0_dr) at r=r_val, t=t)
        g0 is the inner net's 1st output.'''
        with torch.enable_grad():
            r = torch.full_like(t, r_val).requires_grad_(True)
            g0 = self.inner_net(r, t)[:, 0:1]
            dg0_dr = torch.autograd.grad(g0, r, torch.ones_like(g0), create_graph=True)[0]
        if not torch.is_grad_enabled():
            g0, dg0_dr = g0.detach(), dg0_dr.detach()
        return g0, dg0_dr
    
    @property
    def n_layers(self) -> int: return self.inner_net.n_layers
    
    @property
    def n_neurons(self) -> int: return self.inner_net.n_neurons
    
    @property
    def seed(self) -> int: return self.inner_net.seed
    
    def state_dict(self, *args, **kwargs): return self.inner_net.state_dict(*args, **kwargs)
    def load_state_dict(self, *args, **kwargs): return self.inner_net.load_state_dict(*args, **kwargs)