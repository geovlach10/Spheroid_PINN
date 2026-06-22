import torch
from core.data import Dataset
from core.context import PhysicsContext
from core.model import PINN

class Physics():
    def __init__(self, context: PhysicsContext):
        self.ctx = context

    def compute_interstitium_residual(self, norm_dataset: Dataset, model: PINN):
        w = self.ctx.pde_weights
        p = self.ctx.pde_parameters
        phi = self.ctx.get_phi(r = norm_dataset.r)

        r_norm, t_norm = norm_dataset.r.requires_grad_(True), norm_dataset.t.requires_grad_(True)
        pred = model.forward(r_norm, t_norm)

        cf = pred[:, 0:1] * self.ctx.C0
        cb = pred[:, 1:2] * p['r_t']
        # ------------------------------------------------------------------------------------------#
        r = r_norm * self.ctx.R                     # μm
        t = t_norm * self.ctx.tau['uptake'] * 3600   # sec

        Rf = p['r_t'] - cb
        cf_phi = cf / phi

        cf_phi_t = torch.autograd.grad(cf_phi, t, grad_outputs=torch.ones_like(cf_phi), create_graph=True)[0]

        cf_phi_r = torch.autograd.grad(cf_phi, r, grad_outputs=torch.ones_like(cf_phi), create_graph=True)[0]
        flux = p['d'] * r**2 * phi * cf_phi_r
        flux_r = torch.autograd.grad(flux, r, grad_outputs=torch.ones_like(flux), create_graph=True)[0]

        residual_interstitium = phi * cf_phi_t * r**2 - flux_r + p['k_on'] * cf/phi * Rf * r**2 - p['k_off'] * cb * r**2 
        return residual_interstitium