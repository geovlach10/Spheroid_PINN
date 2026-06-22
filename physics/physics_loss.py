import torch

from core.context import PhysicsContext
from core.solver import PINNSolver

from physics.residuals_ver2 import get_pde_residuals, get_center_residual, get_surface_residual, get_initial_residual

class PhysicsLoss():
    def __init__(self, context: PhysicsContext):
        self.phase = "uptake"
        self.ctx = context
        self.ic_target = None

    def get_total_loss(self, solver: PINNSolver):
        w = self.ctx.get_pde_weights()

        pde_res = get_pde_residuals(model=solver.approximator, dataset=solver.pde_training_dataset, ctx=self.ctx)
        center_res = get_center_residual(model=solver.approximator, dataset=solver.center_training_dataset)
        surface_res = get_surface_residual(model=solver.approximator, dataset=solver.surface_training_dataset, ctx=self.ctx, phase=self.phase)
        initial_res = get_initial_residual(model=solver.approximator, dataset=solver.initial_training_dataset, phase=self.phase, ic_target_values=solver.initial_condition)

        loss_pde_f = w['eq_f'] * torch.mean(pde_res['f']**2) 
        loss_pde_b = w['eq_b'] * torch.mean(pde_res['b']**2) 
        loss_pde_i = w['eq_i'] * torch.mean(pde_res['i']**2)
        loss_center = w['center'] * torch.mean(center_res**2)
        loss_surface = w['surface'] * torch.mean(surface_res**2)
        loss_ic_f = w['ic_f'] * torch.mean(initial_res['f']**2)
        loss_ic_b = w['ic_b'] * torch.mean(initial_res['b']**2)
        loss_ic_i = w['ic_i'] * torch.mean(initial_res['i']**2)

        loss = loss_pde_f + loss_pde_b + loss_pde_i + loss_center + loss_surface +loss_ic_f + loss_ic_b + loss_ic_i
        return {
            'total': loss,
            'pde_f': loss_pde_f, 'pde_b': loss_pde_b, 'pde_i': loss_pde_i,
            'center': loss_center, 'surface': loss_surface,
            'ic_f': loss_ic_f,'ic_b': loss_ic_b,'ic_i': loss_ic_i
        }