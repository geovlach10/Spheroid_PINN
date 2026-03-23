import torch
import torch.nn as nn
from core.context import PhysicsContext
from core.model import PINN
from core.data import PINNDataset
import pandas as pd

def pde_loss_fn(approximator: PINN, data: PINNDataset, ctx: PhysicsContext, pointwise=False, term_by_term_analysis=False):
    '''
    args:
      approximator: PINN instance
      data: torch.tensor of shape [num_points, 2] where the first column is the spatial variable r and the second column is the temporal variable t, both normalized to [0,1]
      pointwise: bool
          if True: returns a tuple of 3 column tensors
          if False: returns tuple of scalars
    returns:
      if pointwise: Tuple of 3 column tensors of size: (num_batches * 1)
      if not pointwise: Tuple of 3 scalars representing the mean squared residuals of the 3 equations
    '''
    if isinstance(data, PINNDataset):
        r_hat = data.r.requires_grad_(True)
        t_hat = data.r.requires_grad_(True)
    else:
        r_hat = data[:,0:1].requires_grad_(True)
        t_hat = data[:,1:2].requires_grad_(True)
    ####################################################################################################################################
    pde_weights = ctx.get_pde_weights()
    pde_parameters = ctx.get_pde_parameters()
    ####################################################################################################################################
    # Normalized concentrations (by C0=[c_sol] the initial concentration of the drug in the medium)
    cf_hat, cb_hat, ci_hat = approximator(r_hat, t_hat) 
    
    # spatial derivatives
    phi = 0.44 * (r_hat**3.2) + 0.56 # φ
    cf_tilde = cf_hat / phi # cf~ = cf^ / φ
    cf_tilde_r = torch.autograd.grad(cf_tilde, r_hat, torch.ones_like(cf_tilde), create_graph=True)[0]
    a = (r_hat**2) * phi * cf_tilde_r # r^2 * φ * d/dr(cf^/φ)
    diffucion_term = torch.autograd.grad(a, r_hat, torch.ones_like(a), create_graph=True)[0] # d/dr(r^2 * φ * d/dr(cf^/φ))
    
    # time derivatives
    cf_tilde_t = torch.autograd.grad(cf_tilde, t_hat, torch.ones_like(cf_tilde), create_graph=True)[0]
    cb_hat_t = torch.autograd.grad(cb_hat, t_hat, torch.ones_like(cb_hat), create_graph=True)[0]
    ci_hat_t = torch.autograd.grad(ci_hat, t_hat, torch.ones_like(ci_hat), create_graph=True)[0]
    ####################################################################################################################################
    # 1st equation
    norm_coef_11 = ctx.R**2 * ctx.C0 * (1.0 / 3600 /ctx.tau) # [=] μm^2 * nM /sec
    norm_coef_21 = pde_parameters['d'] * ctx.C0 # [=] μm^2/sec * nM 
    norm_coef_31 = ctx.R**2 * ctx.C0 * pde_parameters['k_off']/pde_parameters['k_d'] * pde_parameters['r_t'] # [=] μm^2 * nM * 1/nM/sec * nM
    norm_coef_41 = ctx.R**2 * ctx.C0 * pde_parameters['k_off'] * (1 + ctx.C0/pde_parameters['k_d']) #[=] μm^2 * nM * 1/sec * nM/nM

    term11 = norm_coef_11 * (phi * cf_tilde_t * r_hat**2) #R^2 * C0 / (3600 * tau) * φ * d/dt(cf^/φ) * r^2 
    term21 = norm_coef_21 * diffucion_term # D * C0 * d/dr(r^2 * φ * d/dr(cf^/φ))
    term31 = norm_coef_31 * (cf_tilde * r_hat**2) # R^2 * C0 * koff/kd * Rt * ρ^2 * cf^/φ
    term41 = norm_coef_41 * (cb_hat * r_hat**2) # R^2 * C0 * koff * (1 + C0/Kd) * cb^ * r^2

    residual_eq1 = term11 - term21 + term31 - term41
    loss_eq1 = pde_weights['eq1'] * torch.mean(residual_eq1**2)
    ####################################################################################################################################
    #2nd equation 
    norm_coef_21 = 1.0 / (3600 * ctx.tau) # 1/sec
    norm_coef_22 = pde_parameters['k_off']/pde_parameters['k_d'] * pde_parameters['r_t'] # 1/sec/nM * nM
    norm_coef_23 = pde_parameters['k_off']/pde_parameters['k_d']  * ctx.C0 # 1/sec/nM * nM
    norm_coef_24 = pde_parameters['k_off'] - pde_parameters['k_int'] # 1/sec

    term21 = norm_coef_21 * cb_hat_t # 1/(3600τ) * d/dt(cb^)
    term22 = norm_coef_22 * cf_tilde # koff/kd * Rt * cf^/φ 
    term23 = norm_coef_23 * cf_tilde * cb_hat # koff/kd * C0 * cf^/φ * cb^ 
    term24 = norm_coef_24 * cb_hat # (koff - kint) * cb^

    residual_eq2 = term21 - term22 + term23 - term24
    loss_eq2 = pde_weights['eq2'] * torch.mean(residual_eq2**2)
    ####################################################################################################################################
    # 3rd equation
    norm_coef_31 = 1.0 / 3600 / ctx.tau # 1/sec
    norm_coef_32 = pde_parameters['k_int']

    term31 = norm_coef_31 * ci_hat_t # (1 / 3600 / τ) * ci^ 
    term32 = norm_coef_32 * cb_hat  # kint * cb^

    residual_eq3 = term31 - term32
    loss_eq3 = pde_weights['eq3'] * torch.mean(residual_eq3**2)
    ####################################################################################################################################

    if term_by_term_analysis:
        return pd.DataFrame({
                'r^': r_hat.detach().cpu().numpy().flatten(),
                't^': t_hat.detach().cpu().numpy().flatten(),
                'φ': phi.detach().cpu().numpy().flatten(),
                'cf^': cf_hat.detach().cpu().numpy().flatten(),
                'cf~' : cf_tilde.detach().cpu().numpy().flatten(),
                'd/dr(cf~)': cf_tilde_r.detach().cpu().numpy().flatten(),
                'cb^': cb_hat.detach().cpu().numpy().flatten(),
                'ci^': ci_hat.detach().cpu().numpy().flatten(),
                'R^2 * C0 / (3600 * tau) * φ * d/dt(cf^/φ) * r^2 ': term11.detach().cpu().numpy().flatten(),
                'D * C0 * d/dr(r^2 * φ * d/dr(cf^/φ))': term21.detach().cpu().numpy().flatten(),
                'R^2 * C0 * koff/kd * Rt * r^2 * cf^/φ': term31.detach().cpu().numpy().flatten(),
                'R^2 * C0 * koff * (1 + C0/Kd) * cb^ * r^2': term41.detach().cpu().numpy().flatten(),
                'residual_eq1': residual_eq1.detach().cpu().numpy().flatten(),
                '1/(3600τ) * d/dt(cb^)': term21.detach().cpu().numpy().flatten(),
                'koff/kd * Rt * cf^/φ': term22.detach().cpu().numpy().flatten(),
                'koff/kd * C0 * cf^/φ * cb^': term23.detach().cpu().numpy().flatten(),
                '(koff - kint) * cb^': term24.detach().cpu().numpy().flatten(),
                'residual_eq2': residual_eq2.detach().cpu().numpy().flatten(),
                '(1 / 3600 / τ) * ci^': term31.detach().cpu().numpy().flatten(),
                'kint * cb^': term32.detach().cpu().numpy().flatten(),
                'residual_eq3': residual_eq3.detach().cpu().numpy().flatten()
                }
                )
    if pointwise:
        return residual_eq1, residual_eq2, residual_eq3  # Vectors
    return loss_eq1, loss_eq2, loss_eq3 # Scalar

def bc_loss_fn(approximator: PINN, ctx: PhysicsContext, data_center: PINNDataset, data_surface: PINNDataset, uptake=True):
    weight_center = ctx.pde_weights['center']
    weight_surface = ctx.pde_weights['surface']
    ####################################################################################################################################
    # @ r=0 (center)
    data_center = data_center.get_nondimentional_data()
    r_center_hat = data_center[:,0:1].requires_grad_(True)
    t_center_hat = data_center[:,1:2].requires_grad_(True)

    cf_center_hat, _, _ = approximator(r_center_hat, t_center_hat)
    cf_center_hat_r = torch.autograd.grad(cf_center_hat, r_center_hat, torch.ones_like(cf_center_hat), create_graph=True)[0]

    residual_center = cf_center_hat_r - 0
    loss_center = weight_center * torch.mean(residual_center**2)
    ####################################################################################################################################
    # @ r=R (surface): 
    data_surface = data_surface.get_nondimentional_data()
    r_surface_hat = data_surface[:,0:1].requires_grad_(True)
    t_surface_hat = data_surface[:,1:2].requires_grad_(True)

    cf_surface_hat, _, _ = approximator(r_surface_hat, t_surface_hat)

    phi = 0.44 * (r_surface_hat**3.2) + 0.56 # φ
    cf_surface_tilde = cf_surface_hat / phi # cf^/φ
    cf_surface_tilde_r = torch.autograd.grad(cf_surface_tilde, r_surface_hat, torch.ones_like(cf_surface_tilde), create_graph=True)[0] # d/dr(cf^/φ)

    norm_coef = ctx.pde_parameters['d'] / ctx.R

    term1 = norm_coef * cf_surface_tilde_r # D/R * d/dr(cf^/φ)
    term2 = ctx.pde_parameters['p_up'] * (1 - cf_surface_tilde) if uptake==True else ctx.pde_parameters['p_cl'] * (1 - cf_surface_tilde) # Pup * (1 - cf^/φ)

    residual_surface = term1 - term2
    loss_surface = weight_surface * torch.mean(residual_surface**2)
    return loss_center, loss_surface 

def ic_loss_fn(approximator, data: PINNDataset, ctx: PhysicsContext):
    weight = ctx.pde_weights['ic']

    data = data.get_nondimentional_data()
    r_hat = data[:,0:1].requires_grad_(True)
    t_hat = data[:,1:2].requires_grad_(True)

    cf_hat, cb_hat, ci_hat = approximator(r_hat, t_hat)
    residual1 = cf_hat - 0 
    residual2 = cb_hat - 0
    residual3 = ci_hat - 0
    loss = weight * (torch.mean(residual1**2) + torch.mean(residual2**2) + torch.mean(residual3**2))
    return loss

def sensor_loss_fn(approximator, sensor_set, sensor_output):
    weight = approximator.pde_weights['sensor_dynweight']

    r = sensor_set[:, 0].view(-1, 1).requires_grad_(True)
    t = sensor_set[:, 1].view(-1, 1).requires_grad_(True)
    c_true = sensor_output.view(-1, 1)

    c = approximator(r, t)
    residual = abs(c - c_true) / c_true
    return weight * residual.pow(2).mean()