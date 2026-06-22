import torch
from core.model import PINN
from core.data import Dataset
from core.context import PhysicsContext

def get_pde_residuals(model: PINN, dataset: Dataset, ctx: PhysicsContext, phase):
    ''' Everything is normilized r=r_hat, t=t_hat, C=C_hat...'''
    # Slice the dataset and conenct the vectors to the graph
    r = dataset.data[:,0:1].requires_grad_(True)
    t = dataset.data[:, 1:2].requires_grad_(True)

    # Get physics from context
    # phi = ctx.get_phi(r) 
    # p = ctx.get_pde_parameters()
    # pi = ctx.get_dimentionaless_params(phase=phase)
    phi = ctx.get_phi(r) 
    pi = ctx.get_dimentionaless_params(phase=phase)
    Rt = 1060
    C_star = 1

    # take the prediction
    cf, cb, ci = model(r, t)
    cf_pore = cf / phi

    # calculate time derivatives
    cf_pore_t = torch.autograd.grad(cf_pore, t, torch.ones_like(cf_pore), create_graph=True)[0]
    cb_t = torch.autograd.grad(cb, t, torch.ones_like(cb), create_graph=True)[0]
    ci_t = torch.autograd.grad(ci, t, torch.ones_like(ci), create_graph=True)[0]

    # Calculate space derivatives
    cf_pore_r = torch.autograd.grad(cf_pore, r, grad_outputs=torch.ones_like(cf_pore), create_graph=True)[0]
    flux =  phi * r**2 * cf_pore_r
    flux_r = torch.autograd.grad(flux, r, torch.ones_like(flux), create_graph=True)[0]

    # Calculate pde terms
    # diffusion = pi['diff'] * flux_r
    # association1 = pi['on1'] * cf/phi * (1 - cb) 
    # association2 = pi['on2'] * cf/phi * (1 - cb)
    # dissociation1 = pi['off1'] * cb
    # dissociation2 = pi['off2'] * cb
    # internalization = pi['int'] * cb

    # return {
    #     'f': phi * cf_pore_t * (r**2) - diffusion + association1 * (r**2) - dissociation1 * (r**2),
    #     'b': cb_t - association2 + dissociation2 + internalization,
    #     'i':  ci_t - internalization
    # }

    diffusion = pi['D'] * flux_r
    reaction = pi['K_on'] * cf/phi * (Rt/C_star - cb) - pi['K_off'] * cb
    internalization = pi['K_int'] * ci
    

    return {
        'f': phi * cf_pore_t * (r**2) - diffusion + reaction * (r**2),
        'b': cb_t - reaction + internalization,
        'i':  ci_t - internalization
    }

def get_center_residual(model: PINN, dataset: Dataset):
    r = dataset.data[:,0:1].requires_grad_(True)
    t = dataset.data[:,1:2].requires_grad_(True)
    
    cf, _, _ = model(r, t)
    cf_r = torch.autograd.grad(cf, r, torch.ones_like(cf), create_graph=True)[0]

    return cf_r - 0 
  

def get_surface_residual(model: PINN, dataset: Dataset, ctx: PhysicsContext, phase="uptake"):
    r = dataset.data[:,0:1].requires_grad_(True)
    t = dataset.data[:,1:2].requires_grad_(True)
    
    pi = ctx.get_dimentionaless_params(phase=phase)
    phi = ctx.get_phi(r)
    C_sol = ctx.C_sol[phase]
    C_star = 1
    robin_coef = pi['P']/pi['D']

    cf, _, _ = model(r, t)
    cf_pore = cf / phi
    cf_pore_r = torch.autograd.grad(cf_pore, r, torch.ones_like(cf_pore), create_graph=True)[0]

    if phase == "uptake":
        return phi * cf_pore_r - robin_coef * (C_sol/C_star - cf/phi) # Robin.
    elif phase == 'fishing':
        return cf - 0 # Dirichlet.
    elif phase == "clearance":
        return phi * cf_pore_r - robin_coef * (C_sol/C_star - cf/phi) # Robin.

def get_initial_residual(model: PINN, dataset: Dataset, phase="uptake", ic_target_values=None, is_first_window=True):
    r = dataset.data[:,0:1].requires_grad_(True)
    t = dataset.data[:,1:2].requires_grad_(True)
    
    cf, cb, ci = model(r, t)

    # phi = ctx.get_phi(r)
    # cf_pore = cf / phi

    # cf_t = torch.autograd.grad(cf, t, torch.ones_like(cf), create_graph=True)[0]
    # cb_t = torch.autograd.grad(cb, t, torch.ones_like(cb), create_graph=True)[0]
    # ci_t = torch.autograd.grad(ci, t, torch.ones_like(ci), create_graph=True)[0]
    if phase == "uptake":
        if is_first_window:
            ic_f, ic_b, ic_i = 0, 0, 0
        else:
            ic_f, ic_b, ic_i = ic_target_values
    if phase == "fishing" or phase == "clearance": # Transfer learning.
        if ic_target_values is None:
            raise ValueError(f"⚠️ Phase is {phase}, but no IC target values were provided!")
        ic_f, ic_b, ic_i = ic_target_values
        
    return {
        'f': cf - ic_f,
        'b': cb - ic_b,
        'i': ci - ic_i
    }

def get_ic_l2_penalty(model: PINN, ic_dataset: Dataset, weight=0.01, phase='uptake'):
    if phase != 'uptake':
        return 0
    preds = model.forward(x=ic_dataset.r, t=ic_dataset.t)
    l2_penalty = sum(torch.mean(p**2) for p in preds)
    return l2_penalty

# def get_all_losses(model: PINN, pde: Dataset, center: Dataset, surface: Dataset, initial: Dataset, ctx: PhysicsContext):
#     w = ctx.get_pde_weights()
    
#     pde_res = get_pde_residuals(model, pde, ctx)
#     center_res = get_center_residual(model, center)
#     surface_res = get_surface_residual(model, surface, ctx)
#     ic_res = get_initial_residual(model, initial, ctx)

#     loss_pde_f = w['eq_f'] * torch.mean(pde_res['f']**2) 
#     loss_pde_b = w['eq_b'] * torch.mean(pde_res['b']**2) 
#     loss_pde_i = w['eq_i'] * torch.mean(pde_res['i']**2)
#     loss_center = w['center'] * torch.mean(center_res**2)
#     loss_surface = w['surface'] * torch.mean(surface_res**2)
#     loss_ic_f = w['ic_f'] * torch.mean(ic_res['f']**2)
#     loss_ic_b = w['ic_b'] * torch.mean(ic_res['b']**2)
#     loss_ic_i = w['ic_i'] * torch.mean(ic_res['i']**2)

#     loss = loss_pde_f + loss_pde_b + loss_pde_i + loss_center + loss_surface +loss_ic_f + loss_ic_b + loss_ic_i
#     return {
#         'total': loss,
#         'pde_f': loss_pde_f, 'pde_b': loss_pde_b, 'pde_i': loss_pde_i,
#         'center': loss_center, 'surface': loss_surface,
#         'ic_f': loss_ic_f,'ic_b': loss_ic_b,'ic_i': loss_ic_i
#     }