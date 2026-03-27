import torch
from core.model import PINN
from core.data import PINNDataset
from core.context import PhysicsContext

def get_pde_residuals(model: PINN, dataset: PINNDataset, ctx: PhysicsContext):
    ''' Everything is normilized r=r_hat, t=t_hat, C=C_hat...'''
    # Slice the dataset and conenct the vectors to the graph
    r = dataset.data[:,0:1].requires_grad_(True)
    t = dataset.data[:, 1:2].requires_grad_(True)

    # Get physics from context
    phi = ctx.get_phi(r) 
    pi = ctx.pi

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
    diffusion = (1.0 / r**2) * flux_r
    association = pi['on'] * cf/phi * (1 - cb)
    dissociation = pi['off'] * cb
    internalization = pi['int'] * cb

    return {
        'f': phi * cf_pore_t - pi['diff'] * diffusion + pi['s_ratio'] * (association - dissociation),
        'b': cb_t - association + dissociation + internalization,
        'i':  ci_t - internalization
    }

def get_center_residual(model: PINN, dataset: PINNDataset):
    r = dataset.data[:,0:1].requires_grad_(True)
    t = dataset.data[:,1:2].requires_grad_(True)
    
    cf, _, _ = model(r, t)
    cf_r = torch.autograd.grad(cf, r, torch.ones_like(cf), create_graph=True)[0]

    res = cf_r - 0 
    return res 

def get_surface_residual(model: PINN, dataset: PINNDataset, ctx: PhysicsContext):
    r = dataset.data[:,0:1].requires_grad_(True)
    t = dataset.data[:,1:2].requires_grad_(True)
    
    pi = ctx.pi
    params = ctx.pde_parameters
    phi = ctx.get_phi(r)

    cf, _, _ = model(r, t)
    cf_pore = cf / phi

    cf_pore_r = torch.autograd.grad(cf_pore, r, torch.ones_like(cf_pore), create_graph=True)[0]
    
    res = phi * cf_pore_r - pi['surface'] * (1 - cf/phi)
    return res

def get_ic_residual(model: PINN, dataset: PINNDataset, ctx: PhysicsContext):
    r = dataset.data[:,0:1].requires_grad_(True)
    t = dataset.data[:,1:2].requires_grad_(True)
    
    phi = ctx.get_phi(r)
    cf_pore = cf / phi

    cf, cb, ci = model(r, t)
    
    cf_pore_t = torch.autograd.grad(cf_pore, t, torch.ones_like(cf_pore), create_graph=True)[0]
    cb_t = torch.autograd.grad(cb, t, torch.ones_like(cb), create_graph=True)[0]
    ci_t = torch.autograd.grad(ci, t, torch.ones_like(ci), create_graph=True)[0]

    return {
        'f': cf_pore_t,
        'b': cb_t,
        'i': ci_t
    }

def get_losses(model: PINN, pde: PINNDataset, center: PINNDataset, surface: PINNDataset, initial: PINNDataset, ctx: PhysicsContext):
    w = ctx.get_pde_weights()
    
    pde_res = get_pde_residuals(model, pde, ctx)
    center_res = get_center_residual(model, center)
    surface_res = get_surface_residual(model, surface, ctx)
    ic_res = get_ic_residual(model, initial, ctx)

    loss_pde_f = w['eq_f'] * torch.mean(pde_res['f']**2) 
    loss_pde_b = w['eq_b'] * torch.mean(pde_res['b']**2) 
    loss_pde_i = w['eq_i'] * torch.mean(pde_res['i']**2)
    loss_center = w['center'] * torch.mean(center_res**2)
    loss_surface = w['center'] * torch.mean(surface_res**2)
    loss_ic_f = w['ic_f'] * torch.mean(ic_res['f']**2)
    loss_ic_b = w['ic_b'] * torch.mean(ic_res['b']**2)
    loss_ic_i = w['ic_i'] * torch.mean(ic_res['i']**2)

    loss = loss_pde_f + loss_pde_b + loss_pde_i + loss_center + loss_surface +loss_ic_f + loss_ic_b + loss_ic_i
    return {
        'pde_f': loss_pde_f, 'pde_b': loss_pde_b, 'pde_i': loss_pde_i,
        'center': loss_center, 'surface': loss_surface,
        'ic_f': loss_ic_f,'ic_b': loss_ic_b,'ic_i': loss_ic_i,
        'total': loss
    }