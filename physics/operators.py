import torch

def compute_grad(u, x):
    ''' 
    computes the partial derivative of u w.r.t x (u_x) and retains the computational graph for higher order derivatives
    args:
        u: column tensor of shape [num_points * 1]
        x: column tensor of shape [num_points * 1]
    returns:
        u_x: column tensor of shape [num_points * 1]
    '''
    return torch.autograd.grad(
        u, x, 
        grad_outputs=torch.ones_like(u), 
        create_graph=True, 
        retain_graph=True
    )[0]

def spherical_laplacian(c, r, eps=1e-8):
    '''
    calculates (1/r^2) * d/dr(r^2 * dc/dr)
    the eps is added to the denominator to avoid division by zero at r=0
    args:
        c: column tensor(output of the PINN)
        r: column tensor(.r attribute of the dataset)
    returns:
        the spherical laplacian of the output of the PINN: column tensor of shape [num_points * 1]
    '''
    c_r = compute_grad(c, r)
    flux = r**2 * c_r
    flux_r = compute_grad(flux, r)
    return (1.0 / (r**2 + eps)) * flux_r


