import deepxde as dde
import torch
from spheroid_deepxde.utils.context import phi_liposomes


# Pde argument of data.TimePde.
def liposome_pde(x, y):
    experiment_duration = 16 # h. conduct until no spatial variation of liposomes is observed, which occurs around 16 hours in the experimental data.

    R = 200
    D = 1.46e-1 # μm2/sec.
    tau = experiment_duration * 3600
    D_star = D / R**2 * tau

    phi = phi_liposomes(x[:, 0:1])
    u = y / phi
    u_t = dde.grad.jacobian(u, x, i=0, j=1)
    u_r = dde.grad.jacobian(u, x, i=0, j=0)
    k =  D_star * x[:, 0:1]**2 * phi * u_r
    k_r = dde.grad.jacobian(k, x, i=0, j=0)
    lhs = phi * u_t * x[:, 0:1]**2
    rhs = k_r
    return lhs - rhs

# Robin Operator func argument (WHAT).
def robin_operator_liposome(x, y, context):
    experiment_duration = 16 # h. conduct until no spatial variation of liposomes is observed, which occurs around 16 hours in the experimental data.
    
    R = 200
    D = 1.46e-1 # μm2/sec.
    tau = experiment_duration * 3600 # sec.
    P = 1.91e-3 # μm/sec.
    C_sol = 0.5 # mM.
    D_star = D / R**2 * tau
    P_star = P * tau / R
    C_sol_star = C_sol / C_sol

    phi = phi_liposomes(x[:, 0:1])
    u = y / phi
    u_r = dde.grad.jacobian(u, x, i=0, j=0)
    lhs = D_star * phi * u_r
    rhs = P_star * (C_sol_star - u)
    return lhs - rhs

# Hard Constraints.
hard_constraint_center_boundary_transform_liposome = lambda x: torch.cat((x[:, 0:1]**2, x[:, 1:2]), dim=1)
hard_constraint_initial_transform_liposome = lambda x, y: x[:, 1:2] * y

