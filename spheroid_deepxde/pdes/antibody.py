import deepxde as dde
import torch
import torch.nn.functional as F
from spheroid_deepxde.utils.context import phi_antibody

# Pde argument of data.TimePde.
def binding_antibody_system_of_pdes(x, y):
    experiment_duration = 24 # h.

    tau = experiment_duration * 3600
    R = 200
    D = 8.38
    Cmax = 60 # nM
    Kd = 6.76
    Koff = 4e-3
    Kint = 1.4e-5
    Kon = Koff / Kd
    Rt = 1060

    D_star = D / R**2 * tau
    Kon_star = Kon * tau * Cmax
    Rt_star = Rt / Cmax
    Koff_star = Koff * tau
    Kint_star = Kint * tau

    r = x[:, 0:1]
    r2 = r ** 2

    y0 = y[:, 0:1]
    y1 = y[:, 1:2]
    y2 = y[:, 2:3]

    phi = phi_antibody(r)
    u0 = y0 / phi
    
    # time derivatives.
    u0_t = dde.grad.jacobian(u0, x, i=0, j=1)
    y1_t = dde.grad.jacobian(y1, x, i=0, j=1)
    y2_t = dde.grad.jacobian(y2, x, i=0, j=1)

    # spatial derivatives.
    u0_r = dde.grad.jacobian(u0, x, i=0, j=0)
    k = r**2 * phi * u0_r
    k_r = dde.grad.jacobian(k, x, i=0, j=0)

    diffusion = D_star * k_r
    association = Kon_star * u0 * (Rt_star - y1)
    dissociation = Koff_star * y1
    internalization = Kint_star * y1
    reaction = association - dissociation

    # pde0.
    lhs0 = phi * u0_t 
    rhs0 = 1 / (r2 + 1e-8) * diffusion - reaction 

    # pde1.
    lhs1 = y1_t
    rhs1 = reaction - internalization

    # pde2.
    lhs2 = y2_t
    rhs2 = internalization

    scale = Kon_star * Rt_star
    # scale = 1.0
    res0 = lhs0 - rhs0
    res1 = lhs1 - rhs1
    res2 = lhs2 - rhs2

    return [res0 / scale, res1 / scale, res2 / scale]


# Robin Operator func argument (WHAT).
def robin_surface_binding_antibody(x, y, context):
    experiment_duration = 24 # h. conduct until no spatial variation of liposomes is observed, which occurs around 16 hours in the experimental data.
    
    R = 200
    D = 8.38 # μm2/sec.
    tau = experiment_duration * 3600 # sec.
    P = 2.5e-4 # μm/sec.
    C_sol = 60 # mM.
    D_star = D / R**2 * tau
    P_star = P * tau / R
    C_sol_star = C_sol / C_sol

    r = x[:, 0:1]
    y0 = y[:, 0:1]
    phi = phi_antibody(r)

    u0 = y0 / phi
    u_r = dde.grad.jacobian(u0, x, i=0, j=0)

    # robin.
    lhs = D_star * phi * u_r
    rhs = P_star * (C_sol_star - u0)
    return lhs - rhs

# Hard boundary constraints.
hard_constraint_center_boundary_transform_antibody = lambda x: torch.cat((x[:, 0:1]**2, x[:, 1:2]), dim=1)
hard_constraint_initial_transform_antibody = lambda x, y: x[:, 1:2].expand_as(y) * F.softplus(y)


############################################################################################################################################################
### fishing
def binding_antibody_system_of_pdes_fishing(x, y):
    experiment_duration = 0.5 / 60 # h.

    tau = experiment_duration * 3600
    R = 200
    D = 8.38
    Cmax = 60 # nM
    Kd = 6.76
    Koff = 4e-3
    Kint = 1.4e-5
    Kon = Koff / Kd
    Rt = 1060

    D_star = D / R**2 * tau
    Kon_star = Kon * tau * Cmax
    Rt_star = Rt / Cmax
    Koff_star = Koff * tau
    Kint_star = Kint * tau

    r = x[:, 0:1]
    r2 = r ** 2

    y0 = y[:, 0:1]
    y1 = y[:, 1:2]
    y2 = y[:, 2:3]

    phi = phi_antibody(r)
    u0 = y0 / phi
    
    # time derivatives.
    u0_t = dde.grad.jacobian(u0, x, i=0, j=1)
    y1_t = dde.grad.jacobian(y1, x, i=0, j=1)
    y2_t = dde.grad.jacobian(y2, x, i=0, j=1)

    # spatial derivatives.
    u0_r = dde.grad.jacobian(u0, x, i=0, j=0)
    k = r**2 * phi * u0_r
    k_r = dde.grad.jacobian(k, x, i=0, j=0)

    diffusion = D_star * k_r
    association = Kon_star * u0 * (Rt_star - y1)
    dissociation = Koff_star * y1
    internalization = Kint_star * y1
    reaction = association - dissociation

    # pde0.
    lhs0 = phi * u0_t 
    rhs0 = 1 / (r2 + 1e-8) * diffusion - reaction 

    # pde1.
    lhs1 = y1_t
    rhs1 = reaction - internalization

    # pde2.
    lhs2 = y2_t
    rhs2 = internalization

    scale = Kon_star * Rt_star
    res0 = lhs0 - rhs0
    res1 = lhs1 - rhs1
    res2 = lhs2 - rhs2

    return [res0 / scale, res1 / scale, res2 / scale]

def hard_constraint_fishing_surface(x, y):
    r = x[:, 0:1]
    y0_constrained = (1.0 - r) * F.softplus(y[:, 0:1])
    y1_constrained = F.softplus(y[:, 1:2])
    y2_constrained = F.softplus(y[:, 2:3])
    return torch.cat([y0_constrained, y1_constrained, y2_constrained], dim=1)