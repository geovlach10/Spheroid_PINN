import deepxde as dde
import torch
from spheroid_deepxde.utils.context import phi_antibody

# Pde argument of data.TimePde.
def binding_antibody_system_of_pdes(x, y):
    experiment_duration = 24 # h.

    tau = experiment_duration * 3600
    R = 200
    D = 8.38
    Csol = 60 # nM
    Kd = 6.76
    Koff = 4e-3
    Kint = 1.4e-5
    Kon = Koff / Kd
    Rt = 1060

    D_star = D / R**2 * tau
    Kon_star = Kon * tau * Csol
    Rt_star = Rt / Csol
    Koff_star = Koff * tau
    Kint_star = Kint * tau

    r = x[:, 0:1]

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
    lhs0 = phi * u0_t * r**2
    rhs0 = diffusion - reaction * r**2

    # pde1.
    lhs1 = y1_t
    rhs1 = reaction - internalization

    # pde2.
    lhs2 = y2_t
    rhs2 = internalization

    return [lhs0 - rhs0, (lhs1 - rhs1) / (Kon_star * Rt_star), (lhs2 - rhs2) / (Kon_star * Rt_star)]


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
hard_constraint_center_boundary_transform_binding = lambda x: torch.cat((x[:, 0:1]**2, x[:, 1:2]), dim=1)
hard_constraint_initial_transform_binding = lambda x, y: x[:, 1:2].expand_as(y) * torch.exp(y)