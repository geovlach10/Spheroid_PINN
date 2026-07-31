# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp
# from . import constants as _CST

# def get_mat(diag_size, sub, main, sup):
#     '''
#         Args:
#             diag_size: size of the main diagonal of the square matrix.
#             sub: scalar to populate the subdiagonal.
#             main: likewise...
#             sup: likewise... the superdiagonal.
#     '''
#     mat = sub * np.diag(np.ones(diag_size-1), k=-1) + main * np.diag(np.ones(diag_size), k=0) + sup * np.diag(np.ones(diag_size-1), k=1) 
#     return mat

# def compute_diff_matrices(m, a, b):
#     '''
#     THE SIZE OF THE MATRICES WILL BE M+1.

#     Args:
#         m: number of steps (steps = number of nodes - 1).
#         a: left space boundary.
#         b: right space boundary.

#     Returns:
#         x :array(shape(m+1,)), 
#         h(step size): float, 
#         Dx/2h: shape(m+1, m+1), 
#         Dxx/h**2 shape(m+1, m+1)
        
#     '''
#     h = (b - a) / m     # Step size
#     x = a + h * np.arange(m+1)
#     Dx = get_mat(diag_size=m+1, sub=-1, main=0, sup=1)
#     Dxx = get_mat(diag_size=m+1, sub=1, main=-2, sup=1)
#     return x, h, Dx / (2 * h), Dxx / h**2  

# def get_diffusion_differential_operator(m, x_domain, phase, debug=False):
#     '''
#     Diffucion_diff_operator = L_u * u = L_u * (φ^-1 * c) = (L_u * φ^-1) * c = L_c * c.

#     Args:
#         m (int): 
#             number of steps in the x domain(diag size = number of nodes = m + 1).
#         x_domain (list | tuple):
#             domain boundaries.
#         ctx (PhysicsContext):
#             Object which contains the parameters of the equation.
#         phase (str):
#             'uptake' or 'fishing' or 'clearance'.
#         debug (bool): 
#             if true it will print some info in the terminal.

#     Returns: 
#         tuple(x, L_c, b)  x: shape (m+1, ), L_c: shape (m+1, m+1), b: shape (m+1, ).
#         x (np.Array): 
#             spatial vector.
#         L_c (np.Array): 
#             differential operator of diffucion acting on the array of the concentration of antibody in the interstium. shape(steps+1, steps+1).
#         b (np.Array): 
#             Handles the external drug influx.
#     '''

#     x, h, Dx, Dxx = compute_diff_matrices(m, x_domain[0], x_domain[1])
#     assert x.shape == (m+1,)
#     assert Dx.shape == (m+1, m+1)
#     assert Dxx.shape == (m+1, m+1)

#     x_inverse = np.zeros_like(x)
#     x_inverse[1:] = 1. / x[1:]

#     phi = 0.44 * x**3.2 + 0.56
#     assert phi.shape == (m+1,)
#     phi_x = 0.44 * 3.2 * x**2.2
#     assert phi_x.shape == (m+1,)

#     # Change first, last row of differention matrices to force the bc into them. 
#     # Neuman bc at r0.
#     Dx[0, :] = 0 
#     Dxx[0, 0] =3 * -2 / h**2 # 3 comes from d'lohpital rule at r0
#     Dxx[0, 1] = 3 * 2 / h**2  

#     # Robin bc at rm.
#     alpha = 2 * h * _CST.P_STAR / _CST.D_STAR / phi[m]
    

#     Dx[m, m] = -alpha / (2 * h)
#     Dx[m, m-1] = 0
#     Dxx[m, m] = -(2 + alpha) / h**2
#     Dxx[m, m-1] = 2 / h**2
    

#     # create the differential operator acting on array u = Φ-1 * c.  
#     C2 = _CST.D_STAR * phi
#     C1 = 2 * _CST.D_STAR * phi * x_inverse + _CST.D_STAR * phi_x
#     L_u = np.diag(C2) @ Dxx + np.diag(C1) @ Dx

#     # create the contant term at the m boundary.
#     b = np.zeros_like(x)
#     b[m] = (C2[m] / h**2 + C1[m] / (2 * h)) * alpha * _CST.C_SOL_STAR # constant-term.

#     # create the differential operator acting on the concentration vector.
#     Phi_inv = np.diag(1.0 / phi)
#     L_c = L_u @ Phi_inv

#     if phase == 'fishing':
#         L_c[m, :] = 0.0
#         L_c[m, m] = -1.0 / h
#         # L_c[m, m] = -1.0 / tau
#         # L_c[m, m] = -1e4

#     if debug:
#         print(f'step size h: {h}')
#         print(f'x: {x.shape}, Dx: {Dx.shape}, Dxx: {Dxx.shape}')
#         print(f'x_inv: {x_inverse.shape}')
#         print(f'phi: {phi.shape}, phi_x: {phi_x.shape}')
#         print(f'alpha: {alpha}')
#         print(f'b: {b.shape}')
#         print(f'\nDx\n{Dx}\nDxx\n{Dxx}\nb\n{b}')
#         print(f'\nL_u: {L_u.shape}\n{L_u}')
#         print(f'\nL_cf: {L_c.shape}\n{L_c}')
#     return x, L_c, b

# def run_fdm(m, phase_name, y0, t_final=1.0):
#     N = m + 1
#     x, L_cf, b = get_diffusion_differential_operator(m=m, x_domain=[0, 1], phase=phase_name)

#     def uptake_rhs(t, y, N):
#         cf = y[:N]
#         cb = y[N:2*N]
#         ci = y[2*N:]

#         phi = 0.44 * x**3.2 + 0.56
        
#         diffusion = L_cf @ cf + b
#         reaction = _CST.K_ON_STAR * cf/phi * (_CST.R_T_STAR - cb) - _CST.K_OFF_STAR * cb
#         internalization = _CST.K_INT_STAR * cb
        
#         cf_t = diffusion - reaction
#         cb_t = reaction - internalization
#         ci_t = internalization
#         return np.concatenate([cf_t, cb_t, ci_t]) 

#     sol = solve_ivp(
#         fun=uptake_rhs,
#         t_span=[0, t_final],
#         y0=y0,
#         method='BDF',
#         args=(N,),
#         dense_output=True,
#         rtol = 1e-6,
#         atol=1e-9
#     )
#     return x, sol

# def plot_solution_at_phase(x, m, sol, time_splits, t_scale=24):
#     t = np.linspace(0, 1.0, time_splits)
#     plt.figure(figsize=(8, 5))
#     plt.plot(x, sol.sol(t)[:m+1, :])
#     plt.xlabel('r')
#     plt.ylabel('C')
#     plt.title(f"TRM - Rt: {_CST.R_T}")
#     plt.legend([f"t={(tj * t_scale):.2f}h" for tj in t])
#     plt.show()

# def run_phase(m, phase_name, y0, time_splits, t_final):
#     N = m + 1
#     x, sol = run_fdm(m=m, phase_name=phase_name, y0=y0, t_final=t_final)
#     print('shape of sol.y: ', sol.y.shape)
#     plot_solution_at_phase(m=m ,time_splits=time_splits, x=x, sol=sol)
#     final = sol.y[:, -1]
#     return x, sol, final

# if __name__ == '__main__':
#     m = 100     # number of steps.
#     N = m + 1   # number of spatial nodes.
#     y0_up = np.zeros(3 * N)     # initial condition for uptake phase.
#     x_up, sol_up, final_up = run_phase(m=m, phase_name='uptake', y0=y0_up, time_splits=4, t_final=1.0)
#     x_cl, sol_cl, _ = run_phase(m=m, phase_name='clearance', y0=final_up, time_splits=4, t_final=1.0)

"""Finite-difference reference solver (the "oracle") for the Trastuzumab
uptake/clearance model.

Method of lines: discretize the radial operator into a matrix, then hand the
resulting stiff ODE system to scipy's BDF integrator. Physics is imported
from spheroid_deepxde.constants, so this solver and the PINN consume identical
parameters by construction.

Scope: uptake (primary), clearance (dormant scaffolding). Fishing removed.
"""

import numpy as np
from scipy.integrate import solve_ivp

"""Single source of truth for Trastuzumab transport/reaction physics.

Both the PINN model and the FDM reference oracle import from here, so they
cannot disagree on physical parameters by construction (Single-Source-of-
Truth: there is only one copy of each number, so there is nothing to drift).

Conventions:
    - concentrations in nM, length in um, time in sec (internally)
    - normalized spatial domain r in [0, 1]; tau folded into the nondim groups
    - physical-nM nondimensionalization (concentrations kept in nM; only time
      and space normalized) -- this is the corrected convention.

Scope: uptake (primary) + clearance (dormant). Fishing is out of scope.
"""

# --- raw physical constants (paper-fitted, Trastuzumab) ---------------------
R = 200.0           # spheroid radius (um)
C_MAX = 60.0        # characteristic concentration (nM)

D_EFF = 8.38        # antibody diffusion coeff in interstitium (um^2/sec)
K_OFF = 4e-3        # dissociation rate constant (1/sec)
K_D = 6.76          # equilibrium dissociation constant, K_D = k_off/k_on (nM)
K_INT = 1.4e-5      # internalization rate constant (1/sec)
R_T = 1060.0        # total receptor concentration (nM)
K_ON = K_OFF / K_D  # association rate constant (1/(sec*nM))

# --- per-phase scales -------------------------------------------------------
TAU = {             # characteristic time of each phase (sec)
    'uptake': 24  * 3600,
    'clearance': 24 * 3600,
}
C_SOL = {           # bath antibody concentration (nM)
    'uptake': 60.0,
    'clearance': 0.0,
}
P = {               # surface mass-transfer coefficient (um/sec)
    'uptake': 2.5e-4 * 3600,
    'clearance': 2.6e-1 * 3600,
}


def phi(r):
    """Porosity profile phi(r) = 0.44 r^3.2 + 0.56.

    Works on numpy arrays (FDM) and torch tensors (PINN): both overload **
    and *, so one definition serves both backends.
    """
    return 0.44 * r ** 3.2 + 0.56


def dphi_dr(r):
    """Analytic derivative phi'(r) = 0.44 * 3.2 * r^2.2.

    Co-located with phi() so the value and its derivative cannot drift apart.
    The PINN takes derivatives via autograd and does not need this; the FDM
    operator does.
    """
    return 0.44 * 3.2 * r ** 2.2


def nondim_params(phase):
    """Dimensionless PDE coefficients on the normalized [0, 1] domain.

    Returns exactly the keys both the FDM and the PINN consume:
        D, K_on, K_off, K_int, P

    Phase-dependent because each group carries tau[phase].
    """
    t = TAU[phase]
    return {
        'D':     t / R ** 2 * D_EFF,
        'K_on':  t * C_MAX * K_ON,
        'K_off': t * K_OFF,
        'K_int': t * K_INT,
        'P':     t / R * P[phase],
    }

def get_mat(diag_size, sub, main, sup):
    """Tridiagonal matrix with constant sub / main / super diagonals."""
    return (
        sub * np.diag(np.ones(diag_size - 1), k=-1)
        + main * np.diag(np.ones(diag_size), k=0)
        + sup * np.diag(np.ones(diag_size - 1), k=1)
    )


def compute_diff_matrices(m, a, b):
    """Central-difference operators on [a, b] with m steps (size m+1).

    Returns x, h, Dx/(2h), Dxx/h^2.
    """
    h = (b - a) / m
    x = a + h * np.arange(m + 1)
    Dx = get_mat(m + 1, sub=-1, main=0, sup=1)
    Dxx = get_mat(m + 1, sub=1, main=-2, sup=1)
    return x, h, Dx / (2 * h), Dxx / h ** 2


def get_diffusion_differential_operator(m, x_domain, phase, debug=False):
    """Build the discretized diffusion operator L_c acting on concentration c.

    L_u acts on u = c/phi; we fold phi^-1 in so L_c = L_u @ diag(1/phi).
    Boundary rows are overwritten to embed:
        r=0 : symmetry (Neumann), with l'Hopital factor 3 on the 2nd deriv.
        r=R : Robin flux.

    Returns x, L_c, b  where b carries the external-influx constant term.
    """
    pi = nondim_params(phase)
    D_eff = pi['D']
    P = pi['P']

    x, h, Dx, Dxx = compute_diff_matrices(m, x_domain[0], x_domain[1])

    x_inv = np.zeros_like(x)
    x_inv[1:] = 1.0 / x[1:]

    phi_arr = phi(x)            # porosity values
    phi_prime = dphi_dr(x)      # dphi/dr

    # r=0 symmetry: kill the advection row, apply l'Hopital (factor 3).
    Dx[0, :] = 0
    Dxx[0, 0] = 3 * -2 / h ** 2
    Dxx[0, 1] = 3 * 2 / h ** 2

    # r=R Robin BC.
    alpha = 2 * h * P / D_eff / phi_arr[m]
    Dx[m, m] = -alpha / (2 * h)
    Dx[m, m - 1] = 0
    Dxx[m, m] = -(2 + alpha) / h ** 2
    Dxx[m, m - 1] = 2 / h ** 2

    # operator on u = c/phi
    C2 = D_eff * phi_arr
    C1 = 2 * D_eff * phi_arr * x_inv + D_eff * phi_prime
    L_u = np.diag(C2) @ Dxx + np.diag(C1) @ Dx

    # influx constant term at the surface node
    b = np.zeros_like(x)
    b[m] = (C2[m] / h ** 2 + C1[m] / (2 * h)) * alpha * C_SOL[phase]

    # operator on concentration c
    L_c = L_u @ np.diag(1.0 / phi_arr)

    if debug:
        print(f'h={h}  alpha={alpha}')
        print(f'L_c shape {L_c.shape}')
    return x, L_c, b


def _rhs(t, y, N, L_c, b, pi, phi_arr):
    """Method-of-lines RHS for the three coupled species.

    Constants (pi, L_c, b, phi_arr) are passed in via args rather than
    recomputed each call -- the integrator evaluates this thousands of times.
    """
    cf = y[:N]
    cb = y[N:2 * N]
    ci = y[2 * N:]

    diffusion = L_c @ cf + b
    reaction = pi['K_on'] * cf / phi_arr * (R_T / C_MAX - cb) - pi['K_off'] * cb
    internalization = pi['K_int'] * cb   # acts on BOUND (corrected)

    cf_t = diffusion - reaction
    cb_t = reaction - internalization
    ci_t = internalization
    return np.concatenate([cf_t, cb_t, ci_t])


def run_fdm(m, phase_name, y0, t_final=1.0):
    N = m + 1
    x, L_c, b = get_diffusion_differential_operator(m, [0, 1], phase_name)
    pi = nondim_params(phase_name)
    phi_arr = phi(x)

    sol = solve_ivp(
        fun=_rhs,
        t_span=[0, t_final],
        y0=y0,
        method='BDF',
        args=(N, L_c, b, pi, phi_arr),
        dense_output=True,
        rtol=1e-6,
        atol=1e-9,
    )
    print(f'[{phase_name}] solver: {sol.message}')
    return x, sol


def plot_solution_at_phase(x, sol, time_splits, phase_name):
    import matplotlib.pyplot as plt
    N = len(x)
    t = np.linspace(0, 1.0, time_splits)

    plt.figure(figsize=(8, 5))
    plt.plot(x, sol.sol(t)[:N, :])   
    plt.xlabel('Normalized distance from center')
    plt.ylabel('C(r, t)')
    plt.title(f'TRM - {phase_name}')
    plt.legend([f't={(tj * 24):.2f}h' for tj in t])
    plt.show()


def run_phase(m, phase_name, y0, time_splits, t_final):
    x, sol = run_fdm(m, phase_name, y0, t_final)
    print('sol.y shape:', sol.y.shape)
    print('pi:', nondim_params(phase_name))
    plot_solution_at_phase(x, sol, time_splits, phase_name)
    return x, sol, sol.y[:, -1]


# if __name__ == '__main__':
#     m = 100
#     N = m + 1
#     y0_up = np.zeros(3 * N)
#     x, sol_up, final_up = run_phase(m, 'uptake', y0_up, time_splits=4, t_final=1.0)
#     # clearance is dormant; uncomment once P_cl units are resolved:
#     # sol_cl, _ = run_phase(m, 'clearance', final_up, time_splits=4, t_final=1.0)
