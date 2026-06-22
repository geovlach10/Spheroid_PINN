import numpy as np
from scipy.integrate import solve_ivp
from core.context import PhysicsContext, physics_configuration

def get_phi(r: np.ndarray):
    return 0.44 * np.power(r, 3.2) + 0.56

def solve_pde(ctx, N=100):
    r = np.linspace(0, 200, N)
    dr = r[1] - r[0]
    pi = ctx.pi

    def fun(t, y):
        
        # Unpack Species (y is a single 3N-length vector)
        cf = y[0:N]
        cb = y[N:2*N]
        ci = y[2*N:3*N]

        # if np.any(np.isnan(y)) or np.any(abs(y) > 1e5):
        #     print(f'overflow detected at t={t}')
        #     breakpoint()
        # cf = np.clip(cf, 0.0, None)
        # cb = np.clip(cb, 0.0, 1)
        # ci = np.clip(ci, 0.0, None)

        # Pore concentration u=cf/phi
        phi = get_phi(r)
        u = cf / phi
        dcf_dt = np.zeros(N)
        
        # --- NODE j=0: CENTER (Symmetry + L'Hopital) ---
        # Analytical Limit: 3 * D * phi(0) * u_rr
        # Finite Difference u_rr with Symmetry: 2 * (u1 - u0) / dr^2
        u_0 = cf[0] / phi[0]
        u_1 = cf[1] / phi[1]

        diffusion_0 = pi['diff'] * 3 * phi[0] * 2 * (u_1 - u_0) / dr**2 # del Hospital rule, ghost node has been used!
        assosiation_0 = pi['on'] * cf[0]/phi[0] * (1 - cb[0])
        dissosiation_0 = pi['off'] * cb[0]
    
        dcf_dt[0] = (diffusion_0 - pi['s_ratio'] * (assosiation_0 - dissosiation_0)) / phi[0]    
       
        # --- INTERIOR NODES (j=1 to N-2): Spherical Flux ---
        for j in range(1, N-1):
            # use half-steps
            r_m = r[j] - dr/2
            r_p = r[j] + dr/2

            phi_m = get_phi(r_m)
            phi_p = get_phi(r_p)

            grad_u_m = (u[j] - u[j-1]) / dr
            grad_u_p = (u[j+1] - y[j]) / dr

            flux_m = r_m**2  * phi_m * grad_u_m
            flux_p = r_p**2  * phi_p * grad_u_p

            diffusion = pi['diff'] * (1.0 / (r[j]**2 + 1e-8)) * (flux_p - flux_m) / dr
            assosiation = pi['on'] * u[j] * (1 - cb[j])
            dissosiation = pi['off'] * cb[j]

            dcf_dt[j] = (diffusion - pi['s_ratio'] * (assosiation - dissosiation)) / phi[j]

        # double precision
        # # --- NODE j=N-1: SURFACE (Robin BC + Ghost Point) ---
        # # BC: phi * du/dr = pi['surface'] * (1 - u)
        # u_Nminus1 = cf[-1] / phi[-1]
        # u_Nminus2 = cf[-2] / phi[-2]
        
        # # u_N=cf_N/phi_N --> ghost node || phi[-1] = 1
        # u_N = (u_Nminus2 + 2 * dr * pi['surface'] * (1 - u_Nminus1)) / phi[-1]
        # u_rr = (u_N - 2*u_Nminus1 + u_Nminus2) / dr**2
        # u_r = (u_N - u_Nminus2) / (2 * dr) 

        # diffusion_N_minus1 = pi['diff'] * ((2.0 / r[-1]) * u_r + u_rr)
        # association_N_minus1 = pi['on'] * cf[-1]/phi[-1] * (1 - cb[-1])
        # dissosiation_N_minus1 = pi['off'] * cb[-1]

        # dcf_dt[-1] = (diffusion_N_minus1 - pi['s_ratio'] * (association_N_minus1 - dissosiation_N_minus1)) / phi[-1]
        u[-1] = cf[-1] / phi[-1]
        flux_in = pi['surface'] * (1 - u[-1])
        association_N_minus1 = pi['on'] * (cf[-1] / phi[-1]) * (1 - cb[-1])
        dissosiation_N_minus1 = pi['off'] * cb[-1]
        dcf_dt[-1] = (pi['diff'] * (flux_in / dr ) - pi['s_ratio'] * (association_N_minus1 - dissosiation_N_minus1)) / phi[-1] 

        dcb_dt = pi['on'] * cf/phi * (1 - cb) - pi['off'] * cb - pi['int'] * cb 
        dci_dt = pi['int'] * cb

        return np.concatenate([dcf_dt, dcb_dt, dci_dt])
    
    y0 = np.zeros(3 * N)
    solution = solve_ivp(fun=fun, t_span=[0, ctx.tau], y0=y0, method='Radau', rtol=1e-3, atol=1e-6)

    return r, solution

def test_pde_solver():
    cfg = physics_configuration
    ctx = PhysicsContext(cfg=cfg)
    pi = ctx.pi
    print(f'Starting solver with pi groups: {pi}')

    try:
        r, sol = solve_pde(ctx=ctx, N=20)
        print(f'success! sol.t final: {sol.t[-1]}')
    except Exception as e:
        print(f'caught error: {e}')

if __name__ == '__main__':
    test_pde_solver()