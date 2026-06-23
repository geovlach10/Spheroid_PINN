import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def solve_fdm(context, phase='uptake'):
    # Grid setup
    N = 100  # Number of spatial points
    dr = 1.0 / (N - 1)
    r = np.linspace(0, 1, N)
    
    # Physical parameters from your context
    D = context.D_star
    kon = context.kon_star
    koff = context.koff_star
    P = context.P_up if phase == 'uptake' else context.P_cl
    C_ext = 1.0 if phase == 'uptake' else 0.0

    def rhs(t, y):
        # y contains [cf_0, ..., cf_N, cb_0, ..., cb_N]
        cf = y[:N]
        cb = y[N:]
        dcf_dt = np.zeros(N)
        
        # 1. Surface Robin BC for cf_N
        # D * (cf_N+1 - cf_N-1)/(2*dr) = P * (C_ext - cf_N)
        # Solve for virtual point cf_N+1 if needed, or use a ghost point
        
        # 2. Diffusion Logic
        for i in range(1, N-1):
            diffusion = D * ((cf[i+1] - 2*cf[i] + cf[i-1])/dr**2 + (2/r[i]) * (cf[i+1] - cf[i-1])/(2*dr))
            reaction = kon * cf[i] - koff * cb[i] # Simplified reaction
            dcf_dt[i] = diffusion - reaction
            
        # 3. Symmetry at r=0
        dcf_dt[0] = 3 * D * (2*(cf[1] - cf[0])/dr**2) - (kon * cf[0] - koff * cb[0])
        
        # Boundary r=1 (Robin)
        # Implement discretized Robin here...
        
        dcb_dt = kon * cf - koff * cb # Bound species (no diffusion)
        return np.concatenate([dcf_dt, dcb_dt])

    # Initial Condition
    y0 = np.zeros(2 * N) # Spheroid starts empty
    t_span = (0, 1) # Normalized time
    sol = solve_ivp(rhs, t_span, y0, method='BDF', t_eval=np.linspace(0, 1, 50))
    return r, sol
