import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import constants as _CST

def get_mat(diag_size, sub, main, sup):
    '''
        Args:
            diag_size: size of the main diagonal of the square matrix.
            sub: scalar to populate the subdiagonal.
            main: likewise...
            sup: likewise... the superdiagonal.
    '''
    mat = sub * np.diag(np.ones(diag_size-1), k=-1) + main * np.diag(np.ones(diag_size), k=0) + sup * np.diag(np.ones(diag_size-1), k=1) 
    return mat

def compute_diff_matrices(m, a, b):
    '''
    THE SIZE OF THE MATRICES WILL BE M+1.

    Args:
        m: number of steps (steps = number of nodes - 1).
        a: left space boundary.
        b: right space boundary.

    Returns:
        x :array(shape(m+1,)), 
        h(step size): float, 
        Dx/2h: shape(m+1, m+1), 
        Dxx/h**2 shape(m+1, m+1)
        
    '''
    h = (b - a) / m     # Step size
    x = a + h * np.arange(m+1)
    Dx = get_mat(diag_size=m+1, sub=-1, main=0, sup=1)
    Dxx = get_mat(diag_size=m+1, sub=1, main=-2, sup=1)
    return x, h, Dx / (2 * h), Dxx / h**2  

def get_diffusion_differential_operator(m, x_domain, phase, debug=False):
    '''
    Diffucion_diff_operator = L_u * u = L_u * (φ^-1 * c) = (L_u * φ^-1) * c = L_c * c.

    Args:
        m (int): 
            number of steps in the x domain(diag size = number of nodes = m + 1).
        x_domain (list | tuple):
            domain boundaries.
        ctx (PhysicsContext):
            Object which contains the parameters of the equation.
        phase (str):
            'uptake' or 'fishing' or 'clearance'.
        debug (bool): 
            if true it will print some info in the terminal.

    Returns: 
        tuple(x, L_c, b)  x: shape (m+1, ), L_c: shape (m+1, m+1), b: shape (m+1, ).
        x (np.Array): 
            spatial vector.
        L_c (np.Array): 
            differential operator of diffucion acting on the array of the concentration of antibody in the interstium. shape(steps+1, steps+1).
        b (np.Array): 
            Handles the external drug influx.
    '''

    x, h, Dx, Dxx = compute_diff_matrices(m, x_domain[0], x_domain[1])
    assert x.shape == (m+1,)
    assert Dx.shape == (m+1, m+1)
    assert Dxx.shape == (m+1, m+1)

    x_inverse = np.zeros_like(x)
    x_inverse[1:] = 1. / x[1:]

    phi = 0.44 * x**3.2 + 0.56
    assert phi.shape == (m+1,)
    phi_x = 0.44 * 3.2 * x**2.2
    assert phi_x.shape == (m+1,)

    # Change first, last row of differention matrices to force the bc into them. 
    # Neuman bc at r0.
    Dx[0, :] = 0 
    Dxx[0, 0] =3 * -2 / h**2 # 3 comes from d'lohpital rule at r0
    Dxx[0, 1] = 3 * 2 / h**2  

    # Robin bc at rm.
    alpha = 2 * h * _CST.P_STAR / _CST.D_STAR / phi[m]
    

    Dx[m, m] = -alpha / (2 * h)
    Dx[m, m-1] = 0
    Dxx[m, m] = -(2 + alpha) / h**2
    Dxx[m, m-1] = 2 / h**2
    

    # create the differential operator acting on array u = Φ-1 * c.  
    C2 = _CST.D_STAR * phi
    C1 = 2 * _CST.D_STAR * phi * x_inverse + _CST.D_STAR * phi_x
    L_u = np.diag(C2) @ Dxx + np.diag(C1) @ Dx

    # create the contant term at the m boundary.
    b = np.zeros_like(x)
    b[m] = (C2[m] / h**2 + C1[m] / (2 * h)) * alpha * _CST.C_SOL_STAR # constant-term.

    # create the differential operator acting on the concentration vector.
    Phi_inv = np.diag(1.0 / phi)
    L_c = L_u @ Phi_inv

    if phase == 'fishing':
        L_c[m, :] = 0.0
        L_c[m, m] = -1.0 / h
        # L_c[m, m] = -1.0 / tau
        # L_c[m, m] = -1e4

    if debug:
        print(f'step size h: {h}')
        print(f'x: {x.shape}, Dx: {Dx.shape}, Dxx: {Dxx.shape}')
        print(f'x_inv: {x_inverse.shape}')
        print(f'phi: {phi.shape}, phi_x: {phi_x.shape}')
        print(f'alpha: {alpha}')
        print(f'b: {b.shape}')
        print(f'\nDx\n{Dx}\nDxx\n{Dxx}\nb\n{b}')
        print(f'\nL_u: {L_u.shape}\n{L_u}')
        print(f'\nL_cf: {L_c.shape}\n{L_c}')
    return x, L_c, b

def run_fdm(m, phase_name, y0, t_final=1.0):
    N = m + 1
    x, L_cf, b = get_diffusion_differential_operator(m=m, x_domain=[0, 1], phase=phase_name)

    def uptake_rhs(t, y, N):
        cf = y[:N]
        cb = y[N:2*N]
        ci = y[2*N:]

        phi = 0.44 * x**3.2 + 0.56
        
        diffusion = L_cf @ cf + b
        reaction = _CST.K_ON_STAR * cf/phi * (_CST.R_T_STAR - cb) - _CST.K_OFF_STAR * cb
        internalization = _CST.K_INT_STAR * cb
        
        cf_t = diffusion - reaction
        cb_t = reaction - internalization
        ci_t = internalization
        return np.concatenate([cf_t, cb_t, ci_t]) 

    sol = solve_ivp(
        fun=uptake_rhs,
        t_span=[0, t_final],
        y0=y0,
        method='BDF',
        args=(N,),
        dense_output=True,
        rtol = 1e-6,
        atol=1e-9
    )
    print(f'[{phase_name}]\nsolver message: {sol.message}')
    return x, sol

def plot_solution_at_phase(x, sol, time_splits, phase_name:str):
    t = np.linspace(0, 1.0, time_splits)
    tau = {'uptake': 24, 'fishing': 0.5/60 * 3600, 'clearance': 24}
    time_scale = 'sec' if phase_name == 'fishing' else 'h'
    plt.figure(figsize=(8, 5))
    plt.plot(x, sol.sol(t)[:101, :])
    plt.xlabel('Normalized distance from center')
    plt.ylabel('C(r, t)')
    plt.title(f"binding antibody - {phase_name}")
    plt.legend([f"t={(tj * tau[phase_name]):.2f}{time_scale}" for tj in t])
    plt.show()

def run_phase(m, phase_name, y0, time_splits, t_final):
    N = m + 1
    x, sol = run_fdm(m=m, phase_name=phase_name, y0=y0, t_final=t_final)
    print('shape of sol.y: ', sol.y.shape)
    plot_solution_at_phase(time_splits=time_splits, x=x, sol=sol, phase_name=phase_name)
    final = sol.y[:, -1]
    return sol, final

if __name__ == '__main__':
    m = 100     # number of steps.
    N = m + 1   # number of spatial nodes.
    y0_up = np.zeros(3 * N)     # initial condition for uptake phase.
    sol_up, final_up = run_phase(m=m, phase_name='uptake', y0=y0_up, time_splits=4, t_final=1.0)
    sol_cl, _ = run_phase(m=m, phase_name='clearance', y0=final_up, time_splits=4, t_final=1.0)