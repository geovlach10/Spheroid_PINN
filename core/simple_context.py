def compute_non_dimentional_params(phase):
    C = 1.0         #nM
    R = 200         #μm
    T = {'uptake': 24 * 3600, 'fishing': 0.5 / 60 * 3600, 'clearance': 24 * 3600}     # sec 

    C_sol = {'uptake': 60, 'fishing': 0, 'clearance': 0}           # nM
    P = {'uptake': 2.5e-4, 'fishing': 0, 'clearance': 2.6e-7}      # μm / sec
    D_eff = 8.38        # μm^2 / sec
    K_d = 6.76          # nM
    K_off = 4e-3        # 1 / sec
    K_int = 1.4e-5      # 1/ sec
    K_on = K_off / K_d  # 1 / sec / nM
    return {
        'D': T[phase] / R**2 * D_eff, 
        'K_on': T[phase] * C * K_on,
        'K_off': T[phase] * K_off,
        'K_int': T[phase] * K_int,
        'P': T[phase] / R * P[phase]
    }