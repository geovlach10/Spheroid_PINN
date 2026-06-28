# --- Transtuzumab parameters ---
## characteristic scales.
R_MAX = 200.0               # spheroid radius (μm)
C_MAX = 60.0                # characteristic concentration (nM)
T_MAX = 24.0                # characteristic time of each phase (h)

## pde params
D_EFF = 8.38 * 3600         # antibody diffusion coeff in interstitium (μm^2/h)
K_OFF = 4e-3  * 3600        # dissociation rate constant (1/h)
K_D = 6.76                  # equilibrium dissociation constant, K_D = k_off/k_on (nM)
K_INT = 1.4e-5  * 3600      # internalization rate constant (1/h)
K_ON = K_OFF / K_D          # association rate constant (1/(h*nM))

C_SOL = 60.0                # bath antibody concentration (nM)
R_T = 1060                  # total receptor concentration (nM)
P = 2.5e-4 * 3600          # surface mass-transfer coefficient (μm/h)

# --- non_dim scheme ---
D_STAR = T_MAX / R_MAX**2 * D_EFF
K_ON_STAR = T_MAX * C_MAX * K_ON
K_OFF_STAR = T_MAX * K_OFF
K_INT_STAR = T_MAX * K_INT

R_T_STAR = R_T / C_MAX
P_STAR = T_MAX / R_MAX * P
C_SOL_STAR = C_SOL / C_MAX

