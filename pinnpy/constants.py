# --- Transtuzumab parameters ---
## characteristic scales.
R_MAX = 200.0               # spheroid radius (μm)
C_MAX = 60.0                # characteristic concentration (nM)
T_MAX = 24.0 * 3600         # characteristic time of each phase (sec)

## pde params
D_EFF = 8.38                # antibody diffusion coeff in interstitium (μm^2/sec)
K_OFF = 4e-3                # dissociation rate constant (1/sec)
K_D = 6.76                  # equilibrium dissociation constant, K_D = k_off/k_on (nM)
K_INT = 1.44e-5             # internalization rate constant (1/sec)
K_ON = K_OFF / K_D          # association rate constant (1/(sec*nM))

C_SOL = 60.0                # bath antibody concentration (nM)
R_T = 1060                  # total receptor concentration (nM)
P = 2.5e-4                  # surface mass-transfer coefficient (μm/sec)

# --- non_dim scheme ---
D_STAR = T_MAX / R_MAX**2 * D_EFF
K_ON_STAR = T_MAX * C_MAX * K_ON
K_OFF_STAR = T_MAX * K_OFF
K_INT_STAR = T_MAX * K_INT

R_T_STAR = R_T / C_MAX
P_STAR = T_MAX / R_MAX * P
C_SOL_STAR = C_SOL / C_MAX

# // Residual normalization
SCALE0 = C_SOL_STAR * (D_STAR + K_ON_STAR * R_T_STAR)
SCALE1 = lambda L: K_ON_STAR * C_SOL_STAR * (L * R_T_STAR) + K_OFF_STAR * R_T_STAR
SCALE2 = K_INT_STAR * R_T_STAR 

