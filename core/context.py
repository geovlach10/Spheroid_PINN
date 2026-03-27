import torch
import torch.nn as nn

physics_configuration = {
    # physical scale constants
    'R': 200,
    'tau': 50,
    'C0': 60,
    # pde parameters
    'd': 8.38,
    'k_off': 4e-3,
    'k_d': 6.76,
    'k_int': 1.4e-5,
    'r_t': 1060.0,
    'p_up': 2.5e-4,
    'p_cl': 2.6e-1,
    # pde weights
    'eq1': 1.0,
    'eq2': 1.0,
    'eq3': 1.0,
    'center': 1.0,
    'surface': 1.0,
    'ic': 1.0
}  

class PhysicsContext(nn.Module):
    def __init__(self, cfg, device='cpu'):
        super().__init__()
        self.device = device

        self.R = cfg['R']       # radius (μm)
        self.tau = cfg['tau']   # τ (h)
        self.C0 = cfg['C0']     # c_sol (nM)

        self.pde_parameters = nn.ParameterDict({
            'd': nn.Parameter(torch.tensor(cfg['d']), requires_grad=False), # (μm^2/sec) diffusion coefficient of the free antibodies in the interstitium  
            'k_off': nn.Parameter(torch.tensor(cfg['k_off']), requires_grad=False), # (1/sec) dissosiation rate constant on the cell surface
            'k_d': nn.Parameter(torch.tensor(cfg['k_d']), requires_grad=False), # (nM) k_d = k_off / k_on, equilibrium dissociation constant between the antibody and its antigen.
            'k_int': nn.Parameter(torch.tensor(cfg['k_int']), requires_grad=False), # (1/sec) rate constant for the internalization rate of the antibody-receptor complex
            'r_t': nn.Parameter(torch.tensor(cfg['r_t']), requires_grad=False), # (nM) the initial unbound receptor concentration.
            'p_up': nn.Parameter(torch.tensor(cfg['p_up']), requires_grad=False), # (μm/sec) mass transfer coefficient for the uptake experiment
            'p_cl': nn.Parameter(torch.tensor(cfg['p_cl']), requires_grad=False), # (μm/sec) mass transfer coefficient for the clearance experiment
        }).to(self.device)

        self.pde_weights = nn.ParameterDict({
            'eq_f': nn.Parameter(torch.tensor(cfg['eq1'])),
            'eq_b': nn.Parameter(torch.tensor(cfg['eq2'])),
            'eq_i': nn.Parameter(torch.tensor(cfg['eq3'])),
            'center': nn.Parameter(torch.tensor(cfg['center'])),
            'surface': nn.Parameter(torch.tensor(cfg['surface'])),
            'ic_f': nn.Parameter(torch.tensor(cfg['ic'])),
            'ic_b': nn.Parameter(torch.tensor(cfg['ic'])),
            'ic_i': nn.Parameter(torch.tensor(cfg['ic']))
        }).to(self.device)

        self.pi = self._calculate_pi_groups()

    def _calculate_pi_groups(self):
        p = self.pde_parameters
        k_on = p['k_off'] / p['k_d']
        return {
            'diff': p['d'] * (1.0 / self.R**2) * 3600 * self.tau,
            'on': k_on * self.C0 * 3600 * self.tau,
            'off': p['k_off'] * 3600 * self.tau,
            'int': p['k_int'] * 3600 * self.tau,
            's_ratio': p['r_t'] / self.C0,
            'surface': self.R / p['d'] * p['p_up'] 
        }
    
    def get_phi(self, r: torch.Tensor):
        return 0.44 * r**3.2 + 0.56

    def get_pde_parameters(self):
        return {key: value for key, value in self.pde_parameters.items()}
    
    def get_pde_weights(self):
        return {key: value for key, value in self.pde_weights.items()}
            
        