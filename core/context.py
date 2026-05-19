import pandas as pd
from tabulate import tabulate

import torch
import torch.nn as nn



cfg = {
    # physical scale constants
    'R': 200,               # (μm)
    'tau_uptake': 24,       # (h)
    'tau_fishing': 0.5/60,  # (h)
    'tau_clearance': 24,    # (h)
    'C_sol_up': 60,         # (nM)
    'C_sol_fi': 0 + 1e-8,         # (nM)
    'C_sol_cl': 0 + 1e-8,         # (nM)
    # pde parameters
    'd': 8.38,              # (μm^2/sec) diffusion coefficient of the free antibodies in the interstitium  
    'k_off': 4e-3,          # (1/sec) dissosiation rate constant on the cell surface
    'k_d': 6.76,            # (nM) k_d = k_off / k_on, equilibrium dissociation constant between the antibody and its antigen.
    'k_int': 1.4e-5,        # (1/sec) rate constant for the internalization rate of the antibody-receptor complex
    'r_t': 1060.0,          # (nM) the initial unbound receptor concentration.
    'p_up': 2.5e-4,         # (μm/sec) mass transfer coefficient for the uptake experiment
    'p_fi': 0,
    'p_cl': 8.0e-3,         # (μm/sec) mass transfer coefficient for the clearance experiment
    # pde weights
    'eq1': 0.0,
    'eq2': 0.0,
    'eq3': 0.0,
    'center': 1.0,
    'surface': 1.0,
    'ic_f': 1.0,
    'ic_b': 1.0,
    'ic_i': 1.0
}  

class PhysicsContext(nn.Module):
    def __init__(self, cfg, device='cpu'):
        super().__init__()
        self.device = device

        self.C_star = 1     # (nm). characteristic concentration. 
        self.R = cfg['R']       # radius (μm)  
        self.tau = {            # τ (h)
            'uptake': cfg['tau_uptake'],
            'fishing': cfg['tau_fishing'],
            'clearance': cfg['tau_clearance'] 
        } 
        self.C_sol = {
            'uptake': cfg['C_sol_up'],     # c_sol (nM)
            'fishing': cfg['C_sol_fi'],     # c_sol (nM)
            'clearance': cfg['C_sol_cl']     # c_sol (nM)
        } 
        
        
        self.P = {
            'uptake': cfg['p_up'],
            'fishing': cfg['p_fi'],
            'clearance': cfg['p_cl']
        }

        self.pde_parameters = nn.ParameterDict({
            'd': nn.Parameter(torch.tensor(cfg['d']), requires_grad=False), # (μm^2/sec) diffusion coefficient of the free antibodies in the interstitium  
            'k_off': nn.Parameter(torch.tensor(cfg['k_off']), requires_grad=False), # (1/sec) dissosiation rate constant on the cell surface
            'k_d': nn.Parameter(torch.tensor(cfg['k_d']), requires_grad=False), # (nM) k_d = k_off / k_on, equilibrium dissociation constant between the antibody and its antigen.
            'k_int': nn.Parameter(torch.tensor(cfg['k_int']), requires_grad=False), # (1/sec) rate constant for the internalization rate of the antibody-receptor complex
            'r_t': nn.Parameter(torch.tensor(cfg['r_t']), requires_grad=False), # (nM) the initial unbound receptor concentration.
            'p_up': nn.Parameter(torch.tensor(cfg['p_up']), requires_grad=False), # (μm/sec) mass transfer coefficient for the uptake experiment
            'p_fi': nn.Parameter(torch.tensor(cfg['p_fi']), requires_grad=False), # (μm/sec) mass transfer coefficient for the fishing experiment
            'p_cl': nn.Parameter(torch.tensor(cfg['p_cl']), requires_grad=False), # (μm/sec) mass transfer coefficient for the clearance experiment
        }).to(self.device)

        self.pde_weights = nn.ParameterDict({
            'eq_f': nn.Parameter(torch.tensor(cfg['eq1']), requires_grad=False),
            'eq_b': nn.Parameter(torch.tensor(cfg['eq2']), requires_grad=False),
            'eq_i': nn.Parameter(torch.tensor(cfg['eq3']), requires_grad=False),
            'center': nn.Parameter(torch.tensor(cfg['center']), requires_grad=False),
            'surface': nn.Parameter(torch.tensor(cfg['surface']), requires_grad=False),
            'ic_f': nn.Parameter(torch.tensor(cfg['ic_f']), requires_grad=False),
            'ic_b': nn.Parameter(torch.tensor(cfg['ic_b']), requires_grad=False),
            'ic_i': nn.Parameter(torch.tensor(cfg['ic_i']), requires_grad=False)
        }).to(self.device)

        self.pi = {
            'uptake': self.get_dimentionaless_params('uptake'),
            'fishing': self.get_dimentionaless_params('fishing'),
            'clearance': self.get_dimentionaless_params('clearance')
        }
        

    def get_dimentionaless_params(self, phase):
        ''' 
        phase: Can be 'uptake', 'fishing', 'clearance'.
        Calculates dimentionless parameters for the normalized domain in each phase of the experiment.
        '''
        p = self.get_pde_parameters()
        tau = self.tau
        # k_on = p['k_off'] / p['k_d'] # 1 / (sec * nM)
        # P = {'uptake': p['p_up'], 'fishing': p['p_fi'], 'clearance': p['p_cl']}
        k_on = cfg['k_off'] / cfg['k_d'] # 1 / (sec * nM)
        P = {'uptake': cfg['p_up'], 'fishing': cfg['p_fi'], 'clearance': cfg['p_cl']}
        
        # non_dimentional_params = {
        #     'diff': p['d'] * (1.0 / self.R**2) * 3600 * tau[phase],
        #     'on1': k_on * 3600 * tau[phase] * p['r_t'],
        #     'off1': p['k_off'] * 3600 * tau[phase] * p['r_t']/self.C_sol[phase] ,
        #     'on2': k_on * 3600 * tau[phase] * self.C_sol[phase],
        #     'off2': p['k_off'] * 3600 * tau[phase],
        #     'int': p['k_int'] * 3600 * tau[phase],
        #     'surface': (P[phase] * self.R) / p['d']  
        # }
        non_dimentional_params = {
            'diff': cfg['d'] * (1.0 / self.R**2) * 3600 * tau[phase],
            'on1': k_on * 3600 * tau[phase] * cfg['r_t'],
            'off1': cfg['k_off'] * 3600 * tau[phase] * cfg['r_t']/self.C_sol[phase] ,
            'on2': k_on * 3600 * tau[phase] * self.C_sol[phase],
            'off2': cfg['k_off'] * 3600 * tau[phase],
            'int': cfg['k_int'] * 3600 * tau[phase],
            'surface': (P[phase] * self.R) / cfg['d']  
        }
        return non_dimentional_params
    
    def get_phi(self, r: torch.Tensor):
        return 0.44 * r**3.2 + 0.56

    def get_pde_parameters(self):
        return {key: value for key, value in self.pde_parameters.items()}
    
    def get_pde_weights(self):
        return {key: value for key, value in self.pde_weights.items()}
            
    # def __repr__(self):
    #     scaling = {
    #         'R(μm)': f'{self.R:.2f}',
    #         'tau(h)': f'{self.tau:.2f}',
    #         'C_sol_up(nM)': f'{self.C_sol:.2f}'
    #     }
    #     df_scaling = pd.DataFrame(scaling.items(), columns=['scaling parameters', 'value'])
    #     df_pde_params = pd.DataFrame(self.get_pde_parameters().items(), columns=['pde parameters', 'value'])
    #     df_pde_weights = pd.DataFrame(self.get_pde_weights().items(), columns=['pde weights', 'value'])
    #     df_pi = pd.DataFrame(self.calculate_pi_groups(), columns=['pi groups', 'value'])
    
    #     df = pd.concat([df_scaling, df_pde_params, df_pde_weights, df_pi], axis=1)
    #     table = tabulate(df, headers='keys', tablefmt='simple', showindex=False, floatfmt=(None, '.1f', None, '.2e', None, '.4f', None, '.6f'))
    #     header = f'{"-" * 110}\n{"Physics Context":^{110}}\n{"-" * 110}'
    #     return f'{header}' \
    #            f'\n{table}' \
    #            f'\n{"-" * 110}'