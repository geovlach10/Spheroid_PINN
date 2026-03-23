
from math import inf
import copy
import openpyxl

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from core.data import DataGenerator, PINNDataset
from core.model import PINN
from physics.residuals import pde_loss_fn, bc_loss_fn, ic_loss_fn, sensor_loss_fn
from core.context import PhysicsContext

class PINNSolver():
    def __init__(self, pde_loss_fn, bc_loss_fn, ic_loss_fn, physics_context: PhysicsContext, sensor_loss_fn=None, device=None, seed=42, dtype=torch.float32, n_pde=4000, layers=4, neurons=16):
        self.seed=seed
        self.device = device if device else('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype

        self.ctx = physics_context
        # self.R = 200.0 # μm
        # self.tau = 50.0 # h
        # self.C0 = 60.0 # nM

        # These must be fed into the approximator
        # self.pinn_parameters = {
        #     'equation_coefficients' : {
        #         'd': nn.Parameter(torch.tensor(8.38)), # (μm^2/sec) diffusion coefficient of the free antibodies in the interstitium  
        #         'k_off': nn.Parameter(torch.tensor(4e-3)), # (1/sec) dissosiation rate constant on the cell surface
        #         'k_d': nn.Parameter(torch.tensor(6.76)), # (nM) k_d = k_off / k_on, equilibrium dissociation constant between the antibody and its antigen.
        #         'k_int': nn.Parameter(torch.tensor(1.4e-5)), # (1/sec) rate constant for the internalization rate of the antibody-receptor complex
        #         'r_t': nn.Parameter(torch.tensor(1060.0)), # (nM) the initial unbound receptor concentration.
        #         'p_up': nn.Parameter(torch.tensor(2.5e-4)), # (μm/sec) mass transfer coefficient for the uptake experiment
        #         'p_cl': nn.Parameter(torch.tensor(2.6e-1)), # (μm/sec) mass transfer coefficient for the clearance experiment
        #     },
        #     'dynamic_weights' : {
        #         'eq1': nn.Parameter(torch.tensor(1.0)),
        #         'eq2': nn.Parameter(torch.tensor(1.0)),
        #         'eq3': nn.Parameter(torch.tensor(1.0)),
        #         'center': nn.Parameter(torch.tensor(1.0)),
        #         'surface': nn.Parameter(torch.tensor(1.0)),
        #         'ic': nn.Parameter(torch.tensor(1.0))
        #     }
        # }

        # Model
        self.approximator = PINN(n_layers=layers, n_neurons=neurons, seed=self.seed)
        self.approximator.to(self.device)   # I excecute these two lines during instatiantion.
        # self.approximator.requires_grad_()  # at first I set every parameter to be included in the graph 
        #                                      # and inside each different aproch i explicitly say to the optimizer what to freeze.
        # self.approximator.show_if_requires_grad()

        # Dataset atttributes
        self.generator = DataGenerator(seed=self.seed)
        self.pde_training_dataset = PINNDataset(n_points=n_pde, sampling_type='lhc', generator=self.generator, device=self.device, dtype=self.dtype, name='pde')
        self.bc_center_training_dataset = PINNDataset(n_points=200, sampling_type='center', generator=self.generator, device=self.device, dtype=self.dtype, name='symmetry')
        self.bc_surface_training_dataset = PINNDataset(n_points=200, sampling_type='surface', generator=self.generator, device=self.device, dtype=self.dtype, name='surface')
        self.ic_training_dataset = PINNDataset(n_points=100, sampling_type='initial', generator=self.generator, device=self.device, dtype=self.dtype, name='ic')
        self.sensor_training_dataset = None
        self.new_pde_training_dataset = None

        # loss attributes
        self.pde_loss_fn = pde_loss_fn
        self.bc_loss_fn = bc_loss_fn
        self.ic_loss_fn = ic_loss_fn
        self.sensor_loss_fn = sensor_loss_fn

        self.optim_groups = {
            'pde_weights' : [{'params': self.ctx.pde_weights, 'lr': 1.0, 'maximize': True, 'name': 'pde_weights'}],
            'network': [{'params': self.approximator.parameters(), 'lr': 1e-4, 'name': 'network'}],
            'pde_parameters': [{'params': self.ctx.pde_parameters , 'lr': 1e-3, 'name': 'pde_parameters'}]
        }

        # History attributes
        self.loss_history = {
          'loss': [],
          'loss_pde_eq1': [],
          'loss_pde_eq2': [],
          'loss_pde_eq3': [],
          'loss_bc_center': [],
          'loss_bc_surface': [],
          'loss_ic': [],
          'loss_sensor': []
        }
        self.pde_weights_history = {
          'eq1': [],
          'eq2': [],
          'eq3': [],
          'center': [],
          'surface': [],
          'ic': [],
          'sensor': []
        }
        self.pde_parameters_history = {
          'd': [],
          'k_off': [],
          'k_d': [],
          'k_int': [],
          'r_t': [],
          'p_up': [],
          'p_cl': []
        }

        # Best state atttributes
        self.best_loss = 1e30
        self.best_model_state = None
        self.best_coef_d = None

        # Remember the number of iteretions
        self.current_iter = 0
        self.lbfgs_iter = 0

        print(f' {10 * "=="} Solver Manager {10 * "=="}\n ')
        print(f'data generator seed: {self.generator.seed}\napproximator seed: {self.approximator.seed}\ndevice: {self.device}\ndtype: {self.dtype}')
        print(f' {10 * "--"} Dataset information {10 * "--"} ')
        print(f'PDE dataset: {self.pde_training_dataset.n_points} points, sampling type: {self.pde_training_dataset.sampling_type}, device: {self.pde_training_dataset.device}, dtype: {self.pde_training_dataset.dtype}')
        print(f'Boundary dataset (left): {self.bc_center_training_dataset.n_points} points, device: {self.bc_center_training_dataset.device}, dtype: {self.bc_center_training_dataset.dtype}')
        print(f'Boundary dataset (right): {self.bc_surface_training_dataset.n_points} points, device: {self.bc_surface_training_dataset.device}, dtype: {self.bc_surface_training_dataset.dtype}')
        print(f'Initial condition dataset: {self.ic_training_dataset.n_points} points, device: {self.ic_training_dataset.device}, dtype: {self.ic_training_dataset.dtype}')
        print(f' {10 * "--"} Neural network information {10 * "--"} ')
        print(f'Neural network with {self.approximator.n_layers} layers and {self.approximator.n_neurons} neurons in each layer')
        print(f'Output size: {self.approximator.output_dim} x (batch_size * 1)')
        print(f'{self.approximator}')

    # def _log_dynamic_weights(self):
    #     print('\ninspect dynamic weight values:')
    #     for name, param in self.approximator.dynamic_weights.items():
    #           print(f'{name}: {param.item():.5f}')

    def train_adam(self, epochs=2000, problem_type='forward', network_lr=1e-3, dynweight_lr=1.0, inference_lr=1e-3, method='vanilla', resample_every=100, rard_sample_size=200, update_weights_every=500, wait_until=5000, accumulate=False):
        print(f' --- Starting Adam optimization --- ')
        print(f'epochs: {epochs}')
        print(f'problem type: {problem_type}')

        if method == 'vanilla':
            print(f'{method} method applied')
            # self._log_dynamic_weights()
        
            optim_group = self.optim_groups['network'] + self.optim_groups['inference'] if problem_type == 'inference' else self.optim_groups['network']
            optimizer = torch.optim.Adam(optim_group)
            print(f'\ninspect vanilla optimizer\n{optimizer}')            

            for epoch in range(epochs):
                # make sure the optimizers dont sum up previeusly cimputed gradients
                optimizer.zero_grad()

                # compute the criterion imported form physics.py
                # this step acctualy calles the neural network so is the forward step

                # # add a possitivity term
                # c1, c2, c3 = self.approximator(self.pde_training_dataset.r, self.pde_training_dataset.t)
                # loss_possitivity = torch.mean(torch.relu(-c1)**2) + torch.mean(torch.relu(-c2)**2) + torch.mean(torch.relu(-c3)**2)

                loss_pde_eq1, loss_pde_eq2, loss_pde_eq3 = self.pde_loss_fn(approximator=self.approximator, data=self.pde_training_dataset, ctx=self.ctx)
                loss_bc_center, loss_bc_surface = self.bc_loss_fn(approximator=self.approximator, data_center=self.bc_center_training_dataset, data_surface=self.bc_surface_training_dataset, ctx=self.ctx)
                loss_ic = self.ic_loss_fn(self.approximator, self.ic_training_dataset, ctx=self.ctx)
                loss_sensor = self.sensor_loss_fn(self.approximator, self.sensor_training_dataset, self.sensor_output) if self.sensor_training_dataset is not None else torch.tensor(0.0)
                loss = loss_pde_eq1 + loss_pde_eq2 + loss_pde_eq3 + loss_bc_center + loss_bc_surface + loss_ic + loss_sensor

                # traverse the graph backwords to populate the .grad attribute of leaf tensors
                loss.backward()
                self.loss_history['loss'].append(loss.item())
                self.loss_history['loss_pde_eq1'].append(loss_pde_eq1.item())
                self.loss_history['loss_pde_eq2'].append(loss_pde_eq2.item())
                self.loss_history['loss_pde_eq3'].append(loss_pde_eq3.item())
                self.loss_history['loss_bc_center'].append(loss_bc_center.item())
                self.loss_history['loss_bc_surface'].append(loss_bc_surface.item())
                self.loss_history['loss_ic'].append(loss_ic.item())
                self.loss_history['loss_sensor'].append(loss_sensor.item()) if problem_type == 'inference' else None
                self.coefficients_history['d'].append(self.approximator.coefficients['d'].item()) if problem_type == 'inference' else None
                self.coefficients_history['k_off'].append(self.approximator.coefficients['k_off'].item()) if problem_type == 'inference' else None
                self.coefficients_history['k_int'].append(self.approximator.coefficients['k_int'].item()) if problem_type == 'inference' else None
                self.coefficients_history['k_d'].append(self.approximator.coefficients['k_d'].item()) if problem_type == 'inference' else None
                self.coefficients_history['r_t'].append(self.approximator.coefficients['r_t'].item()) if problem_type == 'inference' else None
                self.coefficients_history['p_up'].append(self.approximator.coefficients['p_up'].item()) if problem_type == 'inference' else None
                self.coefficients_history['p_cl'].append(self.approximator.coefficients['p_cl'].item()) if problem_type == 'inference' else None
                
                
                # Limits the 'global norm' of all gradients to 1.0
                # torch.nn.utils.clip_grad_norm_(self.approximator.parameters(), max_norm=1.0)

                # Update the parameters which has been passed to the optimizers accordimg to some optimization algorithm
                # the difference between the network vs the network optimization lies in the direction of the gradient descent.ascent algorithm
                optimizer.step()

                current_loss = loss.item()
                # if current_loss < self.best_loss:
                #     self.best_loss = current_loss
                #     self.best_model_state = copy.deepcopy(self.approximator.state_dict())
                #     print(f'new best loss saved in RAM --> {self.best_loss} at epoch {epoch}')

                # # recovery if loss explodes over 5x
                # if epoch > 10000 and epoch % 50 == 0:
                #     if  current_loss > self.best_loss * 5:
                #         print(f' --- explosion at epoch {epoch}! reverting to loss {self.best_loss:.4e} ---')
                #         self.approximator.load_state_dict(self.best_model_state)
                #         # manual weight decay
                #         for param_group in optimizer.param_groups:
                #             param_group['lr'] *= 0.5
                #             print(f"new lr: {param_group['lr']}")
                
                
                if epoch == 0:
                    header = f"{'Epoch':^10} | {'Total':^10} | {'pde1':^10} | {'pde2':^10} | {'pde3':^10} | {'bc0':^10} | {'bcR':^10} | {'ic':^10} "
                    print('-' * len(header))
                    print(header)
                    print('-' * len(header))
                if epoch % 200 == 0:
                    line = f"{epoch:^10} | {loss.item():^10.4e} | {loss_pde_eq1.item():^10.4e} | {loss_pde_eq2.item():^10.4e} | {loss_pde_eq3.item():^10.4e} | {loss_bc_center.item():^10.4e} | {loss_bc_surface.item():^10.4e} | {loss_ic.item():^10.4e} "
                    print(line)
                # if epoch % 1000 == 0:
                #     analysis_df = pde_loss_fn(approximator=self.approximator, data=self.pde_training_dataset, ctx=self.ctx,  term_by_term_analysis=True)
                #     save_path = f'results/term_by_term_analysis_epoch_{epoch}.xlsx'
                #     analysis_df.to_excel(save_path, index=False, engine='openpyxl')
                #     print(f'term by term analysis report saved to {save_path}')
                    # print(f"Epoch: {self.current_iter}\n[Loss: {loss.item():e}] - [Eq1: {loss_pde_eq1.item():e}, Eq2: {loss_pde_eq2.item():e}, Eq3: {loss_pde_eq2.item():e}] - [BC_0: {loss_bc_center.item():e}] - [BC_R: {loss_bc_surface.item():e}] - [IC: {loss_ic.item():e}] - [Sensor: {loss_sensor.item():e}] - [d: {self.approximator.coefficients['d'].item():e}]\n{100 * '--'}")
                self.current_iter += 1

        if method == 'random_resampling':
            print(f'{method} method applied')
            self._log_dynamic_weights()
            optimizer = torch.optim.Adam(network_inference_optim_group)

            for epoch in range(epochs):
                if epoch % resample_every == 0 and epoch > 0:
                    # change the .collocation_training_set attribute of generator object
                    self.generator.collocation_training_set = self.generator._generate_collocation_points(seed_offset=epoch)
                    pde_set = self.generator.collocation_training_set
                    self.vizualize_residual_heatmap()
                    #self.generator.plot_dataset()
                    print(f'resampled {self.generator.n_collocation} points at epoch {epoch}')

                # make sure the optimizers dont sum up previeusly cimputed gradients
                optimizer.zero_grad()

                # compute the criterion imported form physics.py
                # this step acctualy calles the neural network so is the forward step
                loss_pde_eq1, loss_pde_eq2, loss_pde_eq3 = self.pde_loss_fn(approximator=self.approximator, data=self.pde_training_dataset)
                loss_bc_center, loss_bc_surface = self.bc_loss_fn(approximator=self.approximator, data_center=self.bc_center_training_dataset, data_surface=self.bc_surface_training_dataset)
                loss_ic = self.ic_loss_fn(self.approximator, self.ic_training_dataset)
                loss_sensor = self.sensor_loss_fn(self.approximator, self.sensor_training_dataset, self.sensor_output) if self.sensor_training_dataset is not None else torch.tensor(0.0)
                loss = loss_pde_eq1 + loss_pde_eq2 + loss_pde_eq3 + loss_bc_center + loss_bc_surface + loss_ic + loss_sensor

                # traverse the graph backwords to populate the .grad attribute of leaf tensors
                loss.backward()
                self.loss_history['loss'].append(loss.item())
                self.loss_history['loss_sensor'].append(loss_sensor.item())
                self.loss_history['loss_pde'].append(loss_pde.item())
                self.loss_history['loss_bc_center'].append(loss_bc_center.item())
                self.loss_history['loss_bc_surface'].append(loss_bc_surface.item())
                self.loss_history['loss_ic'].append(loss_ic.item())
                self.coefficients_history['coef_d'].append(self.approximator.coefficients['coef_d'].item())

                # Update the parameters which has benn passed to the optimizers accordimg to some optimization algorithm
                # the difference between the network vs the network optimization liea in the direction of the gradient descent.ascent algorithm
                optimizer.step()
                self.current_iter += 1
                if epoch % 1000 == 0:
                    print(f"Epoch: {self.current_iter}\n[Loss: {loss.item():e}] - [Eq1: {loss_pde_eq1.item():e}, Eq2: {loss_pde_eq2.item():e}, Eq3: {loss_pde_eq2.item():e}] - [BC_0: {loss_bc_center.item():e}] - [BC_R: {loss_bc_surface.item():e}] - [IC: {loss_ic.item():e}] - [Sensor: {loss_sensor.item():e}] - [d: {self.approximator.coefficients['d'].item():e}]\n{100 * '--'}")

        if method == 'adaptive_resampling':
            print(f'{method} method applied')
            print(f'the resamling will not start until {wait_until} epoch passes!!')
            self._log_dynamic_weights()

            self.approximator.freeze_dynweight_params()
            self.approximator.show_if_requires_grad()

            optim_group = self.optim_groups['network'] + self.optim_groups['inference'] if problem_type == 'inference' else self.optim_groups['network']
            optimizer = torch.optim.Adam(optim_group)
            print(f'\ninspect optimizer\n{optimizer}')

            for epoch in range(epochs):
                if epoch % resample_every == 0 and epoch > wait_until:
                    self.pde_training_dataset.resample_adaptive(approximator=self.approximator, pde_loss_fn=self.pde_loss_fn, sample_size=rard_sample_size,accumulate=accumulate, max_points=5000)
                    self.pde_training_dataset.plotme()

                # make sure the optimizers dont sum up previeusly cimputed gradients
                optimizer.zero_grad()

                # compute the criterion imported form physics.py
                # this step acctualy calles the neural network so is the forward step
                loss_pde_eq1, loss_pde_eq2, loss_pde_eq3 = self.pde_loss_fn(approximator=self.approximator, data=self.pde_training_dataset)
                loss_bc_center, loss_bc_surface = self.bc_loss_fn(approximator=self.approximator, data_center=self.bc_center_training_dataset, data_surface=self.bc_surface_training_dataset)
                loss_ic = self.ic_loss_fn(self.approximator, self.ic_training_dataset)
                loss_sensor = self.sensor_loss_fn(self.approximator, self.sensor_training_dataset, self.sensor_output) if self.sensor_training_dataset is not None else torch.tensor(0.0)
                loss = loss_pde_eq1 + loss_pde_eq2 + loss_pde_eq3 + loss_bc_center + loss_bc_surface + loss_ic + loss_sensor
                
                # traverse the graph backwords to populate the .grad attribute of leaf tensors
                loss.backward()
                self.loss_history['loss'].append(loss.item())
                self.loss_history['loss_pde_eq1'].append(loss_pde_eq1.item())
                self.loss_history['loss_pde_eq2'].append(loss_pde_eq2.item())
                self.loss_history['loss_pde_eq3'].append(loss_pde_eq3.item())
                self.loss_history['loss_bc_center'].append(loss_bc_center.item())
                self.loss_history['loss_bc_surface'].append(loss_bc_surface.item())
                self.loss_history['loss_ic'].append(loss_ic.item()) 
                self.loss_history['loss_sensor'].append(loss_sensor.item()) if problem_type=='inference' else None
                self.coefficients_history['d'].append(self.approximator.coefficients['d'].item()) if problem_type=='inference' else None
                self.coefficients_history['k_off'].append(self.approximator.coefficients['k_off'].item()) if problem_type=='inference' else None
                self.coefficients_history['k_int'].append(self.approximator.coefficients['k_int'].item()) if problem_type=='inference' else None
                self.coefficients_history['k_d'].append(self.approximator.coefficients['k_d'].item()) if problem_type=='inference' else None
                self.coefficients_history['r_t'].append(self.approximator.coefficients['r_t'].item()) if problem_type=='inference' else None
                self.coefficients_history['p_up'].append(self.approximator.coefficients['p_up'].item()) if problem_type=='inference' else None
                self.coefficients_history['p_cl'].append(self.approximator.coefficients['p_cl'].item()) if problem_type=='inference' else None

                # Update the parameters which has benn passed to the optimizers accordimg to some optimization algorithm
                # the difference between the network vs the network optimization liea in the direction of the gradient descent.ascent algorithm
                optimizer.step()
                self.current_iter += 1
                if epoch % 1000 == 0:
                    print(f"Epoch: {self.current_iter}\n[Loss: {loss.item():e}] - [Eq1: {loss_pde_eq1.item():e}, Eq2: {loss_pde_eq2.item():e}, Eq3: {loss_pde_eq2.item():e}] - [BC_0: {loss_bc_center.item():e}] - [BC_R: {loss_bc_surface.item():e}] - [IC: {loss_ic.item():e}] - [Sensor: {loss_sensor.item():e}] - [d: {self.approximator.coefficients['d'].item():e}]\n{100 * '--'}")

        if method == 'dynweight':
            print(f'using {method} method')
            print(f'dynamic weight algorithm will not beegin until {wait_until} epoch passes!!!')
            
            # self.approximator.unfreeze_dynweight_params()
            # self.approximator.show_if_requires_grad()

            network_inference_optimizer = torch.optim.Adam(self.optim_groups['network'])
            weight_optimizer = torch.optim.Adam(self.optim_groups['dynweight'])

            for epoch in range(epochs):
                # make sure the optimizers dont sum up previeusly computed gradients
                network_inference_optimizer.zero_grad()
                weight_optimizer.zero_grad()

                # compute the criterion imported form physics.py
                # this step acctualy calles the neural network so is the forward step
                loss_pde_eq1, loss_pde_eq2, loss_pde_eq3 = self.pde_loss_fn(approximator=self.approximator, data=self.pde_training_dataset)
                loss_bc_center, loss_bc_surface = self.bc_loss_fn(approximator=self.approximator, data_center=self.bc_center_training_dataset, data_surface=self.bc_surface_training_dataset)
                loss_ic = self.ic_loss_fn(self.approximator, self.ic_training_dataset)
                loss_sensor = self.sensor_loss_fn(self.approximator, self.sensor_training_dataset, self.sensor_output) if self.sensor_training_dataset is not None else torch.tensor(0.0)
                loss = loss_pde_eq1 + loss_pde_eq2 + loss_pde_eq3 + loss_bc_center + loss_bc_surface + loss_ic + loss_sensor


                # traverse the graph backwards to populate the .grad attribute of leaf tensors
                loss.backward()
                self.loss_history['loss'].append(loss.item())
                self.loss_history['loss_pde_eq1'].append(loss_pde_eq1.item())
                self.loss_history['loss_pde_eq2'].append(loss_pde_eq2.item())
                self.loss_history['loss_pde_eq3'].append(loss_pde_eq3.item())
                self.loss_history['loss_bc_center'].append(loss_bc_center.item())
                self.loss_history['loss_bc_surface'].append(loss_bc_surface.item())
                self.loss_history['loss_ic'].append(loss_ic.item())
                self.loss_history['loss_sensor'].append(loss_sensor.item())
                self.coefficients_history['d'].append(self.approximator.coefficients['d'].item())
                self.coefficients_history['k_off'].append(self.approximator.coefficients['k_off'].item())
                self.coefficients_history['k_int'].append(self.approximator.coefficients['k_int'].item())
                self.coefficients_history['k_d'].append(self.approximator.coefficients['k_d'].item())
                self.coefficients_history['r_t'].append(self.approximator.coefficients['r_t'].item())
                self.coefficients_history['p_up'].append(self.approximator.coefficients['p_up'].item())
                self.coefficients_history['p_cl'].append(self.approximator.coefficients['p_cl'].item())
                self.dynweight_history['eq1'].append(self.approximator.dynamic_weights['eq1'].item())
                self.dynweight_history['eq2'].append(self.approximator.dynamic_weights['eq2'].item())
                self.dynweight_history['eq3'].append(self.approximator.dynamic_weights['eq3'].item())
                self.dynweight_history['center'].append(self.approximator.dynamic_weights['center'].item())
                self.dynweight_history['surface'].append(self.approximator.dynamic_weights['surface'].item())
                self.dynweight_history['ic'].append(self.approximator.dynamic_weights['ic'].item())
                self.dynweight_history['sensor'].append(self.approximator.dynamic_weights['sensor'].item()) if problem_type=='inference' else None


                # Update the parameters which has benn passed to the optimizers accordimg to some optimization algorithm
                # the difference between the network vs the network optimization liea in the direction of the gradient descent.ascent algorithm
                network_inference_optimizer.step()
                weight_optimizer.step() if epoch % update_weights_every == 0 and epoch > wait_until else None
                    
                self.current_iter += 1

                if epoch % 1000 == 0:
                    print(f"Epoch: {self.current_iter}\n[Loss: {loss.item():e}] - [Eq1: {loss_pde_eq1.item():e}, Eq2: {loss_pde_eq2.item():e}, Eq3: {loss_pde_eq2.item():e}] - [BC_0: {loss_bc_center.item():e}] - [BC_R: {loss_bc_surface.item():e}] - [IC: {loss_ic.item():e}] - [Sensor: {loss_sensor.item():e}] - [d: {self.approximator.coefficients['d'].item():e}]\n{100 * '--'}")
                    self._log_dynamic_weights()

    def train_lbfgs(self, epochs, problem_type='forward'):
        print(f' --- Starting LBFGS fine-tuning for {epochs} epochs --- ')
        print(f'dataset being used:')
        self.pde_training_dataset.plotme()
        self.approximator.load_state_dict(self.best_model_state)
        print(f'model reloaded on the best state, loss{self.best_loss}')
        
        self.approximator.freeze_coeffitients_params() if problem_type=='forward' else self.approximator.unfreeze_coeffitients_params()
        self.approximator.freeze_dynweight_params()
        self.approximator.show_if_requires_grad()

        optimizer = torch.optim.LBFGS(
          params=self.approximator.parameters(),
          lr=1e-3,
          max_iter=20,
          tolerance_grad=1e-9,
          tolerance_change=1e-9,
          history_size=10,
          line_search_fn='strong_wolfe'
        )
        for epoch in range(epochs):
            def closure():
                optimizer.zero_grad()

                loss_pde_eq1, loss_pde_eq2, loss_pde_eq3 = self.pde_loss_fn(approximator=self.approximator, data=self.pde_training_dataset)
                loss_bc_center, loss_bc_surface = self.bc_loss_fn(approximator=self.approximator, data_center=self.bc_center_training_dataset, data_surface=self.bc_surface_training_dataset)
                loss_ic = self.ic_loss_fn(self.approximator, self.ic_training_dataset)
                loss_sensor = self.sensor_loss_fn(self.approximator, self.sensor_training_dataset, self.sensor_output) if self.sensor_training_dataset is not None else torch.tensor(0.0)
                loss = loss_pde_eq1 + loss_pde_eq2 + loss_pde_eq3 + loss_bc_center + loss_bc_surface + loss_ic + loss_sensor
                loss.backward()

                self.loss_history['loss'].append(loss.item())
                self.loss_history['loss_pde_eq1'].append(loss_pde_eq1.item())
                self.loss_history['loss_pde_eq2'].append(loss_pde_eq2.item())
                self.loss_history['loss_pde_eq3'].append(loss_pde_eq3.item())
                self.loss_history['loss_bc_center'].append(loss_bc_center.item())
                self.loss_history['loss_bc_surface'].append(loss_bc_surface.item())
                self.loss_history['loss_ic'].append(loss_ic.item())
                self.loss_history['loss_sensor'].append(loss_sensor.item()) if problem_type == 'inference' else None

                self.current_iter += 1
                self.lbfgs_iter += 1
                if self.lbfgs_iter % 100 == 0 and self.lbfgs_iter > 0:
                    print(f'LBFGS current iter {self.lbfgs_iter}')
                    print(f"Epoch: {self.current_iter}\n[Loss: {loss.item():e}] - [Eq1: {loss_pde_eq1.item():e}, Eq2: {loss_pde_eq2.item():e}, Eq3: {loss_pde_eq2.item():e}] - [BC_0: {loss_bc_center.item():e}] - [BC_R: {loss_bc_surface.item():e}] - [IC: {loss_ic.item():e}] - [Sensor: {loss_sensor.item():e}] - [d: {self.approximator.coefficients['d'].item():e}]\n{100 * '--'}")
                return loss

            optimizer.step(closure)
            current_loss = closure().item()

            # Ckeckpoint for keeping the best state of the model
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_model_state = self.approximator.state_dict()
                self.best_coef_d = self.approximator.coefficients['coef_d'].item() if problem_type=='iference' else 0.0
                torch.save({
                    'epoch': self.current_iter,
                    'model_state': self.best_model_state,
                    'coef_d discovered': self.best_coef_d,
                    'loss': self.best_loss
                    }, 'best_pinn_checkpoint.pth')
                print(f'New best loss! {current_loss:.6f}, coef_d saved: {self.best_coef_d:.6f}')


            if self.lbfgs_iter > 100:
                if (self.loss_history['loss'][-1] > self.loss_history['loss'][-5]) and (self.loss_history['loss'][-1] > self.loss_history['loss'][-50]):
                    print('LBFGS does not improving')
                    break

        # # Load back the best state of the model
        # self.approximator.load_state_dict(self.best_model_state)
        # print(f'Fine-tuning completed. Restored best model with coed_d={self.best_coef_d}')

    def vizualize_dataset2D(self, dataset:torch.Tensor):
        plt.figure(figsize=(12, 8))
        plt.scatter(dataset[:, 0].detach().numpy(), dataset[:, 1].detach().numpy(), s=10)
        plt.show()

    def vizualize_residual_heatmap(self, grid_dim=100):
        x = np.linspace(self.x_min, self.x_max, grid_dim)
        t = np.linspace(self.t_min, self.t_max, grid_dim)
        X, T = np.meshgrid(x, t)

        x = torch.tensor(X.flatten()[:, None], dtype=self.dtype).to(self.device)
        t = torch.tensor(T.flatten()[:, None], dtype=self.dtype).to(self.device)
        grid_data = torch.cat((x, t), dim=1)

        residual = self.pde_loss_fn(approximator=self.approximator, data=grid_data, pointwise_residual_only=True)
        Z = residual.detach().cpu().numpy().reshape(grid_dim,grid_dim)

        plt.subplots(figsize=(12, 8))
        plt.pcolormesh(X, T, Z,
                       vmin=0,
                       vmax=0.5,
                       cmap='inferno',
                       shading='auto')

        plt.colorbar(label='Absolute PDE residual')
        plt.xlabel('radius')
        plt.ylabel('time')
        plt.title(f'PDE residual map at epoch {self.current_iter}')
        plt.show()

    def vizualize_prediction(self, what :str, timestamps):
        plt.figure(figsize=(12, 8))
        x = torch.linspace(0, self.R, 100).view(-1, 1).to((self.device))
        for time in timestamps:
            t = torch.full_like(x, time*self.tau).to(self.device)
            with torch.no_grad():
                cf, cb, ci = self.approximator(x, t)
                cf, cb, ci = cf*self.C0, cb*self.C0, ci*self.C0 
            if what == 'cf':
                plt.plot(x.cpu().numpy(), cf.cpu(), label=f't = {time}')
            if what == 'cb':
                plt.plot(x.cpu().numpy(), cb.cpu(), label=f't = {time}')
            if what == 'ci':
                plt.plot(x.cpu().numpy(), ci.cpu(), label=f't = {time}')
        plt.xlabel('radius')
        plt.ylabel('concentration')
        plt.legend()
        plt.title('Evolution of concentration over time')
        plt.grid(True, alpha=0.3)
        plt.show()

    def vizualize_coef_d(self, D_true):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(range(len(self.coefficients_history['coef_d'])),self.coefficients_history['coef_d'])
        ax.plot(np.full_like(np.array(self.coefficients_history['coef_d']), D_true), c='red')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("diffucivity")
        ax.set_title("diffucivity vs Training Epochs")
        plt.legend(['diffucivity', 'true diffucivity'])
        plt.savefig('diffucivity_vs_epochs.png')
        plt.show()