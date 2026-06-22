from math import inf
from tqdm.notebook import tqdm
import os

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from core.data import Sampler, Dataset
from core.model import PINN
from physics.residuals import pde_loss_fn, bc_loss_fn, ic_loss_fn, sensor_loss_fn
# from physics.residuals_ver2 import get_all_losses
from physics.residuals_ver2 import get_pde_residuals, get_center_residual, get_surface_residual, get_initial_residual, get_ic_l2_penalty

from core.context import PhysicsContext
from utils.logger import log_to_excel, PINNLoger
from utils.visualizer import Static3DVisualizer, cleanup_folder, create_folder_if_not_exists

from utils.decorators import auto_save

import pandas as pd
def data_to_df(data):
    d = {k: [v.item()] for k, v in data.items()}
    return pd.DataFrame(data=d)

class PINNSolver():
    def __init__(self, physics_configuration: dict, device='mps', seed=42, dtype=torch.float32, n_pde=200, layers=4, neurons=16):
        self.seed=seed
        self.device = device if device else('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype

        self.phase_flag = 'uptake'
        self.initial_condition = (0, 0, 0)

        self.configuration = physics_configuration
        self.ctx = PhysicsContext(cfg=self.configuration, device=self.device)

        # Model
        self.approximator = PINN(n_layers=layers, n_neurons=neurons, seed=self.seed)
        self.approximator.to(self.device)   

        self.is_first_window = True
        self.domain = np.array(([0, 0],
                                [1, 1]))

        # Dataset atttributes
        self.sampler = Sampler(seed=self.seed)
        self.pde_training_dataset = self.sampler.sample_interior_dataset(n_points=n_pde, l_bounds=self.domain[0], u_bounds=self.domain[1])
        self.center_training_dataset = self.sampler.sample_center_dataset(n_points=200, l_bounds=self.domain[0], u_bounds=self.domain[1])
        self.surface_training_dataset = self.sampler.sample_surface_dataset(n_points=200, l_bounds=self.domain[0], u_bounds=self.domain[1])
        self.initial_training_dataset = self.sampler.sample_initial_dataset(n_points=200, l_bounds=self.domain[0], u_bounds=self.domain[1])
        self.sensor_training_dataset = None
        self.new_pde_training_dataset = None
        self.entire_dataset_dataset = self.pde_training_dataset + self.center_training_dataset + self.surface_training_dataset + self.initial_training_dataset

        self.optim_groups = {
            'pde_weights' : [{'params': self.ctx.pde_weights, 'lr': 1.0, 'maximize': True, 'name': 'pde_weights'}],
            'network': [{'params': self.approximator.parameters(), 'lr': 1e-3, 'name': 'network'}],
            'pde_parameters': [{'params': self.ctx.pde_parameters , 'lr': 1e-3, 'name': 'pde_parameters'}]
        }
        self.adam = torch.optim.Adam(params=self.approximator.parameters(), lr=0.001)

        # History attributes
        self.history = {'losses': {}, 'weights': {}, 'params': {} }
       
        # Best state atttributes
        self.best_loss = 1e30
        self.best_model_state = None
        self.best_coef_d = None

        # results directory
        self.static_3D_dir = 'results/frames/static_3D'
        self.video_dir = 'results/antibody_penetration_3D.mp4'
        
        # Remember the number of iteretions
        self.current_iter = 0
        self.lbfgs_iter = 0
        self.adam_iter = 0

    def _shift_dataset_boundaries(self, t_start, t_end):
        self.domain = np.array(([0, t_start],
                                [1, t_end]))
        self.pde_training_dataset = self.sampler.sample_interior_dataset(n_points=2000, l_bounds=self.domain[0], u_bounds=self.domain[1])
        self.center_training_dataset = self.sampler.sample_center_dataset(n_points=200, l_bounds=self.domain[0], u_bounds=self.domain[1])
        self.surface_training_dataset = self.sampler.sample_surface_dataset(n_points=200, l_bounds=self.domain[0], u_bounds=self.domain[1])
        self.initial_training_dataset = self.sampler.sample_initial_dataset(n_points=200, l_bounds=self.domain[0], u_bounds=self.domain[1])
        self.entire_dataset_dataset = self.pde_training_dataset + self.center_training_dataset + self.surface_training_dataset + self.initial_training_dataset
        self.entire_dataset_dataset.plotme()
        print(f"time boundaries: {self.domain[0][1]:.3f} - {24 * self.domain[1][1]:.3f}h")

    def _handoff_initial_condition(self):
        self.approximator.eval()
        with torch.no_grad():
            preds = self.approximator.forward(x=self.initial_training_dataset.r, t=self.initial_training_dataset.t)
            self.initial_condition = tuple(p.detach() for p in preds)
        self.approximator.train()
        print('initial condiotion handoffed succesfully')
        print(self.initial_condition)

    def prepare_next_domain_window(self, t_start, t_end):
        if t_start > 0:
            self._handoff_initial_condition()
            self.is_first_window = False
        self._shift_dataset_boundaries(t_start=t_start, t_end=t_end)
    
        print(f"✅ New training window created...\nstart={t_start:.3f}, end={t_end:.3f}")
        print('is first window??: ', self.is_first_window)

    def save_checkpoint(self, stage_name):
        checkpoint_dir = 'checkpoints'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        checkpoint = {
            'model_state_dict': self.approximator.state_dict(),
            'optimizer_state_dict': None,
            'stage': stage_name,
            'final_loss': None
        }

        filename = os.path.join(checkpoint_dir, f"pinn_{stage_name}_{self.phase_flag}.pt")
        torch.save(checkpoint, filename)
        print(f'💾 Checkpoint saved: {filename}')

    def prepare_next_stage(self, new_phase):
        # Update the phase_flag.
        self.phase_flag = new_phase

        # Update dimentionless params from context.
        self.ctx.pi = self.ctx.get_dimentionaless_params(phase=new_phase)

        # Capture the new IC that must be handoffed to the new phase.
        if self.phase_flag == 'uptake':
            self.initial_condition = (0, 0, 0)
        
        elif self.phase_flag == 'fishing' or self.phase_flag == 'clearance':
            self.approximator.eval()
            with torch.no_grad():
                r = self.initial_training_dataset.r
                t = self.initial_training_dataset.t
                predictions = self.approximator(r, t) # Tuple of three column vectors.
                self.initial_condition = tuple(p.detach() for p in predictions)
            
            # Log the parameters.
            self.approximator.train()
        print(f"✅ Phase switched to {new_phase}. IC targets updated!")
        print(f"[initial_condition] {self.initial_condition}")
        print(f'[Phase duration] {self.ctx.tau[new_phase]:.4f} h')
        print('\n[pde weights]\n',data_to_df(self.ctx.get_pde_weights()))
        print('\n[pde params]\n',data_to_df(self.ctx.get_pde_parameters()))
        print('\n[Pi]\n',data_to_df(self.ctx.pi))

    def get_all_losses(self):
        w = self.ctx.get_pde_weights()

        pde_res = get_pde_residuals(model=self.approximator, dataset=self.pde_training_dataset, ctx=self.ctx, phase=self.phase_flag)
        center_res = get_center_residual(model=self.approximator, dataset=self.center_training_dataset)
        surface_res = get_surface_residual(model=self.approximator, dataset=self.surface_training_dataset, ctx=self.ctx, phase=self.phase_flag)
        initial_res = get_initial_residual(model=self.approximator, dataset=self.initial_training_dataset, phase=self.phase_flag, ic_target_values=self.initial_condition, is_first_window=self.is_first_window)
        
        loss_pde_f = w['eq_f'] * torch.mean(pde_res['f']**2) 
        loss_pde_b = w['eq_b'] * torch.mean(pde_res['b']**2) 
        loss_pde_i = w['eq_i'] * torch.mean(pde_res['i']**2)
        loss_center = w['center'] * torch.mean(center_res**2)
        loss_surface = w['surface'] * torch.mean(surface_res**2)
        loss_ic_f = w['ic_f'] * torch.mean(initial_res['f']**2)
        loss_ic_b = w['ic_b'] * torch.mean(initial_res['b']**2)
        loss_ic_i = w['ic_i'] * torch.mean(initial_res['i']**2)

        loss = loss_pde_f + loss_pde_b + loss_pde_i + loss_center + loss_surface + loss_ic_f + loss_ic_b + loss_ic_i 
        
        # return {
        #     'pde': loss_pde_f + loss_pde_b + loss_pde_i,
        #     'bc': loss_center + loss_surface,
        #     'ic': loss_ic_f + loss_ic_b + loss_ic_i,
        # }
        return {
            'total': loss,
            'pde_f': loss_pde_f, 'pde_b': loss_pde_b, 'pde_i': loss_pde_i,
            'center': loss_center, 'surface': loss_surface,
            'ic_f': loss_ic_f,'ic_b': loss_ic_b,'ic_i': loss_ic_i,
            # 'all_ic': loss_ic_f + loss_ic_b + loss_ic_i,
            # 'all_bc': loss_center + loss_surface,
            # 'all_ic_bc': loss_ic_f + loss_ic_b + loss_ic_i + loss_center + loss_surface
        }

    @auto_save
    def train_adam(self, epochs=2000, network_lr=1e-3, log_every=200):
        
        logger = PINNLoger(num_epochs=epochs)
        static_viz_3D = Static3DVisualizer(r_size=20, t_size=200, t_start=self.domain[0][1], t_end=self.domain[1][1], save_dir=self.static_3D_dir)
    
        optimizer = torch.optim.Adam(params=self.approximator.parameters(), lr=network_lr)
        # optimizer = self.adam
        print(f'\n✅ Optimizer intantiated\n{optimizer}')          

        for epoch in range(epochs):
            # Make sure the optimizers dont sum up previeusly computed gradients
            optimizer.zero_grad()

            # Traverse the graph forword.
            # losses = self.get_all_losses(model=self.approximator, pde=self.pde_training_dataset, center=self.center_training_dataset, surface=self.surface_training_dataset, initial=self.initial_training_dataset, ctx=self.ctx)
            losses = self.get_all_losses()
            loss = losses['total']

            # Traverse the graph backwords to populate the .grad attribute of leaf tensors.
            loss.backward()
            
            # Keep a record of everything.
            logger.update(epoch=epoch, every=log_every, losses=losses, weights=self.ctx.get_pde_weights(), parameters=self.ctx.get_pde_parameters())
            static_viz_3D.update(what='cf', model=self.approximator, device=self.device, epoch=epoch, update_every=log_every, phase=self.phase_flag)
            
            # Update the values.
            optimizer.step()    
            
            self.current_iter += 1
        # self.save_checkpoint(f'Adam_{epochs}_epochs')
        return logger.finalize()
        
    def train_adam_rar(self, epochs=2000, problem_type='forward', network_lr=1e-3, dynweight_lr=1.0, inference_lr=1e-3, method='vanilla', resample_every=100, rard_sample_size=200, update_weights_every=500, wait_until=5000, accumulate=False):
        
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
            loss_ic = self.ic_loss_fn(self.approximator, self.initial_training_dataset)
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

    def train_adam_resample(self, epochs=2000, problem_type='forward', network_lr=1e-3, dynweight_lr=1.0, inference_lr=1e-3, method='vanilla', resample_every=100, rard_sample_size=200, update_weights_every=500, wait_until=5000, accumulate=False):
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
            loss_ic = self.ic_loss_fn(self.approximator, self.initial_training_dataset)
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

    def train_adam_dynweight(self, epochs=2000, problem_type='forward', network_lr=1e-3, dynweight_lr=1.0, inference_lr=1e-3, method='vanilla', resample_every=100, rard_sample_size=200, update_weights_every=500, wait_until=5000, accumulate=False):
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
            loss_ic = self.ic_loss_fn(self.approximator, self.initial_training_dataset)
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

    @auto_save
    def train_lbfgs(self, epochs, max_iter=20, lr=0.001, problem_type='forward', what_loss="total"):
        print(f'\n --- Starting LBFGS fine-tuning for {epochs * max_iter} iterations --- ')
        
        optimizer = torch.optim.LBFGS(
          params=self.approximator.parameters(),
          lr=lr,
          max_iter=max_iter,
          tolerance_grad=1e-9,
          tolerance_change=1e-9,
          history_size=10,
          line_search_fn='strong_wolfe'
        )
        print(f'\n✅ Optimizer intantiated\n{optimizer}')

        # Training loop.
        for epoch in range(epochs):
            def closure():
                optimizer.zero_grad()

                losses = self.get_all_losses()
                loss = losses[what_loss]

                loss.backward()
                [self.history['losses'].setdefault(key, []).append(value.item()) for key, value in losses.items()]
                
                # Log to screen.
                if self.lbfgs_iter == 0:
                    header = f"{'Epoch':^10} | {'Total':^10} | {'pde_f':^10} | {'pde_b':^10} | {'pde_i':^10} | {'bc0':^10} | {'bcR':^10} | {'ic_f':^10} | {'ic_b':^10} | {'ic_i':^10} "
                    print('-' * len(header))
                    print(header)
                    print('-' * len(header))
                if self.lbfgs_iter % 100 == 0:
                    line = f"{self.lbfgs_iter:^10} | {losses['total'].item():^10.4e} | {losses['pde_f'].item():^10.4e} | {losses['pde_b'].item():^10.4e} | {losses['pde_i'].item():^10.4e} | {losses['center'].item():^10.4e} | {losses['surface'].item():^10.4e} | {losses['ic_f'].item():^10.4e} | {losses['ic_b'].item():^10.4e} | {losses['ic_i'].item():^10.4e}"
                    print(line)

                self.current_iter += 1
                self.lbfgs_iter += 1
                return loss

            optimizer.step(closure)
            
            # current_loss = closure().item()

            # Ckeckpoint for keeping the best state of the model
            # if current_loss < self.best_loss:
            #     self.best_loss = current_loss
            #     self.best_model_state = self.approximator.state_dict()
            #     self.best_coef_d = self.approximator.coefficients['coef_d'].item() if problem_type=='iference' else 0.0
            #     torch.save({
            #         'epoch': self.current_iter,
            #         'model_state': self.best_model_state,
            #         'coef_d discovered': self.best_coef_d,
            #         'loss': self.best_loss
            #         }, 'best_pinn_checkpoint.pth')
            #     print(f'New best loss! {current_loss:.6f}, coef_d saved: {self.best_coef_d:.6f}')


            # if self.lbfgs_iter > 100:
            #     if (self.loss_history['loss'][-1] > self.loss_history['loss'][-5]) and (self.loss_history['loss'][-1] > self.loss_history['loss'][-50]):
            #         print('LBFGS does not improving')
            #         break

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

    # def vizualize_prediction(self, what :str, timestamps):
    #     ''' timestamps must be a list of times in hours to show the pred. It is divided by tau.'''
    #     p = self.ctx.get_pde_parameters()

    #     plt.figure(figsize=(12, 8))
    #     x = torch.linspace(0, 1, 100).view(-1, 1).to((self.device))
    #     for time in timestamps:
    #         time_norm = time / self.ctx.tau[self.phase_flag]
    #         t = torch.full_like(x, time_norm).to(self.device)
    #         with torch.no_grad():
    #             cf, cb, ci = self.approximator(x, t)
    #             cf, cb, ci = self.ctx.C0 * cf.detach(), p['r_t'] * cb.detach(), p['r_t'] * ci.detach()
    #         if what == 'cf':
    #             plt.plot(x.cpu().numpy(), cf.cpu(), label=f't = {time}h')
    #         if what == 'cb':
    #             plt.plot(x.cpu().numpy(), cb.cpu(), label=f't = {time}h')
    #         if what == 'ci':
    #             plt.plot(x.cpu().numpy(), ci.cpu(), label=f't = {time}h')
    #     plt.xlabel('radius (μm)')
    #     plt.ylabel('concentration (nM)')
    #     plt.legend()
    #     plt.title(f'Evolution of concentration of Trastuzumab over time during {self.phase_flag} phase')
    #     plt.grid(True, alpha=0.3)
    #     plt.show()

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

    # def __repr__(self):
    #     line =110
    #     title = f'{"-" * line}\n{"Solver Information":^{line}}\n{"-" * line}'
    #     seeds = f'data generator seed: {self.generator.seed}\napproximator seed: {self.approximator.seed}\ndevice: {self.device}\ndtype: {self.dtype}'
    #     context = f'{self.ctx}'
    #     datasets = f'{"-" * line}\n{"Dataset information":^{line}}\n{"-" * line}\n' \
    #            f'PDE dataset: {self.pde_training_dataset.n_points} points, sampling type: {self.pde_training_dataset.sampling_type}, device: {self.pde_training_dataset.device}, dtype: {self.pde_training_dataset.dtype}\n' \
    #            f'Boundary dataset (center): {self.center_training_dataset.n_points} points, device: {self.center_training_dataset.device}, dtype: {self.center_training_dataset.dtype}\n' \
    #            f'Boundary dataset (surface): {self.surface_training_dataset.n_points} points, device: {self.surface_training_dataset.device}, dtype: {self.surface_training_dataset.dtype}\n' \
    #            f'Initial condition dataset: {self.initial_training_dataset.n_points} points, device: {self.initial_training_dataset.device}, dtype: {self.initial_training_dataset.dtype}\n'
    #     model = f'{"-" * line}\n{"Neural network information":^{line}}\n{"-" * line}\n' \
    #             f'Neural network with {self.approximator.n_layers} layers and {self.approximator.n_neurons} neurons in each layer\n' \
    #             f'Output size: {self.approximator.output_dim} x (batch_size * 1)\n' \
    #             f'{self.approximator}'
    #     return f'\n{title}\n\n{seeds}\n\n{context}\n\n{datasets}\n\n{model}\n{"-" * line}'