import torch
from .datasets import DatasetSampler
from .neural_nets import FCNN
from . import residuals

class Pinn():
    def __init__(self, n_col, n_initial, n_center, n_surface, l_bounds=[0, 0], u_bounds=[1, 1/24], device='mps', seed=42, dtype=torch.float32, layers=4, neurons=16):
        self.seed=seed
        self.device = device if device else('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype

        self.lower_bounds = l_bounds
        self.upper_bounds = u_bounds

        self.n_col = n_col
        self.n_initial = n_initial
        self.n_center = n_center
        self.n_surface = n_surface

        self.net = FCNN(n_layers=layers, n_neurons=neurons, seed=self.seed).to(self.device)

        # Dataset atttributes
        self.sampler = DatasetSampler(seed=self.seed)
        self.collocation_training_dataset = self.sampler.sample_collocation_points(n_points=self.n_col, l_bounds=l_bounds, u_bounds=u_bounds)
        self.initial_training_dataset = self.sampler.sample_initial_points(n_points=self.n_initial, l_bounds=l_bounds, u_bounds=u_bounds)
        self.center_training_dataset = self.sampler.sample_center_points(n_points=self.n_center, l_bounds=l_bounds, u_bounds=u_bounds)
        self.surface_training_dataset = self.sampler.sample_surface_points(n_points=self.n_surface, l_bounds=l_bounds, u_bounds=u_bounds)
        self.sensor_training_dataset = None
        self.new_pde_training_dataset = None
        
        self.DATASET = self.collocation_training_dataset + self.center_training_dataset + self.surface_training_dataset + self.initial_training_dataset

        # History attributes
        self.history = {
            'total': [],
            'pde0': [], 'pde1': [], 'pde2': [],
            'center': [], 'surface': [],
            'ic0': [],'ic1': [],'ic2': [],
        }

        # Remember the number of iteretions
        self.current_iter = 0
        self.adam_iter = 0
        self.lbfgs_iter = 0
        

    def get_losses(self, w_pde0=1.0, w_pde1=1.0, w_pde2=1.0, w_ic0=1.0, w_ic1=1.0, w_ic2=1.0, w_center=1.0, w_surface=1.0):
        # // residuals.
        pde0, pde1, pde2 = residuals.pde(net=self.net, dataset=self.collocation_training_dataset)
        ic0, ic1, ic2 = residuals.initial(net=self.net, dataset=self.initial_training_dataset, target_values=(0, 0, 0))
        center = residuals.center_neumann(net=self.net, dataset=self.center_training_dataset, target=0.0)
        surface = residuals.surface_robin(net=self.net, dataset=self.surface_training_dataset)
        # // losses.
        loss_0 = w_pde0 * torch.mean(pde0**2) 
        loss_1 = w_pde1 * torch.mean(pde1**2) 
        loss_2 = w_pde2 * torch.mean(pde2**2)
        loss_ic0 = w_ic0 * torch.mean(ic0**2)
        loss_ic1 = w_ic1 * torch.mean(ic1**2)
        loss_ic2 = w_ic2 * torch.mean(ic2**2)
        loss_center = w_center * torch.mean(center**2)
        loss_surface = w_surface * torch.mean(surface**2)
        
        # // total loss.
        loss = loss_0 + loss_1 + loss_2 + loss_ic0 + loss_ic1 + loss_ic2 + loss_center + loss_surface

        return {
            'total': loss,
            'pde0': loss_0, 'pde1': loss_1, 'pde2': loss_2,
            'center': loss_center, 'surface': loss_surface,
            'ic0': loss_ic0,'ic1': loss_ic1,'ic2': loss_ic2,
        }
    

    def train_adam(self, epochs=2000, network_lr=1e-3, log_every=200):
        optimizer = torch.optim.Adam(params=self.net.parameters(), lr=network_lr)
        print(f'✅ Optimizer intantiated\n{optimizer}')   
        print(f'Adam Training started...\n- Epochs: {epochs}')       

        for epoch in range(epochs):
            optimizer.zero_grad()

            losses = self.get_losses(w_surface=10000, w_pde0=1e-3, w_pde1=1e-3, w_pde2=1e-3)
            loss = losses['total']

            self._record_loss_history(losses)
            self._log(epoch, losses, every=log_every)

            loss.backward()
                    
            optimizer.step()    
            self.adam_iter += 1
            self.current_iter += 1

    def train_lbfgs(self, epochs, max_iter=20, lr=0.001, log_every=200):
        print(f'\n --- Starting LBFGS fine-tuning for {epochs * max_iter} iterations --- ')
        
        optimizer = torch.optim.LBFGS(
            params=self.net.parameters(),
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

                losses = self.get_losses()
                loss = losses['total']

                loss.backward()
                
                self._record_loss_history(losses)
                self._log(epoch, losses, every=log_every)
                
                self.lbfgs_iter += 1
                self.current_iter += 1
                return loss

            optimizer.step(closure)


    def _record_loss_history(self, losses: dict):
        [self.history.setdefault(key, []).append(value.item()) for key, value in losses.items()]
            
    def _log(self, epoch, losses: dict, every: int):
        if epoch == 0:
            header = f"{'Epoch':^10} | {'Total':^10} | {'pde_0':^10} | {'pde_1':^10} | {'pde_2':^10} | {'bc0':^10} | {'bcR':^10} | {'ic_0':^10} | {'ic_1':^10} | {'ic_2':^10} "
            print('-' * len(header))
            print(header)
            print('-' * len(header))
        if epoch % every == 0:
            line = f"{epoch:^10} | {losses['total'].item():^10.4e} | {losses['pde0'].item():^10.4e} | {losses['pde1'].item():^10.4e} | {losses['pde2'].item():^10.4e} | {losses['center'].item():^10.4e} | {losses['surface'].item():^10.4e} | {losses['ic0'].item():^10.4e} | {losses['ic1'].item():^10.4e} | {losses['ic2'].item():^10.4e}"
            print(line)


def get_spatial_antibody_distribution(t_exp, pinn: Pinn):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 9))
    r = torch.linspace(0, 1, 100).reshape(-1, 1)
    for time in [0, 0.25 * t_exp, 0.5 * t_exp, 0.75 * t_exp, t_exp]:
        t = torch.full_like(r, time)
        with torch.no_grad():
            c0, _, _ = pinn.net(r, t)
        plt.plot(r, c0, label=f't={time * 24}h')
    plt.title(f'spatial distribution of TRM || time: [0 - {t_exp * 24}h]')
    plt.xlabel(f'r')
    plt.ylabel(f'[Ab_I]/[C_reference]')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # RES.R_T = 100 # make Rt smaller to reduce stifness.

    # // hyperparameters.
    T = 24 / 24
    # //
    DEVICE = 'cpu'
    # //
    NUM_COLLOC = 2000
    NUM_INITIAL = 200
    NUM_CENTER = 200
    NUM_SURFACE = 200
    # //
    ADAM_EPOCHS = 8000
    LBFGS_EPOCHS = 2000
    # //
    forward_pinn = Pinn(NUM_COLLOC, NUM_INITIAL, NUM_CENTER, NUM_SURFACE, layers=3, neurons=64, device=DEVICE, u_bounds=[1, T])
    forward_pinn.DATASET.plotme()
    forward_pinn.train_adam(epochs=ADAM_EPOCHS)
    get_spatial_antibody_distribution(t_exp = T, pinn=forward_pinn)

    forward_pinn.train_lbfgs(LBFGS_EPOCHS)
    get_spatial_antibody_distribution(t_exp = T, pinn=forward_pinn)

    