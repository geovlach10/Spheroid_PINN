import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import torch

from core.model import PINN

class DataGenerator():
    '''
    This class contains the methods to create the training datasets for the PDE domain, the center boundary(r=0), 
    the surface boundrary(r=1) and the initial condition(t=0).
    returns:
        numpy.ndarray of size: (n_points * 2)
    '''
    def __init__(self, sensor_file=None, seed=42):
        self.seed = seed
        self.sensor_file = sensor_file

    # methods that handle pde domain
    def create_norm_lhc(self, n_points, seed_offset=0):
        sampler = qmc.LatinHypercube(d=2, seed=self.seed + seed_offset)
        return sampler.random(n=n_points)

    def create_norm_sobol(self, n_points, seed_offset=0):
        sampler = qmc.Sobol(d=2, scramble=True, seed=self.seed + seed_offset)
        return sampler.random(n=n_points)

    def create_norm_grid(self, n_r, n_t):
        r = np.linspace(0, 1, n_r)
        t = np.linspace(0, 1, n_t)
        R, T = np.meshgrid(r, t)
        return np.stack((R.flatten(), T.flatten()), axis=1)

    # methods that handle boundary and initial conditions
    def create_norm_initial_condition(self, n_points):
        r = np.linspace(0, 1, n_points).reshape(-1, 1)
        t = np.zeros_like(r)
        return np.hstack([r, t])

    def create_norm_center_boundary(self, n_points):
        t = np.linspace(0, 1, n_points).reshape(-1, 1)
        r = np.zeros_like(t)
        return np.hstack([r, t])

    def create_norm_surface_boundary(self, n_points):
        t = np.linspace(0, 1, n_points).reshape(-1, 1)
        r = np.ones_like(t)
        return np.hstack([r, t])
    
    def create_sensor_dataset(self, sensor_file):
        data = None
        output = None
        return data, output
    
class PINNDataset():
    def __init__(self, n_points=2000, max_points=5000, approximator: PINN=None, generator: DataGenerator=None, R_phys=200, T_phys=50, sampling_type=None, dtype=torch.float32, device=None, name=None):
        self.name = name
        self.generator = generator
        self.approximator = approximator
        self.sampling_type = sampling_type
        self.n_points = n_points
        self.max_points = max_points
        self.R_phys = R_phys
        self.T_phys = T_phys
        self.device = device if device else('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype

        # Attributes to store the state of the dataset
        self.data = None
        self.iteration = 0

        self.resample()

    def resample(self):
        if self.sampling_type == 'lhc':
            dataset = self.generator.create_norm_lhc(n_points=self.n_points, seed_offset=self.iteration)
        elif self.sampling_type == 'sobol':
            dataset = self.generator.create_norm_sobol(n_points=self.n_points, seed_offset=self.iteration)
        elif self.sampling_type == 'grid':
            dataset = self.generator.create_norm_grid(n_r=int(np.sqrt(self.n_points)), n_t=int(np.sqrt(self.n_points)))
        elif self.sampling_type == 'initial':
            dataset = self.generator.create_norm_initial_condition(n_points=self.n_points)
        elif self.sampling_type == 'center':
            dataset = self.generator.create_norm_center_boundary(n_points=self.n_points)
        elif self.sampling_type == 'surface':
            dataset = self.generator.create_norm_surface_boundary(n_points=self.n_points)
        elif self.sampling_type == 'sensor':
            dataset = self.generator.create_sensor_dataset(sensor_file=self.generator.sensor_file) if self.generator.sensor_file is not None else None
        else:
            raise ValueError(f'acceptable sumpling type: lhc, sobol, grid, initial, center, surface')

        # Change the type to torch.tensor, send to device and dtype and save the state to the .r, .t attributes
        self.data = torch.tensor(dataset, device=self.device, dtype=self.dtype)
        self.iteration += 1

    def resample_adaptive(self, approximator, pde_loss_fn, n_canditates=50000, sample_size=200, max_points=4000, accumulate=False, k=3.0, c=0.0):
        raw_canditates = self.generator.create_norm_lhc(n_canditates, seed_offset=self.iteration + 100)
        raw_canditates = torch.tensor(raw_canditates, dtype=self.dtype, device=self.device)

        res_f, res_b, res_i = pde_loss_fn(approximator=approximator, data=raw_canditates, pointwise=True)
        error_map = torch.abs(res_f) + torch.abs(res_b) + torch.abs(res_i)
        error_map = error_map.flatten().detach().cpu().numpy()
        error = error_map**k / np.mean(error_map) + c
        error_norm = ((error + 1e-6) / np.sum(error + 1e-6)).flatten()
        if accumulate:
            if self.n_points < max_points: 
                indices = np.random.choice(a=n_canditates, size=sample_size, replace=False, p=error_norm)
                new_data = torch.tensor(raw_canditates[indices], dtype=self.dtype, device=self.device)
                self.data = torch.cat((self.data, new_data), dim=0)
                self.n_points = self.data.shape[0]
        else:
            indices = np.random.choice(a=n_canditates, size=self.n_points, replace=False, p=error_norm)
            new_data = torch.tensor(raw_canditates[indices], dtype=self.dtype, device=self.device)
            self.data = new_data

        self.iteration += 1
        print(f'{self.name} dataset adaptive refinement: total points: {self.n_points}')


    @property
    def r(self) -> torch.tensor:
        """return: the first column (radius) of data tensor"""
        return self.data[:, 0:1]
    
    @property
    def t(self) -> torch.tensor:
        """return: the second column (time) of data tensor"""
        return self.data[:, 1:2]
    
    def get_nondimentional_data(self) -> torch.tensor:
        ''' return: torch.tensor(n_points * 2)'''
        return self.data
    
    def get_physically_scaled_data(self) -> torch.tensor:
        ''' return: torch.tensor(n_points * 2)'''
        return torch.cat((self.r * self.R_phys, self.t * self.T_phys), dim=1)

    def plotme(self, marker_size=0.2):
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(self.r.cpu(), self.t.cpu(), s=marker_size)
        plt.title(f'{self.name} (normalized)')
        plt.xlabel('r')
        plt.ylabel('t')
        plt.show()

    def plotme_physically_scaled(self):
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(self.r * self.R_phys, self.t * self.T_phys, s=0.2)
        plt.title(f'{self.name} (physically scaled)')
        plt.xlabel('r (μm)')
        plt.ylabel('t (h)')
        plt.show()

    def __add__(self, other):
        '''Overides the + operator to concatenate PINNDataset object verticly'''
        if not isinstance(other, PINNDataset):
            raise TypeError(f'unsupported operand type(s) for +: {type(self)} and {type(other)}')
        if self.R_phys != other.R_phys or self.T_phys != other.T_phys:
            raise ValueError(f'Cannot add datasets with different physical scales!')
        new = PINNDataset(name=f'{self.name}+{other.name}',generator=self.generator,sampling_type=self.sampling_type, R_phys=self.R_phys, T_phys=self.T_phys, device=self.device, dtype=self.dtype)
        r = torch.cat([self.r, other.r], dim=0)
        t = torch.cat([self.t, other.t], dim=0)
        new.data = torch.cat((r, t), dim=1)
        return new
    
    def __len__(self):
        return self.r.shape[0]