import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import torch

class Dataset:
    def __init__(self, data, device='cpu', dtype=torch.float32, name: str=''):
        self.name = name
        self.n_points = 0
        self.data = torch.tensor(data, dtype=dtype, device=device) if not isinstance(data, torch.Tensor) else data

    @property
    def r(self) -> torch.Tensor:
        """return: the first column (radius) of data tensor"""
        return self.data[:, 0:1]
    
    @property
    def t(self) -> torch.Tensor:
        """return: the second column (time) of data tensor"""
        return self.data[:, 1:2]

    def plotme(self, marker_size=0.2):
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(self.r.cpu(), self.t.cpu(), s=marker_size)
        plt.xlabel('r')
        plt.ylabel('t')
        plt.title(f'(number of points: {len(self.data)})')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    def __add__(self, other):
        '''Overides the + operator to concatenate Dataset object verticly'''
        if not isinstance(other, Dataset):
            raise TypeError(f'unsupported operand type(s) for +: {type(self)} and {type(other)}')
        new_data = torch.cat([self.data, other.data], dim=0)
        return Dataset(data=new_data)
        
    def __len__(self):
        return self.r.shape[0]
    
class DatasetSampler:
    '''
    This class contains the methods to create the training datasets for the PDE domain, the center boundary(r=0), 
    the surface boundrary(r=1) and the initial condition(t=0).
    returns:
        Instance of Dataset class.
    '''
    def __init__(self, seed=42, device='cpu', dtype=torch.float32):
        self.seed = seed
        self.device = device
        self.dtype = dtype

    def sample_collocation_points(self, n_points, l_bounds=[0, 0], u_bounds=[1, 1], seed_offset=0):
        sampler = qmc.LatinHypercube(d=2, seed=self.seed + seed_offset)
        dataset = sampler.random(n=n_points)
        dataset = qmc.scale(sample=dataset, l_bounds=l_bounds, u_bounds=u_bounds)
        return Dataset(data=dataset, name='collocation')

    def sample_initial_points(self, n_points, l_bounds=[0, 0], u_bounds=[1, 1]):
        r = np.linspace(0, 1, n_points).reshape(-1, 1) * (u_bounds[0] - l_bounds[0]) + l_bounds[0] 
        t = np.zeros_like(r) + l_bounds[1]
        dataset = np.hstack([r, t])
        return Dataset(data=dataset, name='initial')

    def sample_center_points(self, n_points, l_bounds=[0, 0], u_bounds=[1, 1]):
        t = np.linspace(0, 1, n_points).reshape(-1, 1) * (u_bounds[1] - l_bounds[1]) + l_bounds[1]
        r = np.zeros_like(t) + l_bounds[0]
        dataset = np.hstack([r, t])
        return Dataset(data=dataset, name='center')

    def sample_surface_points(self, n_points, l_bounds=[0, 0], u_bounds=[1, 1]):
        t = np.linspace(0, 1, n_points).reshape(-1, 1) * (u_bounds[1] - l_bounds[1]) + l_bounds[1]
        r = np.ones_like(t) * u_bounds[0]
        dataset = np.hstack([r, t])
        return Dataset(data=dataset, name='surface')