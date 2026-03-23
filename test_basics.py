import torch
import matplotlib.pyplot as plt
from core.data import DataFactory, PINNDataset

factory = DataFactory(seed=42)
pde_dataset = PINNDataset(batch_size=100, R_phys=200, T_phys=50, sampling_type='lhc', dtype=torch.float32, generator=factory, device=None, r_norm=1, t_norm=1, name='dataset')
bc_dataset = PINNDataset(batch_size=20, sampling_type='surface', generator=factory, device=None, r_norm=1, t_norm=1, name='boundary_dataset')
dataset = pde_dataset + bc_dataset

dataset.plotme()
print(torch.cuda.is_available())