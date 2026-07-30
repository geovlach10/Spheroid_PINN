"""Random Fourier feature embedding for PINN backbones.

Reference: Wang, Sankaran, Wang & Perdikaris (2023), "An Expert's Guide
to Training Physics-Informed Neural Networks," eq. (4.3).
"""

import torch
import torch.nn as nn

class FourierFeatures(nn.Module):

    B: torch.Tensor

    """Random Fourier feature embedding, per Wang, Sankaran, Wang & Perdikaris
    (2023), "An Expert's Guide to Training Physics-Informed Neural Networks",
    eq. (4.3):
 
        gamma(x) = [cos(Bx), sin(Bx)],   B_ij ~ N(0, sigma^2)
        
    Args:
        input_dim: dimensionality of the raw input (2 for (r, t)).
        mapping_size: number of random frequencies. Output dim = 2*mapping_size.
        sigma: std-dev of the sampled frequencies B_ij ~ N(0, sigma^2).
            Larger sigma -> higher frequencies representable -> more
            expressive but harder to optimize / more prone to noise-fitting.
            Paper recommends sigma in [1, 10]; sweep within that range.
        seed: separate from the FCNN seed so you can vary the embedding
            independently of the weight init if you want to.
    """
    def __init__(self, input_dim: int=2, mapping_size: int=64, sigma: float=1.0, seed: int=42):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.sigma = sigma

        generator = torch.Generator().manual_seed(seed)
        B = torch.randn([input_dim, mapping_size], generator=generator) * sigma
        self.register_buffer('B', B)

    @property
    def output_dim(self) -> int:
        return 2 * self.mapping_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.cos(x @ self.B), torch.sin(x @ self.B)], dim=-1)