from __future__ import annotations
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    


class FCNN(nn.Module):

    _INITS = {'xavier_normal': nn.init.xavier_normal_, 'xavier_uniform': nn.init.xavier_uniform_}
    
    def __init__(self, n_layers, n_neurons, initialization: str | None=None, input_transformation: Optional[Callable[[torch.Tensor], torch.Tensor]]=None, output_transformation: Optional[Callable[[torch.Tensor], torch.Tensor]]=None, seed: int = 42):
        """initaialization: 'xavier_normal' or 'xavier_uniform'"""
        super().__init__()

        # Reproducibility
        self.seed = seed
        torch.manual_seed(self.seed)

        # Network attributes
        self.input_dim = 2
        self.output_dim = 3
        self.n_layers = n_layers
        self.n_neurons = n_neurons
    
        # Activation
        self.activation = nn.Tanh()

        self.input_transformation = input_transformation
        first_layer_in = getattr(input_transformation, 'output_dim', self.input_dim)

        # Layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(first_layer_in, n_neurons))
        for _ in range(n_layers - 2):
            self.layers.append(nn.Linear(n_neurons, n_neurons))
        self.layers.append(nn.Linear(n_neurons, self.output_dim))

        # Weight initializer
        self.initialization = initialization
        self._initialize_weights(self.initialization)
        
        
        self.output_transformation = output_transformation

    def forward(self, r: torch.Tensor, t:torch.Tensor) -> torch.Tensor:
        x = torch.cat([r, t], dim=1)

        # 1. input transformation.
        if self.input_transformation is not None:
            x = self.input_transformation(x)

        # 2. Feedforward through all hidden layers except the output layer
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        # 3.Output layer (No activation).
        u = self.layers[-1](x)

        # 4. Output transformation.
        if self.output_transformation is not None:
            u = self.output_transformation(u)
        
        return t * u        # architecturaly constrained to produce C(r, 0) = 0 for every t = 0.
    
    def _initialize_weights(self, initialization: str | None):
        if initialization is not None:
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    self._INITS[initialization](layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
            print(f'neural network weights initialized succesfully - initializer: {initialization}')
        else:
            print('weight have not been initialized.')
        


