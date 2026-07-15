from __future__ import annotations
from typing import Callable, Optional

import torch
import torch.nn as nn


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

        # Layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_dim, n_neurons))
        for _ in range(n_layers - 2):
            self.layers.append(nn.Linear(n_neurons, n_neurons))
        self.layers.append(nn.Linear(n_neurons, self.output_dim))

        # Weight initializer
        self.initialization = initialization
        self._initialize_weights(self.initialization)
        
        self.input_transformation = input_transformation
        self.output_transformation = output_transformation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        
        return u
    
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
        


