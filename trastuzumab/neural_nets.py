"""PINN network backbones for the Trastuzumab spheroid model.

Defines the input-embedding / weight-factorization building blocks and the
concrete feedforward architectures that consume them:

    FourierFeatures   -- input embedding (eq. 4.3, Wang et al. 2023)
    RWFLinear         -- weight-factorized nn.Linear replacement (eq. 4.4-4.5)
    BaseMLP           -- abstract backbone: owns everything shared between
                         concrete architectures (construction bookkeeping,
                         layer factory, weight init, transformation hooks,
                         hard-IC output convention)
    MLP               -- standard feedforward backbone (formerly FCNN)
    ModifiedMLP       -- gated-encoder backbone (eq. 6.7-6.11)

All three concrete/embedding classes are drop-in composable: a BaseMLP
subclass takes an optional `input_transformation` (e.g. a FourierFeatures
instance) applied once, upstream of the backbone's own layers, and an
optional `use_rwf` flag that swaps every internal nn.Linear for an
RWFLinear via the shared `_make_layer` factory.

`BaseMLP` is the type the rest of the codebase should depend on --
`BasePinn.net`, `Trainer`, checkpoint (de)serialization -- rather than any
concrete subclass. Adding a new backbone architecture means subclassing
BaseMLP and implementing `_build_layers` / `_compute_hidden`; nothing
elsewhere in the codebase needs to change (open/closed).

Reference: Wang, Sankaran, Wang & Perdikaris (2023), "An Expert's Guide to
Training Physics-Informed Neural Networks."
"""


from __future__ import annotations
from abc import ABC, abstractmethod
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

class RWFLinear(nn.Module):

    """Random Weight Factorization (RWF) linear layer, per Wang, Sankaran,
    Wang & Perdikaris (2023), Sec. 4.3, eq. (4.4)-(4.5) / Algorithm 2:
 
        W = diag(exp(s)) @ V,   s ~ N(mu, sigma^2)  (per-output-neuron scale)
 
    A drop-in replacement for nn.Linear. V is initialized with a standard
    scheme (e.g. Glorot) exactly as an nn.Linear.weight would be; s is then
    sampled from N(mu, sigma^2) and gradient descent is applied directly to
    the factorized parameters (s, V), never to a materialized W.
 
    Why this helps (paper Theorem B.2): under this factorization, the
    effective gradient step on the *composed* weight w = s*v gets rescaled
    by (s^2 + ||v||_2^2) per neuron. Since s and v are themselves trainable,
    this amounts to a self-adaptive, per-neuron learning rate -- neurons
    that need to move further in weight-space can effectively acquire a
    larger step size on their own, without a global LR schedule having to
    guess that for them.
 
    Forward: y = exp(s) * (x @ V^T) + b   (exp(s) broadcasts over the output dim)
 
    Args:
        in_features, out_features: same meaning as nn.Linear.
        mu, sigma: mean/std of the initial scale-factor distribution
            s ~ N(mu, sigma^2). Paper recommends mu in {0.5, 1.0}, sigma = 0.1;
            too small mu/sigma converges to plain-MLP behavior, too large
            destabilizes training (Sec. 4.3).
        initialization: 'xavier_normal' or 'xavier_uniform', applied to V
            at construction -- matches FCNN's existing init scheme, so RWF
            layers and plain nn.Linear layers stay initialized consistently
            when mixed in the same network.
        bias: whether to include a bias term (default True, as in nn.Linear).
    """
     
    _INITS = {'xavier_normal': nn.init.xavier_normal_, 'xavier_uniform': nn.init.xavier_uniform_} # acceptable initialization schemes.

    def __init__(self, in_features: int, out_features: int, mu: float = 1.0, sigma: float = 0.1, initialization: str = '', bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu = mu
        self.sigma = sigma
        self.initialization = initialization
        # ---
        self.V = nn.Parameter(self._get_initialized_V)
        self.s = nn.Parameter(mu + sigma * torch.randn(out_features))
        self.bias = nn.Parameter(torch.zeros([out_features])) if bias else None
        # ---
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.exp(self.s) * F.linear(x, self.V)
        if self.bias is not None:
            y = y + self.bias
        return y
    
    @property
    def _get_initialized_V(self) -> torch.Tensor:
        V = torch.empty([self.out_features, self.in_features])
        init_fn = self._INITS.get(self.initialization, nn.init.xavier_normal_)
        init_fn(V)
        return V
    
    @property
    def weight(self) -> torch.Tensor:
        """Materializes W = diag(exp(s)) @ V for inspection/debugging.
        Not used in forward() -- the factorized form is what's actually
        trained, per the derivation above."""
        return torch.exp(self.s).unsqueeze(1) * self.V
    
    def __repr__(self) -> str:
        return f'RWFLinear(in_features={self.in_features}, out_features={self.out_features}, mu={self.mu}, sigma={self.sigma})'
    

class FCNN(nn.Module):

    _INITS = {'xavier_normal': nn.init.xavier_normal_, 'xavier_uniform': nn.init.xavier_uniform_}
    
    def __init__(self, n_layers, n_neurons, initialization: str = '', input_transformation: Optional[Callable[[torch.Tensor], torch.Tensor]]=None, output_transformation: Optional[Callable[[torch.Tensor], torch.Tensor]]=None,
                 use_rwf: bool = False, rwf_mu: float = 1.0, rwf_sigma: float = 0.1, seed: int = 42):
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
        # ---
        self.use_rwf = use_rwf
        self.rwf_mu = rwf_mu
        self.rwf_sigma = rwf_sigma

        # Activation
        self.activation = nn.Tanh()

        self.input_transformation = input_transformation
        first_layer_in = getattr(input_transformation, 'output_dim', self.input_dim)

        def make_layer(in_f: int, out_f: int) -> nn.Module:
            if use_rwf:     
                return RWFLinear(in_f, out_f, mu=rwf_mu, sigma=rwf_sigma, initialization=initialization)
            else: 
                return nn.Linear(in_f, out_f)
            
        # Layers
        self.layers = nn.ModuleList()
        self.layers.append(make_layer(first_layer_in, n_neurons))
        for _ in range(n_layers - 2):
            self.layers.append(make_layer(n_neurons, n_neurons))
        self.layers.append(make_layer(n_neurons, self.output_dim))

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