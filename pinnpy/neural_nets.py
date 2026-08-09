"""PINN network backbones for the Trastuzumab spheroid model.

Defines the weight-factorization building block and the concrete
feedforward architectures that consume it:

    RWFLinear         -- weight-factorized nn.Linear replacement (eq. 4.4-4.5)
    MLP               -- abstract backbone: owns everything shared between
                         concrete architectures (construction bookkeeping,
                         layer factory, weight init, transformation hooks,
                         hard-IC output convention)
    FCNN              -- standard feedforward backbone
    ModifiedMLP       -- gated-encoder backbone (eq. 6.7-6.11)

Both concrete backbones are drop-in composable: an MLP subclass takes
an optional `input_transformation` (e.g. a FourierFeatures instance, see
pinnpy.embeddings) applied once, upstream of the backbone's own
layers, and an optional `use_rwf` flag that swaps every internal
nn.Linear for an RWFLinear via the shared `_make_layer` factory.

`MLP` is the type the rest of the codebase should depend on --
`Pinn.net`, `Trainer`, checkpoint (de)serialization -- rather than any
concrete subclass. Adding a new backbone architecture means subclassing
MLP and implementing `_build_layers` / `_compute_hidden`; nothing
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

class MLP(nn.Module, ABC):

    """Abstract base class for PINN network backbones (FCNN, ModifiedMLP, ...).

    This is the type the rest of the codebase (BasePinn, Trainer,
    ConstrainedNet, checkpointing) should depend on -- e.g. `net: MLP`
    -- rather than any concrete subclass. `isinstance(net, MLP)` works
    directly, since this is real inheritance.

    Owns every part of a PINN backbone that doesn't vary between concrete
    subclasses: construction bookkeeping, the RWF-vs-plain nn.Linear layer
    factory, weight initialization, the input/output transformation hooks,
    and the hard-IC output convention (architecturally constrained:
    C(r, 0) = 0 for every t = 0, via t * u).

    Attributes (set in __init__, all subclasses inherit these as-is):
        seed: RNG seed used for both torch.manual_seed and reproducibility
            bookkeeping (e.g. checkpoint metadata).
        input_dim: raw coordinate dimensionality (2, for (r, t)).
        output_dim: raw prediction dimensionality (3, for (c0, c1, c2)).
        n_layers, n_neurons: architecture size, as passed at construction.
        use_rwf, rwf_mu, rwf_sigma: RWF configuration, consumed by
            _make_layer.
        activation: nn.Tanh(), shared by every subclass's hidden layers.
        input_transformation, output_transformation: optional callables
            applied before/after the subclass-specific hidden computation.
        first_layer_in: input width of the first layer -- input_dim, or
            input_transformation.output_dim if an input_transformation
            with that property was supplied.
    """

    _INITS = {'xavier_normal': nn.init.xavier_normal_, 'xavier_uniform': nn.init.xavier_uniform_}
        
    def __init__(self, in_dim: int, out_dim: int, n_layers: int, n_neurons: int, activation_instance: nn.Module = nn.Tanh(), initialization: str = '', input_transformation: Optional[Callable[[torch.Tensor], torch.Tensor]]=None, output_transformation: Optional[Callable[[torch.Tensor], torch.Tensor]]=None,
                    use_rwf: bool = False, rwf_mu: float = 1.0, rwf_sigma: float = 0.1, seed: int = 42):
        """
        Args:
            n_layers: total layer count as counted by the concrete subclass
                (for FCNN: 1 first-hidden + (n_layers-2) mid-hidden + 1
                output; ModifiedMLP counts its main stack the same way,
                excluding the two encoders and the output layer).
            n_neurons: hidden width, uniform across all hidden layers.
            initialization: 'xavier_normal' or 'xavier_uniform'; passed
                through to _initialize_weights, and to RWFLinear's own V
                init if use_rwf=True. Empty string / falsy -> no init
                (weights keep PyTorch's default init).
            input_transformation: optional callable (e.g. FourierFeatures)
                applied to the concatenated (r, t) input before it reaches
                the backbone. If it exposes an `output_dim` attribute,
                that's used to size the first layer; otherwise input_dim
                (2) is assumed.
            output_transformation: optional callable applied to the raw
                network output before the hard-IC t-multiply.
            use_rwf, rwf_mu, rwf_sigma: see RWFLinear. When use_rwf=True,
                every layer built via _make_layer is an RWFLinear instead
                of an nn.Linear.
            seed: seeds torch.manual_seed for reproducible weight init,
                and is stored for checkpoint bookkeeping.
        """
        super().__init__()

        # Reproducibility
        self.seed = seed
        torch.manual_seed(self.seed)

        # Network attributes
        self.input_dim = in_dim
        self.output_dim = out_dim
        self.n_layers = n_layers
        self.n_neurons = n_neurons

        # RWF
        self.use_rwf = use_rwf
        self.rwf_mu = rwf_mu
        self.rwf_sigma = rwf_sigma

        # Activation fn
        self.activation_fn = activation_instance
        self.initialization = initialization
        self.input_transformation = input_transformation
        self.output_transformation = output_transformation
        self.first_layer_in = getattr(input_transformation, 'output_dim', self.input_dim)

        self._build_layers(n_layers, n_neurons)
        self._initialize_weights(self.initialization)

    def _make_layer(self, in_f, out_f) -> nn.Module:

        """RWF-vs-plain nn.Linear switch, shared by every subclass so they
        can't drift out of sync on how a layer gets constructed.

        Args:
            `in_f`, 
            `out_f`: layer input/output width.

        Returns:
            `RWFLinear`(in_f, out_f, ...) if self.use_rwf else nn.Linear(in_f, out_f).
        """

        if self.use_rwf:
            return RWFLinear(in_f, out_f, mu=self.rwf_mu, sigma=self.rwf_sigma, initialization=self.initialization)
        return nn.Linear(in_f, out_f)

    @abstractmethod
    def _build_layers(self, n_layers: int, n_neurons: int) -> None:

        """Register whatever modules this backbone needs (self.layers,
        encoders, etc), via self._make_layer. Called once, from __init__,
        before weight init -- so every parameter this method registers
        gets picked up by _initialize_weights's self.modules() walk.

        Args:
            n_layers, n_neurons: as passed to __init__.
        """
        ...


    @abstractmethod
    def _compute_hidden(self, x: torch.Tensor) -> torch.Tensor:

        """(embedded) input -> raw network output, pre output_transformation.
        This is the one method that actually varies between backbones --
        everything else in forward() is identical across subclasses.

        Args:
            x: (N, first_layer_in) -- the (possibly input_transformation-
                embedded) coordinate batch.

        Returns:
            (N, output_dim) raw prediction, before output_transformation
            and before the hard-IC t-multiply.
        """
        ...


    def forward(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        """Full forward pass, identical across all MLP subclasses.

        Args:
            r: (N, 1) radial coordinate.
            t: (N, 1) time coordinate.

        Returns:
            (N, output_dim) prediction -- (c0, c1, c2) columns for this
            project -- with C(r, 0) = 0 enforced architecturally for every
            t = 0 (final t * u multiply).
        """

        x = torch.cat([r, t], dim=1)

        # 1. input transformation (e.g. Fourier features).
        if self.input_transformation is not None:
            x = self.input_transformation(x)

        # 2. backbone-specific hidden computation.
        u = self._compute_hidden(x)

        # 3. output transformation.
        if self.output_transformation is not None:
            u = self.output_transformation(u)

        return t * u        # architecturally constrained to produce C(r, 0) = 0 for every t = 0.


    def _initialize_weights(self, initialization: str | None) -> None:
        """Applies `initialization` to every nn.Linear.

        Args:
            initialization: 'xavier_normal', 'xavier_uniform', or None/''
                (no-op -- weights keep PyTorch's default init).
        """
        if initialization == '':    return
        if initialization is not None:
            for layer in self.modules():
                if isinstance(layer, nn.Linear):
                    self._INITS[initialization](layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
            print(f'neural network weights initialized succesfully - initializer: {initialization}')
        else:
            print('weight have not been initialized.')


class FCNN(MLP):

    """Standard feedforward neural nettwork.

    n_layers total: 1 first-hidden layer (input -> n_neurons), (n_layers-2)
    mid-hidden layers (n_neurons -> n_neurons), 1 output layer
    (n_neurons -> output_dim, no activation).
    """

    def _build_layers(self, n_layers: int, n_neurons: int) -> None:
        """Builds self.layers: [first-hidden, *mid-hidden, output]."""
        self.layers = nn.ModuleList()
        self.layers.append(self._make_layer(self.first_layer_in, n_neurons))
        for _ in range(n_layers - 2):
            self.layers.append(self._make_layer(n_neurons, n_neurons))
        self.layers.append(self._make_layer(n_neurons, self.output_dim))

    def _compute_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Feeds x through every hidden layer (activation applied), then
        the output layer (no activation)."""
        for layer in self.layers[:-1]:
            x = self.activation_fn(layer(x))
        return self.layers[-1](x)     

class ModifiedMLP(MLP):

    """Modified MLP w/ gated encoder fusion, per Wang, Sankaran, Wang &
    Perdikaris (2023), Sec. 6.4, eqs. (6.7)-(6.11).

    Two encoders U, V are computed once from the (embedded) input and
    re-injected at every hidden layer via a learned convex gate:

        U = sigma(W1 x + b1),  V = sigma(W2 x + b2)                  (6.7)
        f^(l) = W^(l) g^(l-1) + b^(l),      g^(0)(x) = x              (6.8)
        g^(l) = sigma(f^(l)) * U + (1 - sigma(f^(l))) * V            (6.9)
        f_theta(x) = W^(L+1) g^(L) + b^(L+1)                         (6.10)

    In practice, demands more compute than FCNN but tends to lower PDE
    residuals / yield more accurate results (paper, Sec 6.4).

    n_layers total: counted the same way as FCNN for the main stack (1
    first-hidden + (n_layers-2) mid-hidden), EXCLUDING encoder_U/encoder_V
    and the final output_layer, which are separate, always-present modules.
    """

    def _build_layers(self, n_layers: int, n_neurons: int) -> None:

        """Builds encoder_U, encoder_V (eq. 6.7), self.layers (the f^(l)
        stack, eq. 6.8), and output_layer (eq. 6.10)."""

        self.encoder_U = self._make_layer(self.first_layer_in, n_neurons)
        self.encoder_V = self._make_layer(self.first_layer_in, n_neurons)

        self.layers = nn.ModuleList()
        self.layers.append(self._make_layer(self.first_layer_in, n_neurons))
        for _ in range(n_layers - 2):
            self.layers.append(self._make_layer(n_neurons, n_neurons))

        self.output_layer = self._make_layer(n_neurons, self.output_dim)

    def _compute_hidden(self, x: torch.Tensor) -> torch.Tensor:

        """Computes U, V (eq. 6.7), then runs the gated recursion of eq.
        6.8-6.9 through self.layers, then the output layer (eq. 6.10)."""

        U = self.activation_fn(self.encoder_U(x))
        V = self.activation_fn(self.encoder_V(x))

        g = x
        for layer in self.layers:
            gate = self.activation_fn(layer(g))
            g = gate * U + (1.0 - gate) * V

        return self.output_layer(g)


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
    

