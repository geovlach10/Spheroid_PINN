import torch
import torch.nn as nn

class FCNN(nn.Module):
    ''' returns: Tuple of 3 column tensors of size: (num_batches * 1)'''
    def __init__(self, n_layers, n_neurons, initialization: bool=True, seed=42):
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
        for i in range(n_layers - 2):
            self.layers.append(nn.Linear(n_neurons, n_neurons))
        self.layers.append(nn.Linear(n_neurons, self.output_dim))

        # Weight initializer
        if initialization:
            self._initialize_weights()

    def forward(self, x, t):
        '''x, t: column vectors of shape (n_points * 1)
        return: torch.Tensor of shape (n_points * 3)'''
        u = torch.cat((x, t), dim=1)
        for i in range(len(self.layers) - 1):
            u = self.activation(self.layers[i](u))
        u = self.layers[-1](u)
        u = t * u 
        return u[:, 0].view(-1, 1), u[:, 1].view(-1, 1), u[:, 2].view(-1, 1)

    def _initialize_weights(self):
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        print('neural network weights initialized succesfully')

    def param_optimization_bool(self):
        for name, param in self.named_parameters():
            if 'layer' not in name:
                 print(f'{name}, optimizability: {param.requires_grad}')

