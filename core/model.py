import torch
import torch.nn as nn
import torch.nn.functional as F

class PINN(nn.Module):
    ''' returns: Tuple of 3 column tensors of size: (num_batches * 1)'''
    def __init__(self, n_layers, n_neurons, seed=42):
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
        self._initialize_weights()

    def forward(self, x, t):
        u = torch.cat((x, t), dim=1)
        for i in range(len(self.layers) - 1):
             u = self.activation(self.layers[i](u))
        u = F.softplus(self.layers[-1](u)) # ensure positivity of the output
        return u[:, 0].view(-1, 1), u[:, 1].view(-1, 1), u[:, 2].view(-1, 1)
    
    def show_if_requires_grad(self):
        print('\ninspect which parameters being recorded in the graph')
        for name, param in self.named_parameters():
            print(f'{name}, requires_grad: {param.requires_grad}')
    
    def freeze_coeffitients_params(self):
        for name, param in self.named_parameters():
            if name.startswith('coefficients'):
                param.requires_grad = False

    def unfreeze_coeffitients_params(self):
        for name, param in self.named_parameters():
            if name.startswith('coefficients'):
                param.requires_grad = True

    def freeze_dynweight_params(self):
        for name, param in self.named_parameters():
            if name.startswith('dynamic_weights'):
                param.requires_grad = False

    def unfreeze_dynweight_params(self):
        for name, param in self.named_parameters():
            if name.startswith('dynamic_weights'):
                param.requires_grad = True

    def _initialize_weights(self):
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        print('neural network weights initialized succesfully')

    def param_optimization_bool(self):
        for name, param in self.named_parameters():
            if 'layer' not in name:
                 print(f'{name}, optimizability: {param.requires_grad}')

    def show_info(self):
        ''' print weights and biases for every layer ---> first 3 elements only'''
        num_of_dashes = 30
        print('\n', num_of_dashes*'-', 'APPROXIMATOR SUMMARY',  num_of_dashes*'-', '\n\n')
        print(num_of_dashes*'-', 'LEYERS', num_of_dashes * '-', '\n', self)
        print('\n', num_of_dashes*'-', 'PARAMETERS', num_of_dashes*'-', '\n')
        for name, param in (self.named_parameters()):
            if param.dim() == 0:
                print(f'parameter: {name}\n{param.shape}\n{param}\n')
            elif param.dim() == 1:
                print(f'parameter: {name}\n{param.shape}\n{param[:3]}            ...\n')
            elif param.dim() == 2:
                print(f'parameter: {name}\n{param.shape}\n{param[:3, :3]}\n                    ...\n')