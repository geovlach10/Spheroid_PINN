import torch
import torch.nn as nn
import deepxde as dde

class ModifiedPINN(nn.Module):
    def __init__(self, in_dim=2, out_dim=3, hidden_dim=256, num_layers=4):
        super().__init__()
        
        self.activation = torch.tanh
        
        # --- 1. DUAL ENCODER NETWORKS (U and V) ---
        # These process the raw input features to dynamically scale hidden states
        self.encoder_U = nn.Linear(in_dim, hidden_dim)
        self.encoder_V = nn.Linear(in_dim, hidden_dim)
        
        # --- 2. HIDDEN LAYERS ---
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        
        # --- 3. OUTPUT LAYER ---
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        
        # Rigorous Xavier/Glorot Normal Initialization for PINN stability
        for layer in [self.encoder_U, self.encoder_V, self.output_layer] + list(self.hidden_layers):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            
        # --- 4. DEEPXDE HOOK PLACEHOLDERS ---
        self._input_transform = None
        self._output_transform = None
        self.regularizer = None

    def forward(self, inputs):
        x = inputs
        
        # A. Feature Transform Step: Applies center symmetry r -> r^2
        if self._input_transform is not None:
            x = self._input_transform(x)
            
        # B. Compute the Multiplicative Scaling Vectors
        U = self.activation(self.encoder_U(x))
        V = self.activation(self.encoder_V(x))
        
        # C. Initialize Hidden State with the First Layer Pass
        h = U  
        
        # D. Execute the Modified MLP Forward Track
        for layer in self.hidden_layers:
            hidden_activation = self.activation(layer(h))
            # Element-wise scaling highway that prevents vanishing/stiff gradients
            h = (1.0 - hidden_activation) * U + hidden_activation * V
            
        # E. Map Hidden Features to your 3 physical outputs
        outputs = self.output_layer(h)
        
        # F. Output Transform Step: Enforces the clean initial zero boundary layer (t * y)
        if self._output_transform is not None:
            outputs = self._output_transform(inputs, outputs)
            
        return outputs

    # DeepXDE Compatibility Hooks
    def apply_feature_transform(self, transform):
        self._input_transform = transform

    def apply_output_transform(self, transform):
        self._output_transform = transform