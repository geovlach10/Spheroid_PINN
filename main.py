import torch
from pinnpy import (
    pinns, 
    neural_nets, 
    trainer as tr, 
    evaluation as ev
)

net = neural_nets.FCNN(in_dim=3, out_dim=64, n_layers=4, n_neurons=64, initialization='xavier_normal', seed=42)
pinn = pinns.ForwardPinn(n_col=2000, n_initial=200, n_center=200, n_surface=200, initial_fn=lambda x: 0.0, u_bounds=(1., 1./24), net=net)
pinn.check_concentration_profile()

optimizer = torch.optim.Adam(pinn.net.parameters(), lr=0.001)

# loss_weights = {'pde0': 1e-4, 'pde1': 1e-4, 'pde2': 1e2, 'surface': 1e6}
loss_weights = {'surface': 1e4}
trainer = tr.Trainer(pinn, loss_weights)
trainer.train(optimizer, epochs=5000, L=0.1)
pinn.check_concentration_profile()