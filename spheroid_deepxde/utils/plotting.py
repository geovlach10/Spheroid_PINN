import matplotlib.pyplot as plt
import deepxde as dde

def scatter_collocation_points(data: dde.data.TimePDE):
    x = data.train_x_all
    x_coor = x[:, 0:1]
    t_coor = x[:, 1:]
    plt.figure(figsize=(8, 8))
    plt.scatter(x_coor, t_coor, s=0.1)
    plt.xlabel('Space (x)')
    plt.ylabel('Time (t)')
    plt.title(f'Collocation points (Total: {len(x)}) - distribution: {data.train_distribution}\nidxs on the data.train_x_bc{data.num_bcs}')
    plt.grid(True, linestyle='--', alpha=0.35)
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
def plot_2d_liposome(model, time_list, C_scale=60, r_scale=200, t_scale=24, save_path=None, lambda_phi=None):
    x = np.linspace(0, 1, 100)
    phi = lambda_phi(x)
    plt.figure(figsize=(8, 6))
    for t in time_list:
        t_array = np.full_like(x, t)
        xt = np.stack((x, t_array), axis=1)
        u = model.predict(xt)
        y = u * phi.reshape(-1, 1) * C_scale
        plt.scatter(xt[:, 0:1] * r_scale, y, s=10, alpha=0.5, label=f't={t*t_scale}h')
    plt.plot(x * r_scale, phi * r_scale, 'k--', label='Porosity Profile')
    plt.xlabel('Radius')
    plt.ylabel('Concentration % of Csol')
    plt.title('LIPOSOME PROFILES\n(non-binding non-reacting) - center/surface hard constrained')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig(f'{save_path}/non_binding_non_reacting_liposomes_profiles.png', dpi=300) if save_path is not None else None
    plt.show()
    
def plot_2d_antibody(model, time_list, C_scale=60, r_scale=200, t_scale=24, save_path=None, lambda_phi=None):
    x = np.linspace(0, 1, 100)
    phi = lambda_phi(x)
    plt.figure(figsize=(8, 6))
    for t in time_list:
        t_array = np.full_like(x, t)
        xt = np.stack((x, t_array), axis=1)
        u = model.predict(xt)
        y = u[:, 0:1] * phi.reshape(-1, 1) * C_scale
        plt.scatter(xt[:, 0:1] * r_scale, y, s=10, alpha=0.5, label=f't={t*t_scale}h')
    # plt.plot(x * r_scale, phi * r_scale, 'k--', label='Porosity Profile')
    plt.xlabel('Radius')
    plt.ylabel('Concentration % of Csol')
    plt.title('ANTIBODY PROFILES\ncenter/surface hard constrained')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig(f'{save_path}/non_binding_non_reacting_antibody_profiles.png', dpi=300) if save_path is not None else None
    plt.show()