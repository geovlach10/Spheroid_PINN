import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import os

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'notebook'

import numpy as np
import torch

import cv2
import glob
import os
import re
import shutil



class InteractiveVisualizer():
    def __init__(self, context, r_size, t_size):
        self.ctx = context

        self.r_size = r_size
        self.t_size = t_size

        self.r, self.t = np.linspace(0, 1, self.r_size), np.linspace(0, 1, self.t_size)
        self.R, self.T = np.meshgrid(self.r, self.t)

    def create_3d_uptake_plot(self, what, model, device):
        r_grid = torch.tensor(self.R, dtype=torch.float32, device=device).reshape(-1, 1)
        t_grid = torch.tensor(self.T, dtype=torch.float32, device=device).reshape(-1, 1)
        print('r, t (1D): ',r_grid.shape, t_grid.shape)
        with torch.no_grad():
            cf, cb, ci = model(r_grid, t_grid)
        z = cf.reshape(self.t_size, self.r_size).cpu()    
        print(z.shape)

        fig = go.Figure(
            data=[
                go.Surface(
                    z=z,
                    x=r_grid,
                    y=t_grid,
                    colorscale='Viridis',
                    colorbar=dict(
                        title=dict(
                            text=f'Concentration of {what}',
                            side='top'
                        ),
                        thickness=15,
                        len=0.5
                    ),
                    cmin=0,
                    cmax=1.1
                )
            ]
        )

        fig.update_layout(
            title='Antibody uptake profile',
            scene=dict(
                xaxis_title='Radius (r)',
                yaxis_title='Time (t)',
                zaxis_title=f'Concentration of {what}',
                zaxis=dict(range=[0, 1.1])
            ),
            width=800,
            height=700,
            margin=dict(l=65, r=50, b=65, t=90)
        )

        fig.show(renderer='notebook')

class Static3DVisualizer():
    def __init__(self, r_size, t_size, t_start=0, t_end=1, save_dir='results/frames'):
        self.save_dir = save_dir
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)

        self.r_size = r_size
        self.t_size = t_size
        
        self.t_start = t_start
        self.t_end = t_end

        self.R, self.T = np.meshgrid(np.linspace(0, 1, self.r_size), np.linspace(self.t_start, self.t_end, self.t_size))

        self.display_handle = display(None, display_id=True)

    def update(self, what: str, update_every, model, device, epoch, phase):
        r_grid = torch.tensor(self.R, dtype=torch.float32, device=device).reshape(-1, 1)
        t_grid = torch.tensor(self.T, dtype=torch.float32, device=device).reshape(-1, 1)
        if epoch % update_every == 0:
            with torch.no_grad():
                cf, cb, ci = model(r_grid, t_grid)
                cf, cb, ci = cf.reshape(self.t_size, self.r_size), cb.reshape(self.t_size, self.r_size), ci.reshape(self.t_size, self.r_size)
            
            Z = cf if what == 'cf' else cb if what=='cb' else ci
            Z = Z.cpu()

            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_wireframe(self.R, self.T, Z, color='royalblue', rstride=4, cstride=4, alpha=0.7)
            ax.set_title(f'Normalized concentration surface [{phase}]\nEpoch {epoch}')
            ax.set_xlabel('Radius (r)')
            ax.set_ylabel('Time (t)')
            ax.set_zlabel(f'Concentration ({what})')

            plt.savefig(os.path.join(self.save_dir, f'{what}_epoch_{epoch}.png'), dpi=100)
            
            self.display_handle.update(fig)
            plt.close(fig)

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    print(f'Folder {folder_path} is ready.')

def cleanup_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            os.remove(os.path.join(folder_path, filename))
        print(f'Cleaned up folder: {folder_path}')
    else:
        os.makedirs(folder_path)
        print(f'Created folder: {folder_path}')
    


def create_pinn_video(image_folder: str, video_folder: str, fps: int = 10):

    image_path = f'{image_folder}/*.png'
    images = glob.glob(image_path)
    if not images:
        print(f'No images found in {image_folder}. Please check the image_path and try again.')
        return
    lambda_key = lambda s: [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
    images.sort(key=lambda_key)

    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    video_path = os.path.join('results', video_folder)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc=fourcc, fps=fps, frameSize=(width, height))

    # Stiching images into video
    print(f'Creating video {video_folder} from images in {image_folder} at {fps} fps...')
    for image in images:
        video.write(cv2.imread(image))

    video.release()
    print(f'Video {video_folder} created successfully!')


def create_interactive_3d_concentration(n_r, n_t, solver):
    r_nodes = np.linspace(0, 1, n_r)
    t_nodes = np.linspace(0, 1, n_t)

    R, T = np.meshgrid(r_nodes, t_nodes)

    numpy_grid = np.concatenate([R.reshape(-1, 1), T.reshape(-1, 1)], axis=1)
    tensor_grid = torch.tensor(numpy_grid, device='mps', dtype=torch.float32)

    with torch.no_grad():
        cf, _, _ = solver.approximator(tensor_grid[:,0:1], tensor_grid[:, 1:2])

    Z = cf.reshape((n_t, n_r)).cpu()

    fig = go.Figure(
        data=[
            go.Surface(
                z=Z,
                x=r_nodes,
                y=t_nodes,
                colorscale='Inferno',
                colorbar={
                    'title': {
                        'text': 'Concentration of [Ab^I]',
                        'side': 'right'
                    }
                }
            )
        ]
    )
    fig.update_layout(
        title=f'Antibody concentration surface [{solver.phase_flag} phase]',
        scene={
            'xaxis_title': 'Radius (r)',
            'yaxis_title': 'Time (t)',
            'zaxis_title': 'Concentration',
        },
        width=700,
        height=600
    )
    fig.show()
    

from ipywidgets import interact, IntSlider

def create_2d_plot(n_r, n_t, model):
   
    # create the node vectors.
    r_nodes = np.linspace(0, 1, n_r)
    t_nodes = np.linspace(0, 1, n_t)

    # Create the mesh from the vectors.
    R, T = np.meshgrid(r_nodes, t_nodes)

    # Create the input tensor of the neural network.
    r_grid = R.flatten().reshape(-1, 1)
    t_grid = T.flatten().reshape(-1, 1)
    grid_points_tensor = torch.tensor(np.concatenate([r_grid, t_grid], axis=1), device='mps', dtype=torch.float32)

    # take the model's prediction
    with torch.no_grad():
        cf, _, _ = model(grid_points_tensor[:, 0:1], grid_points_tensor[:, 1:2])

    # Reshape the prediction to match the shape of R (n_t * n_r)
    prediction = np.array(cf.reshape(R.shape).cpu())

    def plot_at_time(time_idx):
        plt.figure(figsize=(10, 5))
        plt.plot(r_nodes, prediction[time_idx, :], lw=2)

        plt.grid(True, linestyle='--', alpha=0.45)
        plt.title(f'C(r) at t={(t_nodes[time_idx] * 24):.2f}h')
        plt.xlabel('Radius (r)')
        plt.ylabel('Concentration (cf)')
        plt.show()
    

    slider = IntSlider(
        value=0,
        min=0,
        max=len(t_nodes) - 1,
        step=1,
        description='Time Index:',
        layout={'width': '700px'}
    )

    interact(plot_at_time, time_idx=slider)

def create_norm_2d_concentration(what :str, solver, n_r=100, timestamps=[0, 4, 12, 24]):
    ''' timestamps must be a list of times in hours to show the pred. It is divided by tau.'''
    p = solver.ctx.get_pde_parameters()

    plt.figure(figsize=(12, 8))
    r = torch.linspace(0, 1, n_r).view(-1, 1).to((solver.device))

    for time in timestamps:
        time_norm = time / solver.ctx.tau[solver.phase_flag]
        t = torch.full_like(r, time_norm).to(solver.device)
        with torch.no_grad():
            preds = solver.approximator(r, t)
            cf, cb, ci = tuple(p.detach() for p in preds)
            # cf, cb, ci = solver.ctx.C0 * cf, p['r_t'] * cb, p['r_t'] * ci
        
        plt.plot(r.cpu().numpy(), cf.cpu(), label=f't = {time}h') if what == 'cf' else None
        plt.plot(r.cpu().numpy(), cb.cpu(), label=f't = {time}h') if what == 'cb' else None
        plt.plot(r.cpu().numpy(), ci.cpu(), label=f't = {time}h') if what == 'ci' else None

    plt.xlabel('Radius')
    plt.ylabel('Concentration')
    plt.legend()
    plt.title(f'Normalized concentration of {what} during {solver.phase_flag} phase')
    plt.grid(True, alpha=0.3)
    plt.show()