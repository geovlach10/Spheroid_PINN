from turtle import color

import plotly.graph_objects as go
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import os

import numpy as np
import torch

import cv2
import glob
import os
import re

class PINNVisualizer():
    def __init__(self, context, grid_size=50):
        self.grid_size = grid_size
        self.ctx = context

        self.r = np.linspace(0, 1, self.grid_size)
        self.t = np.linspace(0, 1, self.grid_size)
        self.R, self.T = np.meshgrid(self.r, self.t)
        self.grid_points = torch.tensor(np.c_[self.R.ravel(), self.T.ravel()], dtype=torch.float32)

        self.fig = go.FigureWidget(
            data=[go.Surface(
                z=np.zeros_like(self.R),
                x=self.R,
                y=self.T,
                colorscale='Viridis',
                showscale=False
            )]

        )
        self.fig.update_layout(
            title='Live antibody penetration (cf)',
            scene=dict(
                xaxis_title='Radius (r)',
                yaxis_title='Time (t)',
                zaxis_title='Concentration',
                zaxis=dict(range=[0, 1.1])
            ),
            width=600,
            height=500,
            margin=dict(l=0, r=0, b=0, t=40)
        )

    def display(self):
        self.fig.show()

    def update(self, model, device, epoch, update_every=200):
        with torch.no_grad():
            self.grid_points = self.grid_points.to(device)
            cf, _, _ = model(self.grid_points[:, 0:1], self.grid_points[:, 1:2])
            cf_grid = cf.cpu().numpy().reshape(self.R.shape)

        self.fig.data[0].z = cf_grid
        self.fig.update_layout(title=f'Live antibody penetration (cf) - Epoch {epoch}') if epoch % update_every == 0 else None

class Static3DVisualizer():
    def __init__(self, r_size, t_size, save_dir='results/frames'):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.r_size = r_size
        self.t_size = t_size

        self.R, self.T = np.meshgrid(np.linspace(0, 1, self.r_size), np.linspace(0, 1, self.t_size))
        self.display_handle = display(None, display_id=True)

    def update(self, what: str, update_every, model, device, epoch):
        r_grid = torch.tensor(self.R, dtype=torch.float32, device=device).reshape(-1, 1)
        t_grid = torch.tensor(self.T, dtype=torch.float32, device=device).reshape(-1, 1)
        if epoch % update_every == 0:
            with torch.no_grad():
                cf, cb, ci = model(r_grid, t_grid)
                cf, cb, ci = cf.reshape(self.t_size, self.r_size), cb.reshape(self.t_size, self.r_size), ci.reshape(self.t_size, self.r_size)
            
            Z = cf if what == 'cf' else cb if what=='cb' else ci
            Z = Z.cpu()

            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_wireframe(self.R, self.T, Z, color='royalblue', rstride=4, cstride=4, alpha=0.7)
            ax.set_title(f'Live antibody penetration ({what}) - Epoch {epoch}')
            ax.set_xlabel('Radius (r)')
            ax.set_ylabel('Time (t)')
            ax.set_zlabel(f'Concentration ({what})')

            plt.savefig(os.path.join(self.save_dir, f'{what}_epoch_{epoch}.png'), dpi=100)
            
            self.display_handle.update(fig)
            plt.close(fig)

def create_pinn_video(image_folder: str, video_name: str, fps: int = 10):
    path = f'{image_folder}/*.png'
    images = glob.glob(path)
    if not images:
        print(f'No images found in {image_folder}. Please check the path and try again.')
        return
    lambda_key = lambda s: [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
    images.sort(key=lambda_key)

    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f'results/{video_name}.mp4', fourcc=fourcc, fps=fps, frameSize=(width, height))

    print(f'Creating video {video_name}.mp4 from images in {image_folder} at {fps} fps...')
    for image in images:
        video.write(cv2.imread(image))

    video.release()
    print(f'Video {video_name}.mp4 created successfully!')