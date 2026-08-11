"""Training-point datasets and sampling for PINN training.
 
Dataset wraps a batch of (r, t) coordinate pairs as a single tensor,
with named-column accessors and support for concatenation (`+`) and
visualization (`plotme`).
 
DatasetSampler is the generic point-sampling utility PINN uses (directly,
for the interior collocation set) and that IC/BC-building code calls
(to construct the Dataset objects wrapped in InitialCondition /
BoundaryCondition -- see pinns.py). It assumes a 1D radially-symmetric
domain: r spans some [r_min, r_max] and t spans some [t_min, t_max],
which covers any radial PDE problem (not just trastuzumab specifically),
but is not a fully general N-dimensional sampler.
 
Four sampling patterns are provided:
    sample_collocation_points -- quasi-random interior points (Latin
        Hypercube), for evaluating the PDE residual.
    sample_initial_points      -- points spread across r at t = t_min,
        for an initial condition.
    sample_center_points       -- points spread across t at r = r_min
        (the domain's inner boundary, e.g. r=0 for a sphere/cylinder
        center), for an inner boundary condition.
    sample_surface_points      -- points spread across t at r = r_max
        (the domain's outer boundary), for an outer boundary condition.
 
Example:
    sampler = DatasetSampler(seed=42)
    collocation = sampler.sample_collocation_points(n_points=10_000)
    initial     = sampler.sample_initial_points(n_points=200)
    center      = sampler.sample_center_points(n_points=200)
    surface     = sampler.sample_surface_points(n_points=200)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import torch


class Dataset:

    """A batch of (r, t) training points, stored as a single (N, 2) tensor.
 
    Column 0 is r, column 1 is t -- access them via the `.r`/`.t`
    properties rather than indexing `.data` directly, so calling code
    doesn't need to remember the column order.
 
    Datasets concatenate with `+` (e.g. `collocation + more_points`),
    which is how `PINN.resample_datasets` grows the collocation set in
    place, and how `PINN.__init__` combines every IC/BC dataset into
    `self._DATASET` for bookkeeping.
 
    Args:
        data: array-like or Tensor of shape (N, 2) -- (r, t) pairs.
            Converted to a Tensor automatically if not already one.
        n_points: point count, stored for bookkeeping/logging. Not
            derived from `data.shape[0]` automatically -- keep this in
            sync with `data` yourself if you construct a Dataset by hand.
        upper_bounds: the (r, t) upper bounds this dataset was sampled
            within, if any -- stored for reference (e.g. by `plotme`'s
            title), not enforced.
        device, dtype: passed to `torch.tensor(...)` when `data` isn't
            already a Tensor. Has no effect if `data` is already a
            Tensor (its existing device/dtype is kept as-is).
        name: a label for this dataset, e.g. `'collocation'`,
            `'initial'` -- used by `DatasetSampler`'s methods to tag
            what they return, and by `plotme`'s title.
    """

    def __init__(self, data, n_points=0, upper_bounds=None, device='cpu', dtype=torch.float32, name: str=''):
        self.name = name
        self.n_points = n_points
        self.data = torch.tensor(data, dtype=dtype, device=device) if not isinstance(data, torch.Tensor) else data
        self.upper_bounds = upper_bounds

    @property
    def r(self) -> torch.Tensor:
        """The radial coordinate column, shape (N, 1)."""
        return self.data[:, 0:1]

    @property
    def t(self) -> torch.Tensor:
        """The time coordinate column, shape (N, 1)."""
        return self.data[:, 1:2]

    def plotme(self, marker_size=0.2):
        """Scatter-plot this dataset's points in the (r, t) plane --
        useful for a quick visual sanity check of a sampling scheme,
        e.g. after RAR-G refinement (see `sampling.py`)."""
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(self.r.cpu(), self.t.cpu(), s=marker_size)
        plt.xlabel('r')
        plt.ylabel('t')
        plt.title(f'(number of points: {len(self.data)})')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    def __add__(self, other):
        """Concatenates two Datasets vertically (more rows, same 2 columns).
        `n_points` is summed; `name`/`upper_bounds` are NOT carried over
        from either operand -- the result has default `name=''`,
        `upper_bounds=None`. Rename the result yourself if you need a
        label on the combined dataset."""
        if not isinstance(other, Dataset):
            raise TypeError(f'unsupported operand type(s) for +: {type(self)} and {type(other)}')
        new_data = torch.cat([self.data, other.data], dim=0)
        return Dataset(data=new_data, n_points=self.n_points + other.n_points)
        
    def __len__(self):
        return self.r.shape[0]
    
class DatasetSampler:

    """Builds Dataset instances for the four standard PINN training-point
    patterns (interior collocation, initial, inner-boundary/'center',
    outer-boundary/'surface') on a 1D radial domain.
 
    KNOWN LIMITATION: `device`/`dtype` are stored at construction but
    currently NOT used by any of the four sample_* methods below -- every
    Dataset they return is built with Dataset's own defaults
    (device='cpu', dtype=torch.float32), regardless of what this sampler
    was constructed with. If you need points on a specific device/dtype,
    move/cast the returned Dataset's `.data` yourself for now.
 
    Args:
        seed: base seed for `sample_collocation_points`'s Latin
            Hypercube draw (see that method's `seed_offset` for how
            multiple calls vary). Does not affect the other three
            methods, which are deterministic (`np.linspace`-based, no
            randomness).
        device, dtype: see KNOWN LIMITATION above -- currently inert.
 
    Example:
        sampler = DatasetSampler(seed=42)
        collocation = sampler.sample_collocation_points(n_points=10_000, u_bounds=[1.0, 1.0])
    """

    def __init__(self, seed=42, device='cpu', dtype=torch.float32):
        self.seed = seed
        self.device = device
        self.dtype = dtype
    
    def sample_collocation_points(self, n_points, l_bounds=[0, 0], u_bounds=[1, 1], seed_offset=0):
        """Quasi-random interior points via Latin Hypercube Sampling --
        for evaluating the PDE residual (`PDE.residual_fn`) across the
        interior of the (r, t) domain.
 
        Args:
            n_points: how many points to draw.
            l_bounds, u_bounds: `[r_min, t_min]` / `[r_max, t_max]` box
                to sample within.
            seed_offset: added to `self.seed` before drawing, so
                repeated calls (e.g. one per RAR-G round -- see
                `sampling.py`) can draw a different point cloud each
                time without changing the sampler's base seed. NOTE:
                `PINN.resample_datasets` currently does NOT pass a
                changing `seed_offset` when it calls this internally --
                see the project TODO on fixed-seed resampling.
 
        Returns:
            Dataset, name='collocation', with `n_points` rows.
        """
        sampler = qmc.LatinHypercube(d=2, seed=self.seed + seed_offset)
        dataset = sampler.random(n=n_points)
        dataset = qmc.scale(sample=dataset, l_bounds=l_bounds, u_bounds=u_bounds)
        return Dataset(data=dataset, name='collocation', upper_bounds=u_bounds, n_points=n_points)

    def sample_initial_points(self, n_points, l_bounds=[0, 0], u_bounds=[1, 1]):
        """Points spread evenly across r, all at t = `l_bounds[1]`
        (the domain's lower time bound -- typically 0) -- for an
        initial condition (see `InitialCondition` in pinns.py).
 
        Args:
            n_points: how many points to draw.
            l_bounds, u_bounds: `l_bounds[0]`/`u_bounds[0]` set the r
                range these points span; `l_bounds[1]` sets the fixed
                time value every point sits at.
 
        Returns:
            Dataset, name='initial', with `n_points` rows.
        """
        r = np.linspace(0, 1, n_points).reshape(-1, 1) * (u_bounds[0] - l_bounds[0]) + l_bounds[0] 
        t = np.zeros_like(r) + l_bounds[1]
        dataset = np.hstack([r, t])
        return Dataset(data=dataset, name='initial', upper_bounds=u_bounds, n_points=n_points)

    def sample_center_points(self, n_points, l_bounds=[0, 0], u_bounds=[1, 1]):
        """Points spread evenly across t, all at r = `l_bounds[0]`
        (the domain's inner radial boundary -- e.g. r=0, the center of
        a sphere/cylinder) -- for an inner boundary condition (e.g.
        `center_neumann` in trastuzumab.residuals, wrapped in a
        `BoundaryCondition`).
 
        Args:
            n_points: how many points to draw.
            l_bounds, u_bounds: `l_bounds[0]` sets the fixed r value
                every point sits at; `l_bounds[1]`/`u_bounds[1]` set the
                t range these points span.
 
        Returns:
            Dataset, name='center', with `n_points` rows.
        """
        t = np.linspace(0, 1, n_points).reshape(-1, 1) * (u_bounds[1] - l_bounds[1]) + l_bounds[1]
        r = np.zeros_like(t) + l_bounds[0]
        dataset = np.hstack([r, t])
        return Dataset(data=dataset, name='center', upper_bounds=u_bounds, n_points=n_points)

    def sample_surface_points(self, n_points, l_bounds=[0, 0], u_bounds=[1, 1]):
        """Points spread evenly across t, all at r = `u_bounds[0]`
        (the domain's outer radial boundary) -- for an outer boundary
        condition (e.g. `surface_robin` in trastuzumab.residuals,
        wrapped in a `BoundaryCondition`).
 
        Args:
            n_points: how many points to draw.
            l_bounds, u_bounds: `u_bounds[0]` sets the fixed r value
                every point sits at; `l_bounds[1]`/`u_bounds[1]` set the
                t range these points span.
 
        Returns:
            Dataset, name='surface', with `n_points` rows.
        """
        t = np.linspace(0, 1, n_points).reshape(-1, 1) * (u_bounds[1] - l_bounds[1]) + l_bounds[1]
        r = np.ones_like(t) * u_bounds[0]
        dataset = np.hstack([r, t])
        return Dataset(data=dataset, name='surface', upper_bounds=u_bounds, n_points=n_points)