'''contains utility objects that help evaluate the performance of the PINN.'''
from .pinns import Pinn
import numpy as np
import torch

class Evaluator:
    '''evaluates the performance of pinn vs fdm'''
    SPECIES = ('c0', 'c1', 'c2')
    SNAPSHOTS = (0.0, 1/24, 4/24, 8/24, 12/24, 18/24, 24/24)
    def __init__(self, fdm_x: np.ndarray, fdm_sol, n_t=101, device='cpu'):
        '''
        fdm_x   : np.ndarray(N,) spatial nodes tetracted from get_diffucion_differential_operator
        fdm_sol : scipy OdeSolution 
        '''
        self.device = device
        self.r = torch.tensor(fdm_x, dtype=torch.float32).view(-1, 1)
        self.N = fdm_x.shape[0]
        self.t_grid = np.linspace(0.0, 1.0, n_t)

        Y = fdm_sol(self.t_grid)
        self.fdm = {                        # np.ndarray (3N, n_t)
            'c0': Y[:self.N, :],
            'c1': Y[self.N:2*self.N, :],
            'c2': Y[2*self.N:, :]
        }

    def _pinn_field(self, model: Pinn):
        pred = {'c0': [], 'c1': [], 'c2': []}
        net = model.net
        for time in self.t_grid:
            t = torch.full_like(self.r, float(time)).to(self.device)
            with torch.no_grad():
                c0, c1, c2 = net(self.r.to(self.device), t)
            pred['c0'].append(c0.cpu().numpy().ravel())
            pred['c1'].append(c1.cpu().numpy().ravel())
            pred['c2'].append(c2.cpu().numpy().ravel())
        return {k: np.stack(v, axis=1) for k, v in pred.items()}
    
    @staticmethod
    def _rel_l2(pred, true):
        """Relative L2: ||pred-true|| / ||true||, guarded against zero norm."""
        denom = np.linalg.norm(true)
        return float(np.linalg.norm(pred - true) / denom) if denom > 0 else float('nan')

    def score(self, model: Pinn):
        pred = self._pinn_field(model)
        for k in self.SPECIES:
            assert pred[k].shape == self.fdm[k].shape, f'{k}: {pred[k].shape} vs {self.fdm[k].shape}'
        pred_all = np.concatenate([pred[k] for k in self.SPECIES], axis=0)
        true_all = np.concatenate([self.fdm[k] for k in self.SPECIES], axis=0)

        per_species = {k: self._rel_l2(pred[k], self.fdm[k]) for k in self.SPECIES}

        per_snapshot = {}
        for ts in self.SNAPSHOTS:
            idx = int(round(ts * (len(self.t_grid) - 1)))
            per_snapshot[ts] = self._rel_l2(pred['c0'][:, idx], self.fdm['c0'][:, idx])

        return {
            'global_l2'      :self._rel_l2(pred_all, true_all),
            'per_species'    :per_species,
            'per_snapshot'  :per_snapshot,
            'error_field'    :{k: pred[k] - self.fdm[k] for k in self.SPECIES},
            'r'              :self.r.cpu().numpy().ravel(),
            't'              :self.t_grid
        }

    def compare(self, models: dict):
        """models: {label: Pinn}. Returns {label: score_dict}. The bake-off."""
        return {label: self.score(m) for label, m in models.items()}