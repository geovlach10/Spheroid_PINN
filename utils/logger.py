from core.data import PINNDataset
from core.model import PINN
from core.context import PhysicsContext
import os
import pandas as pd

def log_to_excel(epoch, approximator: PINN, pde_dataset: PINNDataset, pde_loss_fn,  ctx: PhysicsContext, first_run=False):
    excel_path = 'results/pinn_physics_report.xlsx'
    context_list = []
    context_list.extend([
        {'parameter': 'R(μm)', 'value': f'{ctx.R:.2f}'},
        {'parameter': 'tau(h)', 'value': f'{ctx.tau:.2f}'},
        {'parameter': 'C0(nM)', 'value': f'{ctx.C0:.2f}'}
    ])
    for name, param in ctx.pde_parameters.items():
        context_list.append({"parameter": name, 'value': f'{param.item():.2e}'})
    for name, param in ctx.pde_weights.items():
        context_list.append({"parameter": name, 'value': f'{param.item():.3f}'})

    df_context = pd.DataFrame(context_list)
    df_physics = pde_loss_fn(approximator=approximator, data=pde_dataset, ctx=ctx, term_by_term_analysis=True)
    
    mode = 'w' if first_run or not os.path.exists(excel_path) else 'a'
    with pd.ExcelWriter(path=excel_path, engine='openpyxl', mode=mode, if_sheet_exists='replace' if mode=='a' else None) as writer:
        df_context.to_excel(excel_writer=writer, sheet_name='Constants', index=False)
        df_physics.to_excel(excel_writer=writer, sheet_name=f'Epoch_{epoch}', index=False)