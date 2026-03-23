from core.data import PINNDataset
from core.model import PINN
from core.context import PhysicsContext
import os
import zipfile
import pandas as pd

def log_to_excel(epoch, approximator: PINN, pde_dataset: PINNDataset, center_dataset: PINNDataset, surface_dataset: PINNDataset, ic_dataset: PINNDataset, pde_loss_fn, bc_loss_fn, ic_loss_fn, ctx: PhysicsContext, first_run=False):
    excel_path = 'results/pinn_physics_report.xlsx'
    if  os.path.exists(excel_path):
        if not zipfile.is_zipfile(excel_path):
            print('⚠️ Corrupt excel found. Deleting...')
            os.remove(excel_path)
            first_run = True

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
    df_pde = pde_loss_fn(approximator=approximator, data=pde_dataset, ctx=ctx, return_df=True)
    df_bc_center, df_bc_surface = bc_loss_fn(approximator=approximator, ctx=ctx, data_center=center_dataset, data_surface=surface_dataset, return_df=True)
    df_ic = ic_loss_fn(approximator=approximator, data=ic_dataset, ctx=ctx, return_df=True)

    mode = 'w' if first_run or not os.path.exists(excel_path) else 'a'
    with pd.ExcelWriter(path=excel_path, engine='openpyxl', mode=mode, if_sheet_exists='replace' if mode=='a' else None) as writer:
        df_context.to_excel(excel_writer=writer, sheet_name='Constants', index=False)
        df_pde.to_excel(excel_writer=writer, sheet_name=f'PDE_Epoch_{epoch}', index=False)
        df_bc_center.to_excel(excel_writer=writer, sheet_name=f'BC0_Epoch_{epoch}', index=False)
        df_bc_surface.to_excel(excel_writer=writer, sheet_name=f'BCR_Epoch_{epoch}', index=False)
        df_ic.to_excel(excel_writer=writer, sheet_name=f'IC_Epoch_{epoch}', index=False)

    print(f'📊 Excel updated: Epoch {epoch} (PDE + BC + IC)')