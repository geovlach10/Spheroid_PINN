import os
import zipfile
import pandas as pd


from tqdm.notebook import tqdm

class PINNLoger():
    def __init__(self, num_epochs):
        self.pbar = tqdm(range(num_epochs), desc='Training', unit='epoch')
        self.history = {'loss': {}, 'weights': {}, 'parameters': {}}
        

    def update(self, epoch, losses: dict, weights: dict, parameters: dict, every=1000):
        # Update history
        for key, value in losses.items():
            self.history['loss'].setdefault(key, []).append(value.item())
        for key, value in weights.items():
            self.history['weights'].setdefault(key, []).append(value.item())
        for key, value in parameters.items():
            self.history['parameters'].setdefault(key, []).append(value.item())

        # Update progress bar
        self.pbar.update(1)
        self.pbar.set_postfix({key: f'{value.item():.4e}' for key, value in losses.items()})

        # Update table display
        if epoch % every == 0:
            row = {key: f'{value[-1]:.4e}' for key, value in self.history['loss'].items()}
            row['epoch'] = epoch
            self.table.append(row)

            df = pd.DataFrame(self.table)
            cols = ['epoch'] + [c for c in df.columns if c != 'epoch']
            df = df[cols]

            self.table_handle.update(df)
    
    def finalize(self):
        self.pbar.close()
        return pd.DataFrame(self.history)
    

def log_to_excel(epoch, approximator: PINN, pde_dataset: Dataset, center_dataset: Dataset, surface_dataset: Dataset, ic_dataset: Dataset, pde_loss_fn, bc_loss_fn, ic_loss_fn, ctx: PhysicsContext, first_run=False):
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


## Simple logger.
# Log to screen.
        # if epoch == 0:
        #     header = f"{'Epoch':^10} | {'Total':^10} | {'pde_f':^10} | {'pde_b':^10} | {'pde_i':^10} | {'bc0':^10} | {'bcR':^10} | {'ic_f':^10} | {'ic_b':^10} | {'ic_i':^10} "
        #     print('-' * len(header))
        #     print(header)
        #     print('-' * len(header))
        # if epoch % 200 == 0:
        #     line = f"{epoch:^10} | {losses['total'].item():^10.4e} | {losses['pde_f'].item():^10.4e} | {losses['pde_b'].item():^10.4e} | {losses['pde_i'].item():^10.4e} | {losses['center'].item():^10.4e} | {losses['surface'].item():^10.4e} | {losses['ic_f'].item():^10.4e} | {losses['ic_b'].item():^10.4e} | {losses['ic_i'].item():^10.4e}"
        #     print(line)
        # # if epoch % 1000 == 0:
        # #     log_to_excel(epoch=epoch, approximator=self.approximator, pde_dataset=self.pde_training_dataset, center_dataset=self.center_training_dataset, surface_dataset=self.surface_training_dataset, ic_dataset=self.initial_training_dataset, pde_loss_fn=self.pde_loss_fn, bc_loss_fn=self.bc_loss_fn,ic_loss_fn=self.ic_loss_fn, ctx=self.ctx)
        # pbar.set_postfix({key: f'{value.item():.4e}' for key, value in losses.items()})