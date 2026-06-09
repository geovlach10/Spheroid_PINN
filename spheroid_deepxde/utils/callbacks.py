import torch
import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde

class GradientDiagnosticCallback(dde.callbacks.Callback):
    def __init__(self, check_step=100):
        super().__init__()
        self.check_step = check_step

    def on_epoch_end(self):
        # Trigger exactly at the specified step
        if self.model.train_state.step == self.check_step:
            layer_names = []
            max_grads = []
            mean_grads = []

            # Dig into the PyTorch backend to extract the gradients
            for name, param in self.model.net.named_parameters():
                # We ignore biases to keep the plot clean, focusing only on weight matrices
                if param.requires_grad and "bias" not in name and param.grad is not None:
                    layer_names.append(name)
                    max_grads.append(param.grad.abs().max().item())
                    mean_grads.append(param.grad.abs().mean().item())

            # Render the gradient flow bar chart
            plt.figure(figsize=(8, 6))
            plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c", label="Max gradient")
            plt.bar(np.arange(len(mean_grads)), mean_grads, alpha=0.5, lw=1, color="b", label="Mean gradient")
            
            plt.hlines(0, -1, len(mean_grads), lw=2, color="k")
            plt.xticks(range(0, len(mean_grads)), layer_names, rotation=45, ha="right")
            plt.xlim(left=-1, right=len(mean_grads))
            
            plt.xlabel("Network Layers (Input $\\rightarrow$ Output)")
            plt.ylabel("Gradient Magnitude (Log Scale)")
            plt.yscale("log") # Log scale is essential for PINN gradient tracking
            plt.title(f"Gradient Flow Check at Step {self.check_step}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()

class AdaptiveLossWeighting(dde.callbacks.Callback):
    def __init__(self, update_every=100, momentum=0.9, base_weights=None):
        super().__init__()
        self.update_every = update_every
        self.momentum = momentum
        self.current_weights = np.array(base_weights, dtype=np.float32) if base_weights else None
        
    def on_train_begin(self):
        # Initialize weights if not provided, based on the number of targets
        if self.current_weights is None:
            num_losses = len(self.model.train_state.loss_train)
            self.current_weights = np.ones(num_losses, dtype=np.float32)
            
        print(f"Starting Adaptive Weighting with initial weights: {self.current_weights}")

    def on_epoch_end(self):
        epoch = self.model.train_state.step
        
        # Only update every N epochs to keep gradients stable
        if epoch % self.update_every == 0 and epoch > 0:
            # 1. Get the raw loss values from the current step
            raw_losses = np.array(self.model.train_state.loss_train)
            
            # Prevent division by zero if a loss perfectly hits 0
            safe_losses = np.maximum(raw_losses, 1e-8)
            
            # 2. Calculate the inverse proportions (higher loss = lower dynamic weight)
            # We normalize so the mean weight always stays around 1.0
            inverse_losses = 1.0 / safe_losses
            normalized_new_weights = inverse_losses / np.mean(inverse_losses)
            
            # 3. Apply momentum (EMA) to smooth the transition
            self.current_weights = (self.momentum * self.current_weights) + ((1.0 - self.momentum) * normalized_new_weights)
            
            # 4. Inject the new weights back into the DeepXDE backend
            # DeepXDE converts loss_weights to a tensor, so we must update it safely
            if self.model.loss_weights is not None:
                # Update the Python list
                self.model.loss_weights = self.current_weights.tolist()
                
                # Update the PyTorch backend tensor in-place
                if hasattr(self.model, 'net') and dde.backend.backend_name == "pytorch":
                    with torch.no_grad():
                        # DeepXDE stores the compiled loss weights internally.
                        # We pull the current learning rate from the optimizer if needed.
                        if hasattr(self.model, 'lr'):
                            lr = self.model.lr
                        elif hasattr(self.model, 'opt'):
                            lr = self.model.opt.defaults.get('lr', 1e-4)
                        else:
                            lr = 1e-4

                        self.model.compile(
                            optimizer=self.model.opt_name,
                            lr=lr,
                            loss_weights=self.current_weights.tolist()
                        )
                        
            print(f"\n[Epoch {epoch}] Adaptive Weights Updated: {np.round(self.current_weights, 3)}")