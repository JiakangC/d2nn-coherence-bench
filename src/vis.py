# src/vis.py
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

def save_phase_mask(model, path="reports/figures/phase_mask.png"):
    """Save learned phase mask phi = pi * tanh(theta)."""
    with torch.no_grad():
        theta = model.mask.theta.detach().cpu()
        phi = math.pi * torch.tanh(theta)
        phi_np = phi.numpy()
    plt.figure()
    plt.imshow(phi_np, cmap="twilight", interpolation="nearest")
    plt.colorbar()
    plt.title("Learned phase mask (phi)")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

@torch.no_grad()
def save_intensity_grid(model, data_loader, grid_n=8,
                        path="reports/figures/intensity_grid.png", device=None):
    """Save a small grid of detector intensities for sample inputs (coherent path)."""
    model.eval()
    x, _ = next(iter(data_loader))
    if device:
        x = x.to(device)
    # forward once (coherent)
    _, I = model(x)
    I = I[:grid_n]  # [K,1,N,N]
    K = I.shape[0]
    cols = int(np.ceil(np.sqrt(K)))
    rows = int(np.ceil(K / cols))
    plt.figure(figsize=(cols*2, rows*2))
    for i in range(K):
        plt.subplot(rows, cols, i+1)
        plt.imshow(I[i,0].detach().cpu().numpy(), interpolation="nearest")
        plt.axis("off")
    plt.suptitle("Detector intensity samples")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
