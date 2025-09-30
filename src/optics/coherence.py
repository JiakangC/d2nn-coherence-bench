# src/optics/coherence.py
import torch
import torch.nn.functional as F

def _gaussian_kernel_2d(size: int, sigma: float, device):
    ax = torch.arange(size, device=device) - (size - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    g = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    g = g / g.sum()
    return g

def sample_phase_screen(N: int, corr_px: float, device, std: float = 1.0):
    """
    Sample a correlated phase screen by convolving white noise with a Gaussian kernel.
    corr_px: correlation length in pixels (sigma of Gaussian).
    Returns a real-valued [N, N] tensor.
    """
    noise = torch.randn(1, 1, N, N, device=device)
    ksize = max(3, int(6 * corr_px) | 1)  # odd size
    g = _gaussian_kernel_2d(ksize, max(corr_px, 1e-6), device).view(1, 1, ksize, ksize)
    smooth = F.conv2d(noise, g, padding=ksize // 2)
    phase = std * smooth[0, 0]  # [N, N]
    return phase

def partial_coherence_propagate(u, propagate_fn, K: int = 4, corr_px: float = 3.0):
    """
    Average intensity over K phase-screen samples to emulate partial spatial coherence.
    u: complex field [B, N, N] (dtype complex64)
    propagate_fn: function(field)->field that maps [B, N, N] complex -> [B, N, N] complex
    Returns intensity [B, 1, N, N] (real, >=0).
    """
    B, N, _ = u.shape
    acc = None
    for _ in range(K):
        phase = sample_phase_screen(N, corr_px, u.device, std=1.0)
        u_pert = u * torch.exp(1j * phase.to(u.dtype))  # [B, N, N] complex
        out = propagate_fn(u_pert)                      # [B, N, N] complex
        # ---- fixed line below ----
        I = (out.real**2 + out.imag**2).unsqueeze(1)    # [B, 1, N, N]
        acc = I if acc is None else acc + I
    return acc / K
