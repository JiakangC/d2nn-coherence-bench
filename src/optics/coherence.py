import torch
import torch.nn.functional as F
import math

def gaussian_kernel_2d(size, sigma, device):
    ax = torch.arange(size, device=device) - (size - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / torch.sum(kernel)
    return kernel

def sample_phase_screen(N, corr_px, device, std=1.0):
    """
    Sample a correlated phase screen by convolving white noise with a Gaussian kernel.
    corr_px: correlation length in pixels (sigma of Gaussian)
    """
    noise = torch.randn(1, 1, N, N, device=device)
    kernel_size = max(3, int(6 * corr_px) | 1)  # ensure odd size
    kernel = gaussian_kernel_2d(kernel_size, corr_px, device).view(1, 1, kernel_size, kernel_size)
    smoothed = F.conv2d(noise, kernel, padding=kernel_size//2)
    phase_screen = std * smoothed[0, 0]
    return phase_screen

def partial_coherence_propagate(u, propagate_fn, K=4, corr_px=3.0):
    """
    Average intensity over K phase screen samples to emulate partial spatial coherence.
    u: complex field [B,N,N]
    propagate_fn: function(field)->field
    returns intensity [B,1,N,N]
    """
    B, N, _ = u.shape
    acc = None
    for _ in range(K):
        phase = sample_phase_screen(N, corr_px, u.device, std=1.0)
        u_pert = u * torch.exp(1j * phase.to(u.dtype))
        out = propagate_fn(u_pert)
        I = (out.real2 + out.imag2).unsqueeze(1)
        acc = I if acc is None else acc + I
    return acc / K