import torch
import torch.nn as nn
import math

# define the frequency grid
def make_frequency_grid(N, dx, device, dtype):
    fx = torch.fft.fftfreq(N, d=dx, device=device, dtype=dtype) # shape [N]
    fy = fx
    FX, FY = torch.meshgrid(fx, fy, indexing='ij') # shape [N, N]
    return FX, FY

def angular_spectrum_H(N, dx, lam, z, device, dtype):
    '''
    N: image size
    dx: pixel size
    lam: wavelength
    z: propagation distance
    '''
    FX, FY = make_frequency_grid(N, dx, device, dtype) # shape [N, N]
    k = 2 * math.pi / lam
    kx = 2 * math.pi * FX
    ky = 2 * math.pi * FY
    kz_squared = k**2 - kx**2 - ky**2
    kz_squared = torch.clamp(kz_squared, min=0.0) # ev
    kz = torch.sqrt(kz_squared + 0j) # shape [N, N]
    H = torch.exp(1j * kz * z) # shape [N, N] complex
    return H.to(torch.complex64)

class FresnelProp(nn.Module):
    def __init__(self, N, dx, lam, z):
        super(FresnelProp, self).__init__()
        self.N = N
        self.dx = dx
        self.lam = lam
        self.z = z
        self.register_buffer('H', None, persistent=False)
        
    def build_H(self, device):
        H = angular_spectrum_H(self.N, self.dx, self.lam, self.z, device, torch.float32)
        self.H = H
    
    def forward(self, u):
        if self.H is None or self.H.device != u.device:
            self.build_H(u.device)
        U = torch.fft.fft2(u) * self.H
        return torch.fft.ifft2(U)

class PhaseMask(nn.Module):
    def __init__(self, N, init_std=0.01):
        super(PhaseMask, self).__init__()
        self.theta = nn.Parameter(torch.randn(N, N) * init_std)
        
    def forward(self, u):
        phi = math.pi * torch.tanh(self.theta)
        return u * torch.exp(1j * phi.to(u.dtype))