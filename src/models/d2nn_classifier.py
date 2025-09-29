import torch
import torch.nn as nn
from ..optics.fourier import FresnelProp, PhaseMask

class D2NNClassifier(nn.Module):
    
    def __init__(self,N=32, dx=8e-6, lam=532e-9, z=0.02, hidden=128):
        super().__init__()
        self.N = N
        self.prop1 = FresnelProp(N, dx, lam, z/2)
        self.mask = PhaseMask(N)
        self.prop2 = FresnelProp(N, dx, lam, z/2)
        self.readout = nn.Sequential(
        nn.Flatten(),
        nn.Linear(N*N, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, 10)
        )
        
    def forward_field(self, x):
        # x: [B,1,N,N] real in [0,1], treat as amplitude
        u0 = x.squeeze(1).to(torch.complex64)
        u1 = self.prop1(u0)
        u2 = self.mask(u1)
        u3 = self.prop2(u2)
        return u3

    def forward(self, x):
        u = self.forward_field(x)
        I = (u.real**2 + u.imag**2).unsqueeze(1)   # [B,1,N,N]
        y = self.readout(I)
        return y, I