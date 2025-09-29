# src/optics/noise.py
import torch

def apply_shot_noise(I: torch.Tensor, scale: float = 100.0) -> torch.Tensor:
    """
    Apply Poisson (shot) noise to intensity I (>=0).
    - If scale <= 0, returns I unchanged.
    - Robust to tiny negative values / NaNs / Infs.
    """
    if scale is None or scale <= 0:
        return I

    # Clean numerical artifacts BEFORE scaling/poisson
    I = torch.nan_to_num(I, nan=0.0, posinf=0.0, neginf=0.0)
    I = I.clamp_min(0.0)

    # Build Poisson rate and ensure valid dtype/range
    rate = (I * float(scale)).to(dtype=torch.float32)
    rate = torch.clamp(rate, min=0.0, max=1e12)

    noisy_counts = torch.poisson(rate)
    return noisy_counts / float(scale)

def apply_readout_noise(I: torch.Tensor, sigma: float = 0.0) -> torch.Tensor:
    if sigma is None or sigma <= 0:
        return I
    return I + sigma * torch.randn_like(I)

def quantize(I: torch.Tensor, bits=None) -> torch.Tensor:
    if bits is None:
        return I
    levels = max(1, 2**int(bits) - 1)
    Iq = torch.round(I * levels) / levels
    return Iq.clamp_min(0.0)
