# src/train.py
import os
import math
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .models.d2nn_classifier import D2NNClassifier
from .optics.coherence import partial_coherence_propagate
from .optics.noise import apply_shot_noise, apply_readout_noise, quantize


def get_data(N: int, batch_size: int):
    """
    MNIST dataloaders with resize to NxN.
    """
    tfm = transforms.Compose([
        transforms.Resize((N, N)),
        transforms.ToTensor()
    ])
    train = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    loader_train = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    loader_test = DataLoader(test, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
    return loader_train, loader_test


def forward_with_coherence(model: D2NNClassifier, x: torch.Tensor, cfg: dict) -> torch.Tensor:
    """
    Runs the optical block with partial-coherence averaging (K>1) or
    just a single coherent pass (K=1). Returns intensity I [B,1,N,N].
    """
    if cfg["coherence"]["K"] > 1:
        # Wrap the optical path as a function (field -> field) to pass into the K-sample average.
        def prop_fn(u_field: torch.Tensor) -> torch.Tensor:
            u1 = model.prop1(u_field)
            u2 = model.mask(u1)
            u3 = model.prop2(u2)
            return u3

        u0 = x.squeeze(1).to(torch.complex64)                # [B,N,N] complex field
        I = partial_coherence_propagate(
            u0, prop_fn,
            K=cfg["coherence"]["K"],
            corr_px=cfg["coherence"]["corr_px"]
        )                                                     # [B,1,N,N]
    else:
        # Fully coherent: use model forward once
        _, I = model(x)                                       # [B,1,N,N]
    return I


def apply_measurement_effects(I: torch.Tensor, cfg: dict) -> torch.Tensor:
    """
    Add shot noise, readout noise, and quantization on intensity.
    """
    I = apply_shot_noise(I, scale=cfg["noise"]["shot_scale"])
    I = apply_readout_noise(I, sigma=cfg["noise"]["read_sigma"])
    I = quantize(I, bits=cfg["noise"]["quant_bits"])
    return I


def train_one_epoch(model, opt, loader, device, cfg):
    model.train()
    ce = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    loss_sum = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        # Optical block → intensity (with/without partial coherence)
        I = forward_with_coherence(model, x, cfg)
        # Measurement pipeline
        I = apply_measurement_effects(I, cfg)
        # Linear readout
        logits = model.readout(I)

        loss = ce(logits, y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    loss_sum = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        I = forward_with_coherence(model, x, cfg)
        I = apply_measurement_effects(I, cfg)
        logits = model.readout(I)

        loss = ce(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return loss_sum / total, correct / total


def train_main(cfg_path: str):
    # --- load config ---
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- data ---
    N = cfg["model"]["N"]
    train_loader, test_loader = get_data(N, cfg["train"]["batch_size"])

    # --- model ---
    model = D2NNClassifier(
        N=N,
        dx=cfg["optics"]["dx"],
        lam=cfg["optics"]["lam"],
        z=cfg["optics"]["z"],
        hidden=cfg["model"]["hidden"],
    ).to(device)

    # --- optimizer ---
    opt = optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["wd"])

    # --- train loop ---
    best_acc = 0.0
    epochs = cfg["train"]["epochs"]
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, opt, train_loader, device, cfg)
        te_loss, te_acc = evaluate(model, test_loader, device, cfg)
        print(f"Epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| test loss {te_loss:.4f} acc {te_acc:.4f}")

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(
                {"model": model.state_dict(), "cfg": cfg},
                f"checkpoints/d2nn_epoch{epoch:03d}_acc{te_acc:.4f}.pt",
            )

    print(f"Best test acc: {best_acc:.4f}")


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    train_main(args.config)


if __name__ == "__main__":
    cli()
