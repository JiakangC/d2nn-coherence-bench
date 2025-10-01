# src/train.py
import os
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

from tools.logger import CSVLogger
from .vis import save_phase_mask, save_intensity_grid

torch.manual_seed(123); torch.cuda.manual_seed_all(123)

def get_data(N: int, batch_size: int):
    """
    MNIST dataloaders with resize to NxN using torchvision.
    """
    tfm = transforms.Compose([
        transforms.Resize((N, N)),
        transforms.ToTensor()
    ])
    train = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    loader_train = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    loader_test  = DataLoader(test, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
    return loader_train, loader_test


def forward_with_coherence(model: D2NNClassifier, x: torch.Tensor, cfg: dict) -> torch.Tensor:
    """
    Optical block with optional partial-coherence averaging.
    Returns intensity I [B,1,N,N].
    """
    if cfg["coherence"]["K"] > 1:
        def prop_fn(u_field: torch.Tensor) -> torch.Tensor:
            u1 = model.prop1(u_field)
            u2 = model.mask(u1)
            u3 = model.prop2(u2)
            return u3
        u0 = x.squeeze(1).to(torch.complex64)  # [B,N,N]
        I = partial_coherence_propagate(
            u0, prop_fn,
            K=cfg["coherence"]["K"],
            corr_px=cfg["coherence"]["corr_px"]
        )
    else:
        _, I = model(x)  # coherent single pass
    return I


def apply_measurement_effects(I: torch.Tensor, cfg: dict) -> torch.Tensor:
    """
    Shot noise, readout noise, quantization on intensity.
    Includes guards against tiny negatives/NaNs for Poisson.
    """
    I = torch.nan_to_num(I, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
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

        I = forward_with_coherence(model, x, cfg)
        I = apply_measurement_effects(I, cfg)
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


def train_main(cfg_path: str, outdir: str = "reports"):
    # --- output dirs ---
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "figures"), exist_ok=True)
    ckpt_dir = os.path.join(outdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- load config ---
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- data (torchvision) ---
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

    # --- logger ---
    logger = CSVLogger(os.path.join(outdir, "metrics.csv"))

    # --- train loop ---
    best_acc = 0.0
    epochs = cfg["train"]["epochs"]

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, opt, train_loader, device, cfg)
        te_loss, te_acc = evaluate(model, test_loader, device, cfg)
        print(f"Epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"test loss {te_loss:.4f} acc {te_acc:.4f}")
        logger.log_epoch(epoch, tr_loss, tr_acc, te_loss, te_acc)

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(
                {"model": model.state_dict(), "cfg": cfg},
                os.path.join(ckpt_dir, f"d2nn_epoch{epoch:03d}_acc{te_acc:.4f}.pt"),
            )

    # --- visuals ---
    try:
        save_phase_mask(model, path=os.path.join(outdir, "figures", "phase_mask.png"))
        save_intensity_grid(model, test_loader, grid_n=8,
                            path=os.path.join(outdir, "figures", "intensity_grid.png"),
                            device=device)
    except Exception as e:
        print("[WARN] Visualization failed:", e)

    logger.log_summary("best_test_acc", f"{best_acc:.4f}")
    print(f"Best test acc: {best_acc:.4f}")


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--outdir", type=str, default="reports", help="Output dir for this run")
    args = parser.parse_args()
    train_main(args.config, outdir=args.outdir)


if __name__ == "__main__":
    cli()
