from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from network import GlyphNet
from settings import Settings
from synth import make_dataset

log = logging.getLogger("glyphsort")


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _split(X: np.ndarray, y: np.ndarray, share: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    cut = int((1.0 - share) * len(X))
    tr, va = order[:cut], order[cut:]
    return (
        TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(y[tr])),
        TensorDataset(torch.from_numpy(X[va]), torch.from_numpy(y[va])),
    )


def _epoch(net, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    net.train(training)
    total_loss = total_hits = total = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if training:
                optimizer.zero_grad()
            logits = net(xb)
            loss = criterion(logits, yb)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
            total_hits += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
    return total_loss / total, total_hits / total


def train(cfg: Settings) -> float:
    device = pick_device()
    log.info("device: %s", device)

    X, y = make_dataset(cfg.samples_per_digit, cfg.image_side, cfg.seed)
    log.info("dataset: X=%s y=%s", X.shape, y.shape)

    train_ds, val_ds = _split(X, y, cfg.validation_share)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    net = GlyphNet(cfg.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    ckpt = Path(cfg.weights_path)
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    best = 0.0
    stale = 0

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = _epoch(net, train_loader, criterion, device, optimizer)
        scheduler.step()
        _, va_acc = _epoch(net, val_loader, criterion, device)

        log.info(
            "epoch %02d/%d loss=%.4f train=%.2f%% val=%.2f%%",
            epoch, cfg.epochs, tr_loss, tr_acc * 100, va_acc * 100,
        )

        if va_acc > best:
            best, stale = va_acc, 0
            torch.save(
                {
                    "state": net.state_dict(),
                    "best_val": best,
                    "epoch": epoch,
                    "num_classes": cfg.num_classes,
                    "image_side": cfg.image_side,
                },
                ckpt,
            )
            log.info("checkpoint saved: %s", ckpt)
        else:
            stale += 1
            if stale >= cfg.patience:
                log.info("early stopping at epoch %d", epoch)
                break

    log.info("best val: %.2f%% -> %s", best * 100, ckpt.resolve())
    return best


def load_model(cfg: Settings, device: torch.device | None = None):
    device = device or pick_device()
    ckpt = Path(cfg.weights_path)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {ckpt}. Run: python glyphsort.py train"
        )

    try:
        payload = torch.load(ckpt, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(ckpt, map_location=device)

    net = GlyphNet(payload.get("num_classes", cfg.num_classes)).to(device)
    net.load_state_dict(payload["state"])
    net.eval()
    return net, device, payload