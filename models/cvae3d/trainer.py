# Additional training routines for CVAE3D if needed.
# Training is primarily handled in model.fit_step in this implementation.
# gen/models/cvae3d/trainer.py
import math
import time
import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class TrainCfg:
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 1.0
    log_every: int = 50
    ckpt_path: Optional[str] = None
    scheduler: Optional[str] = "cosine"  # ["none","step","cosine"]
    step_size: int = 20
    gamma: float = 0.5


def _make_optimizer(model, cfg: TrainCfg):
    return optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def _make_scheduler(optimizer, cfg: TrainCfg):
    if cfg.scheduler is None or cfg.scheduler == "none":
        return None
    if cfg.scheduler == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    if cfg.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    return None


def train_epoch(model, loader, optimizer, device, cfg: TrainCfg):
    model.train()
    running = 0.0
    n = 0
    for it, batch in enumerate(loader, 1):
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)
        loss = model.fit_step(batch)
        if isinstance(loss, (tuple, list)):
            loss = loss[0]
        optimizer.zero_grad()
        loss.backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        running += loss.item()
        n += 1
        if cfg.log_every and it % cfg.log_every == 0:
            logging.info(f"[CVAE3D] iter {it:05d} | loss={running/n:.4f}")
    return running / max(1, n)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    running = 0.0
    n = 0
    for batch in loader:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)
        loss = model.fit_step(batch)
        if isinstance(loss, (tuple, list)):
            loss = loss[0]
        running += float(loss.item())
        n += 1
    return running / max(1, n)


def train(model, train_loader, val_loader, device, cfg: TrainCfg):
    """Optional specialized trainer for CVAE-3D (the unified trainer works too)."""
    optimizer = _make_optimizer(model, cfg)
    scheduler = _make_scheduler(optimizer, cfg)
    best_val = math.inf
    for epoch in range(1, cfg.epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, device, cfg)
        if val_loader:
            va = validate(model, val_loader, device)
            logging.info(f"[CVAE3D] epoch {epoch:03d}/{cfg.epochs} | train={tr:.4f} | val={va:.4f}")
            if va < best_val:
                best_val = va
                if cfg.ckpt_path:
                    torch.save(model.state_dict(), cfg.ckpt_path)
                    logging.info(f"[CVAE3D] saved best checkpoint: {cfg.ckpt_path}")
        else:
            logging.info(f"[CVAE3D] epoch {epoch:03d}/{cfg.epochs} | train={tr:.4f}")
        if scheduler:
            scheduler.step()
