# Additional training utilities for GraphAF if needed.
# In this implementation, the training logic is handled in model.fit_step, so this file can remain minimal.
# gen/models/graphaf/trainer.py
"""
Specialized trainer for GraphAF-style autoregressive training with teacher forcing.
You can keep using the unified trainer, but this gives you:
- scheduled teacher forcing
- optional entropy regularization on node/edge decisions
"""
import math
import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.optim as optim


@dataclass
class TrainCfg:
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 1.0
    log_every: int = 50
    ckpt_path: Optional[str] = None
    scheduler: Optional[str] = "cosine"
    step_size: int = 20
    gamma: float = 0.5
    teacher_forcing_start: float = 1.0  # fraction
    teacher_forcing_end: float = 0.5
    entropy_coef: float = 0.0


def _make_optimizer(model, cfg: TrainCfg):
    return optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def _make_scheduler(optimizer, cfg: TrainCfg):
    if cfg.scheduler is None or cfg.scheduler == "none":
        return None
    if cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    return None


def _epoch_teacher_forcing_ratio(cfg: TrainCfg, epoch: int) -> float:
    # linear anneal from start to end
    T = cfg.epochs
    start, end = cfg.teacher_forcing_start, cfg.teacher_forcing_end
    return max(end, start - (start - end) * (epoch - 1) / max(1, T - 1))


def train(model, train_loader, val_loader, device, cfg: TrainCfg):
    model.train()
    optimizer = _make_optimizer(model, cfg)
    scheduler = _make_scheduler(optimizer, cfg)

    best_val = math.inf
    for epoch in range(1, cfg.epochs + 1):
        tf_ratio = _epoch_teacher_forcing_ratio(cfg, epoch)
        running = 0.0
        n = 0
        for it, batch in enumerate(train_loader, 1):
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            # Use model's fit_step as core loss; if you wanted to integrate teacher forcing inside model,
            # add a flag in batch like batch["tf_ratio"] = tf_ratio and use it there.
            batch["tf_ratio"] = torch.tensor([tf_ratio], device=device)
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
                logging.info(f"[GraphAF] iter {it:05d} | loss={running/n:.4f} | tf={tf_ratio:.2f}")
        tr = running / max(1, n)

        if val_loader is not None:
            va = _validate(model, val_loader, device)
            logging.info(f"[GraphAF] epoch {epoch:03d}/{cfg.epochs} | train={tr:.4f} | val={va:.4f} | tf={tf_ratio:.2f}")
            if va < best_val and cfg.ckpt_path:
                best_val = va
                torch.save(model.state_dict(), cfg.ckpt_path)
                logging.info(f"[GraphAF] saved best checkpoint: {cfg.ckpt_path}")
        else:
            logging.info(f"[GraphAF] epoch {epoch:03d}/{cfg.epochs} | train={tr:.4f} | tf={tf_ratio:.2f}")

        if scheduler:
            scheduler.step()


@torch.no_grad()
def _validate(model, loader, device):
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
        running += loss.item()
        n += 1
    model.train()
    return running / max(1, n)
