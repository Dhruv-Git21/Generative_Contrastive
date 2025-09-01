# The GraphVAE training is handled via model.fit_step in this implementation.
# Additional training utilities can be placed here if needed.
# gen/models/graphvae/trainer.py
"""
GraphVAE trainer with KL-annealing (beta schedule).
You can still use the unified trainer; this adds KL warmup support.
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
    # KL anneal
    beta_start: float = 0.0
    beta_end: float = 1.0
    warmup_epochs: int = 10


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


def _beta(epoch: int, cfg: TrainCfg) -> float:
    if epoch <= cfg.warmup_epochs:
        return cfg.beta_start + (cfg.beta_end - cfg.beta_start) * (epoch / max(1, cfg.warmup_epochs))
    return cfg.beta_end


def train(model, train_loader, val_loader, device, cfg: TrainCfg):
    model.train()
    optimizer = _make_optimizer(model, cfg)
    scheduler = _make_scheduler(optimizer, cfg)

    best_val = math.inf
    for epoch in range(1, cfg.epochs + 1):
        beta = _beta(epoch, cfg)
        running = 0.0
        n = 0
        for it, batch in enumerate(train_loader, 1):
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            # Temporarily set a global beta in the model (or pass via batch flag)
            # For simplicity, we scale KLD by replacing model.fit_step if it returns components.
            loss = model.fit_step(batch)
            if isinstance(loss, (tuple, list)):
                # If you decide to return (recon, kld) from fit_step, handle here
                total = loss[0] + beta * loss[1]
            else:
                total = loss
            optimizer.zero_grad()
            total.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            running += float(total.item())
            n += 1
            if cfg.log_every and it % cfg.log_every == 0:
                logging.info(f"[GraphVAE] iter {it:05d} | beta={beta:.3f} | loss={running/n:.4f}")
        tr = running / max(1, n)

        if val_loader is not None:
            va = _validate(model, val_loader, device, beta)
            logging.info(f"[GraphVAE] epoch {epoch:03d}/{cfg.epochs} | train={tr:.4f} | val={va:.4f} | beta={beta:.3f}")
            if va < best_val and cfg.ckpt_path:
                best_val = va
                torch.save(model.state_dict(), cfg.ckpt_path)
                logging.info(f"[GraphVAE] saved best checkpoint: {cfg.ckpt_path}")
        else:
            logging.info(f"[GraphVAE] epoch {epoch:03d}/{cfg.epochs} | train={tr:.4f} | beta={beta:.3f}")

        if scheduler:
            scheduler.step()


@torch.no_grad()
def _validate(model, loader, device, beta: float):
    model.eval()
    running = 0.0
    n = 0
    for batch in loader:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)
        loss = model.fit_step(batch)
        if isinstance(loss, (tuple, list)):
            total = loss[0] + beta * loss[1]
        else:
            total = loss
        running += float(total.item())
        n += 1
    model.train()
    return running / max(1, n)
