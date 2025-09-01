# Additional training routines for EquivariantFlowMatching if needed.
# Our model.fit_step handles one integration step training. In practice, you'd sample multiple t and integrate more steps.
# gen/models/equifm/trainer.py
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
    # Flow-matching specific
    n_time_samples: int = 4  # number of random t samples per graph per batch


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


def _flow_matching_step(model, batch, device, n_time_samples: int):
    """
    Path-consistency training: sample t ~ U(0,1), interpolate states, predict velocities,
    regress to target velocities (x1 - x0), (y1 - y0).
    """
    B = len(batch["num_nodes"])
    total = 0.0
    for i in range(B):
        n = batch["num_nodes"][i]
        if n == 0:
            continue
        coords_1 = batch["coords"][i][:n].to(device)
        types_1 = batch["node_feats"][i][:n].to(device)
        embed = batch["embeddings"][i].unsqueeze(0).to(device)

        # Start state: random noise for coords, dummy for types
        coords_0 = torch.randn_like(coords_1)
        types_0 = torch.zeros_like(types_1)
        if types_0.shape[1] > 0:
            types_0[:, 0] = 1.0  # index 0 = dummy

        # sample multiple ts for better supervision
        for _ in range(n_time_samples):
            t = torch.rand(1, device=device).item()
            x_t = (1 - t) * coords_0 + t * coords_1
            y_t = (1 - t) * types_0 + t * types_1
            v_x_target = (coords_1 - coords_0)  # linear path -> constant velocity
            v_y_target = (types_1 - types_0)

            v_x_pred, v_y_pred = model(x_t, y_t, embed)
            loss = torch.mean((v_x_pred - v_x_target) ** 2) + torch.mean((v_y_pred - v_y_target) ** 2)
            total += loss
    return total / max(1, B * n_time_samples)


def train(model, train_loader, val_loader, device, cfg: TrainCfg):
    """Optional specialized trainer for Equivariant Flow Matching."""
    model.train()
    optimizer = _make_optimizer(model, cfg)
    scheduler = _make_scheduler(optimizer, cfg)

    best_val = math.inf
    for epoch in range(1, cfg.epochs + 1):
        running = 0.0
        n = 0
        for it, batch in enumerate(train_loader, 1):
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            loss = _flow_matching_step(model, batch, device, cfg.n_time_samples)
            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            running += loss.item()
            n += 1
            if cfg.log_every and it % cfg.log_every == 0:
                logging.info(f"[EquiFM] iter {it:05d} | loss={running/n:.4f}")
        tr = running / max(1, n)
        if val_loader is not None:
            va = _validate(model, val_loader, device, cfg)
            logging.info(f"[EquiFM] epoch {epoch:03d}/{cfg.epochs} | train={tr:.4f} | val={va:.4f}")
            if va < best_val and cfg.ckpt_path:
                best_val = va
                torch.save(model.state_dict(), cfg.ckpt_path)
                logging.info(f"[EquiFM] saved best checkpoint: {cfg.ckpt_path}")
        else:
            logging.info(f"[EquiFM] epoch {epoch:03d}/{cfg.epochs} | train={tr:.4f}")
        if scheduler:
            scheduler.step()


@torch.no_grad()
def _validate(model, loader, device, cfg: TrainCfg):
    model.eval()
    running = 0.0
    n = 0
    for batch in loader:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)
        loss = _flow_matching_step(model, batch, device, cfg.n_time_samples)
        running += loss.item()
        n += 1
    model.train()
    return running / max(1, n)
