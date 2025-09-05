import argparse
import logging
import os
from typing import Dict, Any, Tuple, Optional

import yaml
import torch
from torch.utils.data import DataLoader

# Project modules
from common import seed as seed_util
from common import utils as utils
from common import registry
from data.embeddings import EmbeddingDataset
from train import trainer as train_module
from generate import runner as generate_module
from eval import evaluate_generated as eval_module


# -----------------------------
# Helpers to read nested config
# -----------------------------
def cfg_get(cfg: Dict[str, Any], path: str, default=None):
    """
    Safe get for nested keys: cfg_get(cfg, "training.batch_size", 64)
    """
    cur = cfg
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def cfg_set(cfg: Dict[str, Any], path: str, value):
    """
    Safe set for nested keys: cfg_set(cfg, "training.batch_size", 64)
    Creates missing levels as needed.
    """
    parts = path.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


# -----------------------------
# Build datasets & loaders
# -----------------------------
def make_dataset(
    csv_path: str,
    cfg: Dict[str, Any],
    atom_types=None,
    mode: str = "train",
) -> EmbeddingDataset:
    embed_dim = int(cfg_get(cfg, "model_cfg.embed_dim", 128))
    use_embeddings = bool(cfg_get(cfg, "data.use_embeddings", True))
    ignore_embeddings = bool(cfg_get(cfg, "data.ignore_embeddings", False))
    need_3d = bool(cfg_get(cfg, "data.build_3d", False))

    ds = EmbeddingDataset(
        csv_file=csv_path,
        embed_dim=embed_dim,
        use_embeddings=use_embeddings,
        ignore_embeddings=ignore_embeddings,
        atom_types=atom_types,
        need_3d=need_3d,
    )
    return ds


def make_loader(
    ds: EmbeddingDataset, cfg: Dict[str, Any], shuffle: bool
) -> DataLoader:
    batch_size = int(cfg_get(cfg, "training.batch_size", 64))
    num_workers = int(cfg_get(cfg, "data.num_workers", 4))
    pin_memory = bool(cfg_get(cfg, "data.pin_memory", True))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=ds.collate_fn(),
    )


# -----------------------------
# Model dims from dataset + cfg
# -----------------------------
def infer_model_dims(
    model_name: str, ds, cfg: Dict[str, Any]
) -> Dict[str, int]:
    """
    Derive dimensions for model construction.
    Prefers explicit config values; falls back to dataset-derived.
    """
    dims = {}
    dims["embed_dim"] = int(cfg_get(cfg, "model_cfg.embed_dim", getattr(ds, "embed_dim", 128)))
    dims["latent_dim"] = int(cfg_get(cfg, "model_cfg.latent_dim", 64))
    dims["hidden_dim"] = int(cfg_get(cfg, "model_cfg.hidden_dim", 128))
    if model_name in ("graphaf", "graphvae"):
        dims["max_nodes"] = int(cfg_get(cfg, "model_cfg.max_nodes", getattr(ds, "max_nodes", 32)))
        dims["node_feat_size"] = int(getattr(ds, "num_types", 6))
        # edges
        dims["edge_feat_size"] = int(cfg_get(cfg, "model_cfg.edge_feat_size", 1))
    if model_name == "cvae3d":
        dims["grid_size"] = int(cfg_get(cfg, "model_cfg.grid_size", 24))
        dims["num_atom_types"] = int(cfg_get(cfg, "model_cfg.num_atom_types", getattr(ds, "num_types", 5)))
    if model_name == "equifm":
        dims["num_atom_types"] = int(cfg_get(cfg, "model_cfg.num_atom_types", getattr(ds, "num_types", 5)))
    return dims


# -----------------------------
# Load & merge config + CLI
# -----------------------------
def load_and_merge_config(args) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}

    # Load YAML if provided
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}

    # Defaults if not present
    cfg.setdefault("paths", {})
    cfg.setdefault("data", {})
    cfg.setdefault("training", {})
    cfg.setdefault("model_cfg", {})
    cfg.setdefault("generation", {})

    # CLI overrides (top-level short-hands are mapped into nested)
    if args.model:
        cfg_set(cfg, "model", args.model)
    if args.train_csv:
        cfg_set(cfg, "data.train_csv", args.train_csv)
    if args.cond_csv:
        cfg_set(cfg, "data.cond_csv", args.cond_csv)
    if args.output:
        cfg_set(cfg, "paths.output", args.output)
    if args.checkpoint:
        cfg_set(cfg, "paths.checkpoint", args.checkpoint)
    if args.label:
        cfg_set(cfg, "data.label", args.label)
    if args.num_samples is not None:
        cfg_set(cfg, "generation.num_samples", args.num_samples)

    # Optional CLI hints
    if args.embed_dim is not None:
        cfg_set(cfg, "model_cfg.embed_dim", args.embed_dim)
    if args.use_embeddings is not None:
        cfg_set(cfg, "data.use_embeddings", bool(args.use_embeddings))
    if args.ignore_embeddings is not None:
        cfg_set(cfg, "data.ignore_embeddings", bool(args.ignore_embeddings))
    if args.build_3d is not None:
        cfg_set(cfg, "data.build_3d", bool(args.build_3d))

    # Training knobs (optional CLI)
    if args.batch_size is not None:
        cfg_set(cfg, "training.batch_size", args.batch_size)
    if args.epochs is not None:
        cfg_set(cfg, "training.epochs", args.epochs)
    if args.lr is not None:
        cfg_set(cfg, "training.lr", args.lr)

    # Final sanity defaults
    cfg["paths"].setdefault("checkpoint_dir", "checkpoints")
    cfg["paths"].setdefault("log_dir", "logs")
    cfg["paths"].setdefault("output_dir", "outs")
    cfg["data"].setdefault("use_embeddings", True)
    cfg["data"].setdefault("ignore_embeddings", False)
    cfg["data"].setdefault("build_3d", False)
    cfg["model_cfg"].setdefault("embed_dim", 128)

    return cfg


def main():
    parser = argparse.ArgumentParser(description="Generative Models CLI: train / generate / eval")
    parser.add_argument("--mode", choices=["train", "generate", "eval"], required=True)
    parser.add_argument("--model", choices=["graphaf", "graphvae", "cvae3d", "rga", "equifm"], required=True)
    parser.add_argument("--config", type=str, default=None)

    # Data paths
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument("--cond_csv", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)

    # Core knobs
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--use_embeddings", type=int, choices=[0, 1], default=None)
    parser.add_argument("--ignore_embeddings", type=int, choices=[0, 1], default=None)
    parser.add_argument("--build_3d", type=int, choices=[0, 1], default=None)

    # Training overrides (optional)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)

    args = parser.parse_args()

    # Logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info(f"Starting program in {args.mode} mode for model '{args.model}'")

    # Config
    cfg = load_and_merge_config(args)

    # Seed & device
    seed_value = cfg_get(cfg, "training.seed", 42)
    seed_util.seed_all(seed_value)
    logging.info(f"Using random seed: {seed_value}")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(cfg_get(cfg, "training.device", device_str))
    logging.info(f"Using device: {device}")

    model_name = cfg.get("model")

    # -------------------------
    # TRAIN
    # -------------------------
    if args.mode == "train":
        train_csv = cfg_get(cfg, "data.train_csv", None)
        if not train_csv:
            logging.error("Training CSV file path must be provided for training mode.")
            return

        # Build datasets/loaders
        need_3d = bool(cfg_get(cfg, "data.build_3d", model_name in ["cvae3d", "equifm"]))
        ds_train = make_dataset(train_csv, cfg, atom_types=None, mode="train")
        ds_val = None
        val_csv = cfg_get(cfg, "data.val_csv", None)
        if val_csv:
            ds_val = make_dataset(val_csv, cfg, atom_types=ds_train.atom_types, mode="val")

        train_loader = make_loader(ds_train, cfg, shuffle=True)
        val_loader = make_loader(ds_val, cfg, shuffle=False) if ds_val else None

        # Instantiate model (RGA has no trainable model)
        if model_name == "rga":
            logging.info("RGA is a genetic algorithm; no training needed.")
            return

        ModelClass = registry.get_model_class(model_name)
        dims = infer_model_dims(model_name, ds_train, cfg)

        if model_name == "graphaf":
            model = ModelClass(
                node_feat_size=dims["node_feat_size"],
                edge_feat_size=dims["edge_feat_size"],
                latent_dim=dims["latent_dim"],
                embed_dim=dims["embed_dim"],
                max_nodes=dims["max_nodes"],
            ).to(device)
        elif model_name == "graphvae":
            model = ModelClass(
                max_nodes=dims["max_nodes"],
                node_feat_dim=dims["node_feat_size"],
                embed_dim=dims["embed_dim"],
                latent_dim=dims["latent_dim"],
            ).to(device)
        elif model_name == "cvae3d":
            model = ModelClass(
                embed_dim=dims["embed_dim"],
                latent_dim=dims["latent_dim"],
                grid_size=dims["grid_size"],
                num_atom_types=dims["num_atom_types"],
            ).to(device)
        elif model_name == "equifm":
            model = ModelClass(
                embed_dim=dims["embed_dim"],
                num_atom_types=dims["num_atom_types"],
            ).to(device)
        else:
            logging.error(f"Unknown model type {model_name}")
            return

        # Train
        logging.info("Starting training...")
        ckpt_dir = cfg_get(cfg, "paths.checkpoint_dir", "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        # Set checkpoint file path in config for trainer
        ckpt_file = os.path.join(ckpt_dir, f"{model_name}_last.pth")
        cfg_set(cfg, "checkpoint", ckpt_file)  # Highlight: set for trainer

        # Train
        logging.info("Starting training...")
        train_module.train_model(model, train_loader, cfg, val_loader, device)
        logging.info("Training finished.")

    # -------------------------
    # GENERATE
    # -------------------------
    elif args.mode == "generate":
        cond_csv = cfg_get(cfg, "data.cond_csv", None)
        if not cond_csv:
            logging.error("Conditional CSV file path must be provided for generation mode.")
            return

        # Keep atom vocab consistent with train set if provided
        train_atom_types = None
        if cfg_get(cfg, "data.train_csv", None):
            ds_train = make_dataset(cfg_get(cfg, "data.train_csv", None), cfg, atom_types=None, mode="train")
            train_atom_types = ds_train.atom_types

        ds_cond = make_dataset(cond_csv, cfg, atom_types=train_atom_types, mode="cond")

        # RGA doesn't load a model
        model = None
        if model_name != "rga":
            ModelClass = registry.get_model_class(model_name)
            dims = infer_model_dims(model_name, ds_cond, cfg)
            if model_name == "graphaf":
                model = ModelClass(
                    node_feat_size=dims["node_feat_size"],
                    edge_feat_size=dims["edge_feat_size"],
                    latent_dim=dims["latent_dim"],
                    embed_dim=dims["embed_dim"],
                    max_nodes=dims["max_nodes"],
                ).to(device)
            elif model_name == "graphvae":
                model = ModelClass(
                    max_nodes=dims["max_nodes"],
                    node_feat_dim=dims["node_feat_size"],
                    embed_dim=dims["embed_dim"],
                    latent_dim=dims["latent_dim"],
                ).to(device)
            elif model_name == "cvae3d":
                model = ModelClass(
                    embed_dim=dims["embed_dim"],
                    latent_dim=dims["latent_dim"],
                    grid_size=dims["grid_size"],
                    num_atom_types=dims["num_atom_types"],
                ).to(device)
            elif model_name == "equifm":
                model = ModelClass(
                    embed_dim=dims["embed_dim"],
                    num_atom_types=dims["num_atom_types"],
                ).to(device)
            else:
                logging.error(f"Unknown model type {model_name}")
                return

            # Load weights if provided
            ckpt_path = cfg_get(cfg, "paths.checkpoint", None)
            if ckpt_path:
                try:
                    model.load_state_dict(torch.load(ckpt_path, map_location=device))
                    logging.info(f"Loaded model weights from {ckpt_path}")
                except Exception as e:
                    logging.error(f"Failed to load checkpoint: {e}")
            else:
                logging.warning("No checkpoint specified. Generating with untrained model may be poor.")

        # Generate
        out_dir = cfg_get(cfg, "paths.output_dir", "outs")
        os.makedirs(out_dir, exist_ok=True)
        output_file = cfg_get(cfg, "paths.output", os.path.join(out_dir, f"generated_{model_name}.csv"))
        target_label = cfg_get(cfg, "data.label", None)
        num_samples = int(cfg_get(cfg, "generation.num_samples", 100))

        logging.info(f"Generating with '{model_name}' for label: {target_label or 'ALL'}; samples: {num_samples}")
        generate_module.generate(
            model=model,
            model_name=model_name,
            cond_dataset=ds_cond,
            output_file=output_file,
            num_samples=num_samples,
            target_label=target_label,
            device=device,
        )
        logging.info(f"Generation finished. Results saved to {output_file}")

    # -------------------------
    # EVAL
    # -------------------------
    elif args.mode == "eval":
        gen_file = cfg_get(cfg, "paths.output", None)
        cond_csv = cfg_get(cfg, "data.cond_csv", None)
        if not gen_file or not cond_csv:
            logging.error("Both generated output CSV (--output) and condition CSV (--cond_csv) are required for evaluation.")
            return
        train_smiles_file = cfg_get(cfg, "data.train_smiles_file", None)
        logging.info(f"Evaluating generated molecules in {gen_file}")
        eval_module.evaluate_results(
            gen_file,
            cond_csv,
            train_smiles_file,
            cfg_get(cfg, "paths.embed_model_path", None),
        )
    else:
        logging.error(f"Unsupported mode {args.mode}")


if __name__ == "__main__":
    main()
