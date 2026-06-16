#!/usr/bin/env python
"""Train an EAGG generator on an MGG-style dataset.

This script uses the same core model, loss, hand cache format, and grasp
representation as the released checkpoint, while keeping logging and experiment
management focused on the training workflow.
"""

from __future__ import annotations

import argparse
import copy
import glob
import math
import os
import random
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
try:
    from torch.amp import GradScaler
except ImportError:  # pragma: no cover - compatibility with older PyTorch
    from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from configs.eagg_config import config as default_config
from datasets.mgg_dataset import MGGDataset
from losses.diffusion_loss import compute_grasp_loss
from models.jgt_model_mulgripper_v2 import JGTModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EAGG on an MGG-style dataset.")
    parser.add_argument("--data-root", default=None, help="Path to data/graspit_grasps.")
    parser.add_argument("--object-models", default=None, help="Path to data/Object_Models.")
    parser.add_argument("--out-dir", default="checkpoints/training_runs/eagg_full")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None, help="cuda, cuda:0, mps, or cpu.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--train-grippers", default=None, help="Comma-separated gripper names.")
    parser.add_argument("--train-objects", default=None, help="Comma-separated object IDs.")
    parser.add_argument("--hand-init", default=None, help="Hand-cognition checkpoint path.")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision.")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile when available.")
    parser.add_argument("--smoke-test", action="store_true", help="Use a reduced subset for a quick smoke test.")
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> Dict:
    cfg = copy.deepcopy(default_config)
    if args.data_root:
        cfg["dataset_root"] = args.data_root
    if args.object_models:
        cfg["object_models_dir"] = args.object_models
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["learning_rate"] = args.lr
    if args.device:
        cfg["device"] = args.device
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.train_grippers:
        cfg["train_grippers"] = [x.strip() for x in args.train_grippers.split(",") if x.strip()]
    if args.train_objects:
        cfg["train_object_ids"] = [x.strip() for x in args.train_objects.split(",") if x.strip()]
    if args.hand_init is not None:
        cfg["pretrained_hand_model_path"] = args.hand_init
    cfg["use_amp"] = bool(cfg.get("use_amp", True) and not args.no_amp)
    cfg["compile_model"] = bool(cfg.get("compile_model", False) or args.compile)
    cfg["smoke_test"] = bool(args.smoke_test)
    cfg["ckpt_dir"] = args.out_dir
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    requested = (requested or "cuda").lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    if requested == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def rot6d_to_rotmat(x6: torch.Tensor) -> torch.Tensor:
    a1 = x6[:, 0:3]
    a2 = x6[:, 3:6]
    b1 = nn.functional.normalize(a1, dim=1, eps=1e-6)
    b2 = a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1
    b2 = nn.functional.normalize(b2, dim=1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=2)


def build_dataset(cfg: Dict) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, MGGDataset]:
    full = MGGDataset(
        dataset_root=cfg["dataset_root"],
        object_models_dir=cfg["object_models_dir"],
        grippers=cfg["train_grippers"],
        target_object_id=cfg["train_object_ids"],
        min_fall_time=cfg["min_fall_time"],
        synergy_dim=cfg["synergy_dim"],
        n_points=cfg["n_points"],
        use_cache=cfg.get("use_cache", True),
        cache_dir=cfg["cache_dir"],
        normalize=True,
        augment=True,
        position_scale=cfg["position_scale"],
        custom_synergy_dir=cfg["custom_synergy_dir"],
        hand_config_dir=cfg["hand_config_dir"],
        synergy_clip=cfg.get("synergy_clip", 5.0),
        hand_cache_dir=cfg.get("hand_cache_dir", "data/cache/hand_cognition"),
    )
    if len(full) == 0:
        raise RuntimeError(
            "No training samples were found. Check --data-root, --object-models, "
            "--train-grippers, and --train-objects."
        )

    if cfg.get("smoke_test", False):
        keep = max(1, min(len(full), int(len(full) * 0.02)))
        full = Subset(full, list(range(keep)))

    val_size = max(1, int(len(full) * cfg.get("val_ratio", 0.02))) if len(full) > 1 else 0
    train_size = len(full) - val_size
    if val_size == 0:
        return full, full, full.dataset if isinstance(full, Subset) else full

    train_set, val_set = random_split(
        full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.get("seed", 42)),
    )
    base_dataset = full.dataset if isinstance(full, Subset) else full
    return train_set, val_set, base_dataset


def build_hand_graphs_from_cache(dataset: MGGDataset, cfg: Dict) -> Dict[int, Dict[str, torch.Tensor]]:
    cache_dir = cfg.get("hand_cache_dir", "data/cache/hand_cognition")
    graphs = {}
    for gripper_name, hand_id in dataset.hand_to_id.items():
        patterns = [
            os.path.join(
                cache_dir,
                f"{gripper_name}_pts{cfg['n_points']}_syn{cfg['synergy_dim']}_scale*_v2.pt",
            ),
            os.path.join(
                cache_dir,
                f"{gripper_name}_pts{cfg['n_points']}_syn{cfg['synergy_dim']}_scale*.pt",
            ),
        ]
        cache_path = None
        for pattern in patterns:
            hits = sorted(glob.glob(pattern))
            if hits:
                cache_path = hits[0]
                break
        if cache_path is None:
            raise FileNotFoundError(f"Missing hand cache for {gripper_name} under {cache_dir}")

        item = torch.load(cache_path, map_location="cpu", weights_only=False)
        graphs[hand_id] = {
            "node_feats": torch.as_tensor(item["node_feats"]).float(),
            "adj": torch.as_tensor(item["adj"]).float(),
            "canonical_cloud": torch.as_tensor(item["canonical_cloud"]).float(),
        }
    return graphs


def pack_hand_graphs(graphs: Dict[int, Dict[str, torch.Tensor]], device: torch.device):
    max_id = max(graphs.keys())
    max_nodes = max(g["node_feats"].shape[0] for g in graphs.values())
    feat_dim = next(iter(graphs.values()))["node_feats"].shape[1]
    n_points = next(iter(graphs.values()))["canonical_cloud"].shape[0]

    feats = torch.zeros((max_id + 1, max_nodes, feat_dim), device=device)
    adj = torch.zeros((max_id + 1, max_nodes, max_nodes), device=device)
    clouds = torch.zeros((max_id + 1, n_points, 3), device=device)
    for hand_id, graph in graphs.items():
        n = graph["node_feats"].shape[0]
        feats[hand_id, :n] = graph["node_feats"].to(device)
        adj[hand_id, :n, :n] = graph["adj"].to(device)
        clouds[hand_id] = graph["canonical_cloud"].to(device)
    return feats, adj, clouds


def build_model(cfg: Dict, device: torch.device) -> nn.Module:
    hand_init = cfg.get("pretrained_hand_model_path")
    if cfg.get("freeze_hand", True):
        if not hand_init:
            raise ValueError(
                "Frozen hand-cognition backbone requires a checkpoint. "
                "Pass --hand-init or pretrain one with train/pretrain_hand_cognition.py."
            )
        if not os.path.exists(hand_init):
            raise FileNotFoundError(
                f"Hand-cognition checkpoint not found: {hand_init}. "
                "Download the released checkpoint archive or pass --hand-init "
                "to a checkpoint trained by train/pretrain_hand_cognition.py."
            )

    input_dim = cfg["synergy_dim"] + 3 + 6
    hand_feat_dim = 27 + 1 + cfg["synergy_dim"] + 1
    model = JGTModel(
        input_dim=input_dim,
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        depth=cfg["depth"],
        synergy_dim=cfg["synergy_dim"],
        hand_node_feat_dim=hand_feat_dim,
        pretrained_hand_model_path=hand_init,
        freeze_hand=cfg.get("freeze_hand", True),
    ).to(device)
    if cfg.get("compile_model", False) and hasattr(torch, "compile"):
        model = torch.compile(model)
    return model


def make_dynamic_hand_inputs(
    x_t: torch.Tensor,
    hand_id: torch.Tensor,
    batch_mu: torch.Tensor,
    batch_sigma: torch.Tensor,
    hand_feats: torch.Tensor,
    hand_adj: torch.Tensor,
    hand_clouds: torch.Tensor,
    cfg: Dict,
):
    syn_dim = cfg["synergy_dim"]
    base_feats = hand_feats[hand_id]
    adj = hand_adj[hand_id]
    canonical = hand_clouds[hand_id]

    s_raw = x_t[:, :syn_dim] * batch_sigma + batch_mu
    w_nodes = base_feats[..., -syn_dim:]
    mean_vals = base_feats[..., -syn_dim - 1].unsqueeze(-1)
    current_joint_angles = (s_raw.unsqueeze(1) * w_nodes).sum(dim=-1, keepdim=True) + mean_vals
    dynamic_feats = torch.cat([base_feats, current_joint_angles], dim=-1)

    trans = x_t[:, syn_dim : syn_dim + 3]
    rot6d = x_t[:, syn_dim + 3 : syn_dim + 9]
    rot = rot6d_to_rotmat(rot6d)
    posed_cloud = torch.bmm(canonical.float(), rot.transpose(1, 2)) + trans.float().unsqueeze(1)
    return dynamic_feats, adj, posed_cloud


def forward_loss(
    model: nn.Module,
    batch,
    hand_feats: torch.Tensor,
    hand_adj: torch.Tensor,
    hand_clouds: torch.Tensor,
    cfg: Dict,
    device: torch.device,
    current_t_limit: float,
    synergy_weights: torch.Tensor,
):
    x0, point_cloud, hand_id, batch_mu, batch_sigma = batch
    x0 = x0.to(device)
    point_cloud = point_cloud.to(device)
    hand_id = hand_id.to(device)
    batch_mu = batch_mu.to(device)
    batch_sigma = batch_sigma.to(device)

    batch_size = x0.shape[0]
    t = torch.rand(batch_size, device=device) * current_t_limit
    anchor_count = int(batch_size * 0.1)
    if anchor_count > 0:
        t[:anchor_count] = 0.8 + torch.rand(anchor_count, device=device) * 0.18

    noise = torch.randn_like(x0)
    alpha = (1.0 - t).unsqueeze(1)
    sigma = t.unsqueeze(1)
    x_t = alpha * x0 + sigma * noise

    dyn_feats, dyn_adj, posed_cloud = make_dynamic_hand_inputs(
        x_t=x_t,
        hand_id=hand_id,
        batch_mu=batch_mu,
        batch_sigma=batch_sigma,
        hand_feats=hand_feats,
        hand_adj=hand_adj,
        hand_clouds=hand_clouds,
        cfg=cfg,
    )

    model_out = model(
        x=x_t,
        point_cloud=point_cloud,
        hand_node_feats=dyn_feats,
        hand_adj=dyn_adj,
        canonical_cloud=posed_cloud,
        t=t,
    )

    alpha_safe = alpha.clamp(min=1e-5)
    sigma_safe = sigma.clamp(min=1e-5)
    pred_mode = cfg.get("prediction_mode", "x")
    loss_mode = cfg.get("loss_mode", "x")

    if pred_mode == "x":
        pred_x = model_out
        pred_eps = (x_t - alpha * pred_x) / sigma_safe
        pred_v = pred_eps - pred_x
    elif pred_mode == "epsilon":
        pred_eps = model_out
        pred_x = (x_t - sigma * pred_eps) / alpha_safe
        pred_v = pred_eps - pred_x
    elif pred_mode == "v":
        pred_v = model_out
        pred_x = x_t - sigma * pred_v
        pred_eps = x_t + alpha * pred_v
    else:
        raise ValueError(f"Unknown prediction_mode: {pred_mode}")

    if loss_mode == "x":
        loss_input, loss_target = pred_x, x0
    elif loss_mode == "epsilon":
        loss_input, loss_target = pred_eps, noise
    elif loss_mode == "v":
        loss_input, loss_target = pred_v, noise - x0
    else:
        raise ValueError(f"Unknown loss_mode: {loss_mode}")

    time_weights = torch.exp(-2.0 * t)
    loss, stats = compute_grasp_loss(
        x_pred=loss_input,
        x_target=loss_target,
        synergy_dim=cfg["synergy_dim"],
        weights={
            "syn": cfg.get("loss_weight_syn", 1.0),
            "pos": cfg.get("loss_weight_pos", 10.0),
            "rot": cfg.get("loss_weight_rot", 1.0),
        },
        synergy_weights=synergy_weights,
        time_weights=time_weights,
    )
    return loss, stats, batch_size


def global_stats_from_dataset(dataset: MGGDataset, cfg: Dict):
    stats = getattr(dataset, "gripper_stats", {})
    if not stats:
        return {
            "mean": torch.zeros(cfg["synergy_dim"]),
            "std": torch.ones(cfg["synergy_dim"]),
        }
    means = [torch.as_tensor(v["mean"]).float() for v in stats.values()]
    stds = [torch.as_tensor(v["std"]).float() for v in stats.values()]
    return {"mean": torch.stack(means).mean(0), "std": torch.stack(stds).mean(0)}


def save_checkpoint(path: str, model: nn.Module, optimizer, epoch: int, best_loss: float, dataset: MGGDataset, cfg: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model_state = model.state_dict()
    torch.save(
        {
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_loss,
            "config": cfg,
            "hand_to_id_train": dataset.hand_to_id,
            "gripper_stats": getattr(dataset, "gripper_stats", {}),
            "global_stats": global_stats_from_dataset(dataset, cfg),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    cfg = make_config(args)
    set_seed(cfg.get("seed", 42))
    device = resolve_device(cfg.get("device", "cuda"))
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    print(f"[init] device={device}")
    print(f"[init] output={cfg['ckpt_dir']}")
    print(f"[data] grippers={cfg['train_grippers']}")
    print(f"[data] objects={len(cfg['train_object_ids'])}")

    train_set, val_set, base_dataset = build_dataset(cfg)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 8),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 8),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    graphs = build_hand_graphs_from_cache(base_dataset, cfg)
    hand_feats, hand_adj, hand_clouds = pack_hand_graphs(graphs, device)
    model = build_model(cfg, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, cfg["epochs"]),
        eta_min=1e-6,
    )

    use_amp = bool(cfg.get("use_amp", True) and device.type == "cuda")
    try:
        scaler = GradScaler(device.type, enabled=use_amp)
    except TypeError:
        scaler = GradScaler(enabled=use_amp)
    decay = 0.8
    synergy_weights = torch.tensor(
        [decay**i for i in range(cfg["synergy_dim"])],
        dtype=torch.float32,
        device=device,
    )
    synergy_weights = synergy_weights / synergy_weights.mean()

    best_val = float("inf")
    global_step = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        if cfg.get("use_curriculum", True):
            ramp = max(1, cfg.get("ramp_up_epochs", 5))
            progress = min(1.0, epoch / ramp)
        else:
            progress = 1.0
        t_limit = cfg.get("t_max_start", 0.3) + progress * (
            cfg.get("t_max_end", 0.98) - cfg.get("t_max_start", 0.3)
        )

        train_sum = 0.0
        train_count = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg['epochs']}")
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss, stats, batch_size = forward_loss(
                    model,
                    batch,
                    hand_feats,
                    hand_adj,
                    hand_clouds,
                    cfg,
                    device,
                    t_limit,
                    synergy_weights,
                )
            scaler.scale(loss).backward()
            if cfg.get("grad_clip_norm", None) is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            train_sum += loss.item() * batch_size
            train_count += batch_size
            if global_step % cfg.get("log_interval", 50) == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}", pos=f"{stats['mse_pos']:.4f}")

        train_loss = train_sum / max(1, train_count)
        scheduler.step()

        model.eval()
        val_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    loss, _, batch_size = forward_loss(
                        model,
                        batch,
                        hand_feats,
                        hand_adj,
                        hand_clouds,
                        cfg,
                        device,
                        cfg.get("t_max_end", 0.98),
                        synergy_weights,
                    )
                val_sum += loss.item() * batch_size
                val_count += batch_size
        val_loss = val_sum / max(1, val_count)
        print(
            f"[epoch {epoch + 1:03d}] train={train_loss:.6f} "
            f"val={val_loss:.6f} lr={optimizer.param_groups[0]['lr']:.3e}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(cfg["ckpt_dir"], f"eagg_best_epoch{epoch + 1:03d}.pth")
            save_checkpoint(best_path, model, optimizer, epoch, best_val, base_dataset, cfg)
            print(f"[save] {best_path}")

        save_every = int(cfg.get("save_every_epochs", 0))
        if save_every > 0 and (epoch + 1) % save_every == 0:
            path = os.path.join(cfg["ckpt_dir"], f"eagg_epoch{epoch + 1:03d}.pth")
            save_checkpoint(path, model, optimizer, epoch, best_val, base_dataset, cfg)

    final_path = os.path.join(cfg["ckpt_dir"], "eagg_final.pth")
    save_checkpoint(final_path, model, optimizer, cfg["epochs"] - 1, best_val, base_dataset, cfg)
    print(f"[done] final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
