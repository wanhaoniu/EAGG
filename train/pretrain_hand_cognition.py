#!/usr/bin/env python
"""Pretrain the EAGG hand-cognition backbone from URDF/synergy assets."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.amp import GradScaler
except ImportError:  # pragma: no cover - compatibility with older PyTorch
    from torch.cuda.amp import GradScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.hand_cognition_dataset import HandCognitionDataset  # noqa: E402
from models.hand_cognition_model import HandCognitionModel  # noqa: E402


HAND_URDFS = {
    "AbilityHand": "isaac_sim_grasping/grippers/AbilityHand/AbilityHand.urdf",
    "Allegro": "isaac_sim_grasping/grippers/Allegro/Allegro.urdf",
    "AllegroL": "isaac_sim_grasping/grippers/Allegro/AllegroL.urdf",
    "Barrett": "isaac_sim_grasping/grippers/Barrett/Barrett.urdf",
    "DexHand": "isaac_sim_grasping/grippers/DexHand/DexHand.urdf",
    "FreedomHand": "isaac_sim_grasping/grippers/FreedomHand/FreedomHand.urdf",
    "HumanHand": "isaac_sim_grasping/grippers/HumanHand/HumanHand.urdf",
    "franka_panda": "isaac_sim_grasping/grippers/franka_panda/franka_panda.urdf",
    "jaco_robot": "isaac_sim_grasping/grippers/jaco_robot/jaco_robot.urdf",
    "robotiq_3finger": "isaac_sim_grasping/grippers/robotiq_3finger/robotiq_3finger.urdf",
    "sawyer": "isaac_sim_grasping/grippers/sawyer/sawyer.urdf",
    "shadow_hand": "isaac_sim_grasping/grippers/shadow_hand/shadow_hand.urdf",
    "wsg_50": "isaac_sim_grasping/grippers/wsg_50/wsg_50.urdf",
}


def parse_grippers(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names or any(name.lower() == "all" for name in names):
        return list(HAND_URDFS)
    unknown = [name for name in names if name not in HAND_URDFS]
    if unknown:
        available = ", ".join(HAND_URDFS)
        raise SystemExit(f"Unknown gripper(s): {', '.join(unknown)}. Available: {available}")
    return list(dict.fromkeys(names))


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain the EAGG hand-cognition backbone.")
    parser.add_argument("--grippers", default="all", help="Comma-separated gripper names or 'all'.")
    parser.add_argument("--out-dir", default="checkpoints/training_runs/hand_cognition")
    parser.add_argument("--cache-dir", default="data/cache/hand_cognition")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--samples-per-epoch", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-points", type=int, default=1024)
    parser.add_argument("--position-scale", type=float, default=10.0)
    parser.add_argument("--synergy-dim", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--enc-depth", type=int, default=4)
    parser.add_argument("--dec-depth", type=int, default=2)
    parser.add_argument("--gnn-layers", type=int, default=3)
    parser.add_argument("--no-gnn", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile when available.")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--smoke-test", action="store_true", help="Use a tiny run for installation checks.")
    return parser.parse_args()


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


def hand_configs(grippers: list[str]) -> list[dict[str, str]]:
    configs = []
    for name in grippers:
        urdf = ROOT / HAND_URDFS[name]
        synergy = ROOT / "isaac_sim_grasping" / "grippers_synergy" / f"{name}.pickle"
        if not urdf.exists():
            raise FileNotFoundError(f"Missing URDF for {name}: {urdf}")
        if not synergy.exists():
            raise FileNotFoundError(f"Missing synergy file for {name}: {synergy}")
        configs.append({"name": name, "urdf": str(urdf), "synergy": str(synergy)})
    return configs


def build_model(args: argparse.Namespace, device: torch.device) -> HandCognitionModel:
    feat_dim = 27 + 1 + args.synergy_dim + 1
    model = HandCognitionModel(
        synergy_dim=args.synergy_dim,
        feat_dim=feat_dim,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        enc_depth=args.enc_depth,
        dec_depth=args.dec_depth,
        use_gnn=not args.no_gnn,
        gnn_layers=args.gnn_layers,
    ).to(device)
    return model


def save_checkpoint(
    path: Path,
    model: HandCognitionModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    args: argparse.Namespace,
    grippers: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "config": vars(args),
            "grippers": grippers,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.epochs = min(args.epochs, 1)
        args.samples_per_epoch = min(args.samples_per_epoch, max(args.batch_size, 16))
        args.batch_size = min(args.batch_size, 16)
        args.num_workers = 0

    set_seed(args.seed)
    device = resolve_device(args.device)
    grippers = parse_grippers(args.grippers)
    cache_dir = resolve_path(args.cache_dir)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[init] device={device}")
    print(f"[data] grippers={grippers}")
    print(f"[data] cache={cache_dir}")
    print(f"[save] out_dir={out_dir}")

    dataset = HandCognitionDataset(
        hand_config_list=hand_configs(grippers),
        n_points=args.n_points,
        samples_per_epoch=args.samples_per_epoch,
        pos_scale=args.position_scale,
        synergy_dim=args.synergy_dim,
        cache_dir=str(cache_dir),
        augment=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    base_model = build_model(args, device)
    model = base_model
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(base_model)

    optimizer = torch.optim.AdamW(
        base_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=1e-6,
    )
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    try:
        scaler = GradScaler(device.type, enabled=use_amp)
    except TypeError:
        scaler = GradScaler(enabled=use_amp)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        count = 0
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred_cloud = model(batch)
                loss = F.mse_loss(pred_cloud, batch["posed_cloud"])

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            batch_size = batch["posed_cloud"].shape[0]
            loss_sum += float(loss.item()) * batch_size
            count += batch_size
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        scheduler.step()
        train_loss = loss_sum / max(1, count)
        print(f"[epoch {epoch + 1:03d}] train={train_loss:.6f}")

        save_checkpoint(
            out_dir / "latest_checkpoint.pth",
            base_model,
            optimizer,
            epoch,
            train_loss,
            args,
            grippers,
        )
        if train_loss < best_loss:
            best_loss = train_loss
            save_checkpoint(
                out_dir / "eagg_hand_cognition_best.pth",
                base_model,
                optimizer,
                epoch,
                train_loss,
                args,
                grippers,
            )
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                out_dir / f"eagg_hand_cognition_epoch{epoch + 1:03d}.pth",
                base_model,
                optimizer,
                epoch,
                train_loss,
                args,
                grippers,
            )

    final_path = out_dir / "eagg_hand_cognition_final.pth"
    save_checkpoint(final_path, base_model, optimizer, args.epochs - 1, best_loss, args, grippers)
    print(f"[done] final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
