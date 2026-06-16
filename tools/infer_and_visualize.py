#!/usr/bin/env python
"""Run EAGG checkpoint inference and render a 3D preview."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, Iterable, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(ROOT, ".cache"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(ROOT, ".cache", "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from configs.eagg_config import config as default_config
from datasets.mgg_dataset import MGGDataset, load_object_point_cloud
from models.jgt_model_mulgripper_v2 import JGTModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EAGG inference and save a grasp preview.")
    parser.add_argument("--checkpoint", default="checkpoints/final/eagg_base.pth")
    parser.add_argument("--gripper", default="wsg_50")
    parser.add_argument("--object-id", default=None, help="Object ID loaded from data/Object_Models.")
    parser.add_argument("--point-cloud", default="demo_data/point_clouds/024_bowl.xyz")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="outputs/demo_wsg50")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-center", action="store_true", help="Do not center the input point cloud.")
    parser.add_argument("--no-igi", action="store_true", help="Disable iterative geometry injection.")
    parser.add_argument("--no-preview", action="store_true", help="Skip PNG rendering.")
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    req = (device or "cuda").lower()
    if req.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device)
    if req == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def rot6d_to_rotation_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    x_raw = rot6d[:, 0:3]
    y_raw = rot6d[:, 3:6]
    x = torch.nn.functional.normalize(x_raw, dim=-1)
    dot = torch.sum(x * y_raw, dim=-1, keepdim=True)
    y = torch.nn.functional.normalize(y_raw - dot * x, dim=-1)
    z = torch.cross(x, y, dim=-1)
    return torch.stack((x, y, z), dim=-1)


def rotation_matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    batch = matrix.shape[0]
    m00 = matrix[:, 0, 0]
    m11 = matrix[:, 1, 1]
    m22 = matrix[:, 2, 2]
    trace = m00 + m11 + m22
    q = torch.zeros((batch, 4), device=matrix.device)

    positive = trace > 0
    if positive.any():
        s = torch.sqrt(trace[positive] + 1.0) * 2
        q[positive, 0] = 0.25 * s
        q[positive, 1] = (matrix[positive, 2, 1] - matrix[positive, 1, 2]) / s
        q[positive, 2] = (matrix[positive, 0, 2] - matrix[positive, 2, 0]) / s
        q[positive, 3] = (matrix[positive, 1, 0] - matrix[positive, 0, 1]) / s

    negative = ~positive
    if negative.any():
        diag = torch.stack([m00, m11, m22], dim=1)
        choice = torch.argmax(diag, dim=1)
        for out_i, src_i in enumerate(torch.where(negative)[0]):
            mat = matrix[src_i]
            k = choice[src_i]
            if k == 0:
                s = torch.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2
                q[src_i] = torch.tensor(
                    [
                        (mat[2, 1] - mat[1, 2]) / s,
                        0.25 * s,
                        (mat[0, 1] + mat[1, 0]) / s,
                        (mat[0, 2] + mat[2, 0]) / s,
                    ],
                    device=matrix.device,
                )
            elif k == 1:
                s = torch.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2
                q[src_i] = torch.tensor(
                    [
                        (mat[0, 2] - mat[2, 0]) / s,
                        (mat[0, 1] + mat[1, 0]) / s,
                        0.25 * s,
                        (mat[1, 2] + mat[2, 1]) / s,
                    ],
                    device=matrix.device,
                )
            else:
                s = torch.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2
                q[src_i] = torch.tensor(
                    [
                        (mat[1, 0] - mat[0, 1]) / s,
                        (mat[0, 2] + mat[2, 0]) / s,
                        (mat[1, 2] + mat[2, 1]) / s,
                        0.25 * s,
                    ],
                    device=matrix.device,
                )
    return q


def rot6d_to_quaternion(rot6d: torch.Tensor) -> torch.Tensor:
    return rotation_matrix_to_quaternion(rot6d_to_rotation_matrix(rot6d))


def parse_pcd(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PCD file: {path}")
            decoded = line.decode("utf-8", errors="ignore").strip()
            header_lines.append(decoded)
            if decoded.startswith("DATA"):
                data_start = f.tell()
                break
        raw = f.read()

    header = {}
    for line in header_lines:
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        header[parts[0].upper()] = parts[1:]

    fields = header.get("FIELDS", [])
    sizes = [int(x) for x in header.get("SIZE", [])]
    types = header.get("TYPE", [])
    counts = [int(x) for x in header.get("COUNT", ["1"] * len(fields))]
    points = int(header.get("POINTS", [header.get("WIDTH", ["0"])[0]])[0])
    data_kind = header.get("DATA", ["ascii"])[0].lower()

    if data_kind == "ascii":
        arr = np.loadtxt(path, comments="#", skiprows=len(header_lines), dtype=np.float32)
        return arr[:, :3].astype(np.float32)

    dtype_fields = []
    for name, size, typ, count in zip(fields, sizes, types, counts):
        if typ == "F" and size == 4:
            dt = "<f4"
        elif typ == "F" and size == 8:
            dt = "<f8"
        elif typ == "U" and size == 4:
            dt = "<u4"
        elif typ == "I" and size == 4:
            dt = "<i4"
        else:
            raise ValueError(f"Unsupported PCD field type: {name} {typ}{size}")
        if count == 1:
            dtype_fields.append((name, dt))
        else:
            dtype_fields.append((name, dt, (count,)))
    dtype = np.dtype(dtype_fields)
    cloud = np.frombuffer(raw, dtype=dtype, count=points)
    return np.stack([cloud["x"], cloud["y"], cloud["z"]], axis=1).astype(np.float32)


def parse_ply_ascii(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        vertex_count = None
        header_done = False
        for line in f:
            line = line.strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            if line == "end_header":
                header_done = True
                break
        if not header_done or vertex_count is None:
            raise ValueError(f"Invalid ASCII PLY file: {path}")
        rows = []
        for _ in range(vertex_count):
            parts = f.readline().split()
            if len(parts) >= 3:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.asarray(rows, dtype=np.float32)


def load_point_cloud(path: str, n_points: int, center: bool, seed: int) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pcd":
        pts = parse_pcd(path)
    elif ext in {".xyz", ".txt", ".pts"}:
        pts = np.loadtxt(path, dtype=np.float32)[:, :3]
    elif ext == ".npy":
        pts = np.load(path).astype(np.float32)[:, :3]
    elif ext == ".ply":
        pts = parse_ply_ascii(path)
    else:
        raise ValueError(f"Unsupported point cloud format: {ext}")

    pts = np.asarray(pts, dtype=np.float32)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        raise ValueError(f"Point cloud is empty: {path}")
    if center:
        pts = pts - pts.mean(axis=0, keepdims=True)

    rng = np.random.default_rng(seed)
    if len(pts) > n_points:
        idx = rng.choice(len(pts), n_points, replace=False)
        pts = pts[idx]
    elif len(pts) < n_points:
        idx = rng.choice(len(pts), n_points - len(pts), replace=True)
        pts = np.concatenate([pts, pts[idx]], axis=0)
    return pts.astype(np.float32)


def build_hand_graphs_from_cache(dataset: MGGDataset, cfg: Dict, device: torch.device):
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


class EAGGInference:
    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        gripper: str | None = None,
        extra_grippers: List[str] | None = None,
    ):
        self.device = device
        self.checkpoint_path = checkpoint_path
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        cfg = default_config.copy()
        cfg.update(ckpt.get("config", {}))
        cfg["dataset_root"] = default_config["dataset_root"]
        cfg["object_models_dir"] = default_config["object_models_dir"]
        cfg["cache_dir"] = default_config["cache_dir"]
        cfg["hand_cache_dir"] = default_config["hand_cache_dir"]
        cfg["custom_synergy_dir"] = default_config["custom_synergy_dir"]
        cfg["hand_config_dir"] = default_config["hand_config_dir"]
        cfg["target_object_id"] = default_config["target_object_id"]
        self.config = cfg
        self.position_scale = cfg.get("position_scale", 10.0)

        base_grippers = list(cfg.get("train_grippers", [])) + list(cfg.get("test_grippers", []))
        requested_grippers = []
        if gripper:
            requested_grippers.append(gripper)
        if extra_grippers:
            requested_grippers.extend(extra_grippers)
        all_grippers = list(dict.fromkeys(base_grippers + requested_grippers))
        self.dataset = MGGDataset(
            dataset_root=cfg["dataset_root"],
            object_models_dir=cfg["object_models_dir"],
            grippers=all_grippers,
            target_object_id=cfg.get("train_object_ids", None),
            min_fall_time=cfg["min_fall_time"],
            synergy_dim=cfg["synergy_dim"],
            n_points=cfg["n_points"],
            use_cache=False,
            cache_dir=cfg["cache_dir"],
            normalize=True,
            augment=False,
            position_scale=self.position_scale,
            custom_synergy_dir=cfg["custom_synergy_dir"],
            hand_config_dir=cfg["hand_config_dir"],
            synergy_clip=cfg.get("synergy_clip", 5.0),
            hand_cache_dir=cfg.get("hand_cache_dir", "data/cache/hand_cognition"),
        )
        self.hand_to_id = self.dataset.hand_to_id
        self.hand_feats, self.hand_adj, self.hand_clouds = build_hand_graphs_from_cache(
            self.dataset,
            cfg,
            device,
        )

        self.gripper_stats = ckpt.get("gripper_stats", {})
        self.global_stats = ckpt.get("global_stats", None)
        if self.global_stats is None:
            self.global_stats = {
                "mean": torch.zeros(cfg["synergy_dim"], device=device),
                "std": torch.ones(cfg["synergy_dim"], device=device),
            }
        else:
            self.global_stats = {
                "mean": torch.as_tensor(self.global_stats["mean"], dtype=torch.float32, device=device),
                "std": torch.as_tensor(self.global_stats["std"], dtype=torch.float32, device=device),
            }
        for name, stats in self.gripper_stats.items():
            stats["mean"] = torch.as_tensor(stats["mean"], dtype=torch.float32, device=device)
            stats["std"] = torch.as_tensor(stats["std"], dtype=torch.float32, device=device)

        input_dim = cfg["synergy_dim"] + 3 + 6
        hand_feat_dim = 27 + 1 + cfg["synergy_dim"] + 1
        self.model = JGTModel(
            input_dim=input_dim,
            embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            depth=cfg["depth"],
            synergy_dim=cfg["synergy_dim"],
            hand_node_feat_dim=hand_feat_dim,
            pretrained_hand_model_path=None,
            freeze_hand=False,
        ).to(device)

        state_dict = {}
        for key, val in ckpt["model_state_dict"].items():
            new_key = key
            if new_key.startswith("_orig_mod.module."):
                new_key = new_key[len("_orig_mod.module.") :]
            elif new_key.startswith("module."):
                new_key = new_key[len("module.") :]
            state_dict[new_key] = val
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    def _stats_for_gripper(self, gripper_name: str):
        if gripper_name in self.gripper_stats:
            stats = self.gripper_stats[gripper_name]
            return stats["mean"], stats["std"]
        return self.global_stats["mean"], self.global_stats["std"]

    def _sample_batch(
        self,
        points: np.ndarray,
        gripper_name: str,
        batch_size: int,
        steps: int,
        use_igi: bool,
    ) -> torch.Tensor:
        cfg = self.config
        syn_dim = cfg["synergy_dim"]
        input_dim = syn_dim + 3 + 6
        hand_idx = self.hand_to_id[gripper_name]

        pts = torch.from_numpy(points * self.position_scale).float().unsqueeze(0)
        pts = pts.repeat(batch_size, 1, 1).to(self.device)
        base_feats = self.hand_feats[hand_idx].unsqueeze(0).repeat(batch_size, 1, 1)
        adj = self.hand_adj[hand_idx].unsqueeze(0).repeat(batch_size, 1, 1)
        cloud = self.hand_clouds[hand_idx].unsqueeze(0).repeat(batch_size, 1, 1)

        x = torch.randn(batch_size, input_dim, device=self.device)
        t_seq = torch.linspace(cfg.get("t_max_end", 0.98), 0.0, steps + 1, device=self.device)
        mean, std = self._stats_for_gripper(gripper_name)

        with torch.no_grad():
            for i in range(steps):
                t_now = t_seq[i]
                t_next = t_seq[i + 1]
                t_in = torch.full((batch_size,), float(t_now), device=self.device)

                w_nodes = base_feats[..., -syn_dim:]
                mean_vals = base_feats[..., -syn_dim - 1].unsqueeze(-1)
                if use_igi:
                    s_raw = x[:, :syn_dim] * std.unsqueeze(0) + mean.unsqueeze(0)
                    theta = (s_raw.unsqueeze(1) * w_nodes).sum(dim=-1, keepdim=True) + mean_vals
                else:
                    theta = mean_vals
                dynamic_feats = torch.cat([base_feats, theta], dim=-1)

                model_out = self.model(
                    x=x,
                    point_cloud=pts,
                    hand_node_feats=dynamic_feats,
                    hand_adj=adj,
                    canonical_cloud=cloud,
                    t=t_in,
                )

                mode = cfg.get("prediction_mode", "x")
                if mode == "x":
                    x0 = model_out
                elif mode == "v":
                    x0 = x - t_now * model_out
                elif mode == "epsilon":
                    x0 = (x - t_now * model_out) / (1.0 - t_now).clamp(min=1e-5)
                else:
                    raise ValueError(f"Unknown prediction_mode: {mode}")

                if t_next < 1e-5:
                    x = x0
                else:
                    noise_est = (x - (1.0 - t_now) * x0) / t_now.clamp(min=1e-5)
                    x = (1.0 - t_next) * x0 + t_next * noise_est
        return x

    def sample(self, points: np.ndarray, gripper_name: str, num_samples: int, steps: int, max_batch_size: int, use_igi: bool):
        outputs = []
        remaining = num_samples
        while remaining > 0:
            batch_size = min(max_batch_size, remaining)
            raw = self._sample_batch(points, gripper_name, batch_size, steps, use_igi)
            outputs.extend(self.decode(raw, gripper_name))
            remaining -= batch_size
        return outputs

    def decode(self, x_tensor: torch.Tensor, gripper_name: str):
        cfg = self.config
        syn_dim = cfg["synergy_dim"]
        x_np = x_tensor.detach().cpu().numpy()
        mean, std = self._stats_for_gripper(gripper_name)
        mean_np = mean.detach().cpu().numpy()
        std_np = std.detach().cpu().numpy()
        pca_model = self.dataset.synergy_models.get(gripper_name)
        if pca_model is None:
            raise ValueError(f"No synergy model found for {gripper_name}.")

        results = []
        for sample in x_np:
            sample = sample.copy()
            sample[:syn_dim] = sample[:syn_dim] * std_np + mean_np
            s_valid = sample[: pca_model.n_components]
            clip_val = cfg.get("synergy_clip", None)
            if clip_val is not None and clip_val > 0:
                s_valid = np.clip(s_valid, -float(clip_val), float(clip_val))
            dofs = pca_model.inverse_transform(s_valid)

            pos = sample[syn_dim : syn_dim + 3] / self.position_scale
            rot6d = torch.from_numpy(sample[syn_dim + 3 : syn_dim + 9]).float().unsqueeze(0)
            quat = rot6d_to_quaternion(rot6d).squeeze(0).numpy()
            results.append({"pose": np.concatenate([pos, quat]).tolist(), "dofs": dofs.tolist()})
        return results


def quat_wxyz_to_matrix(q: Iterable[float]) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def render_preview(points: np.ndarray, grasps: List[Dict], out_path: str, title: str) -> None:
    fig = plt.figure(figsize=(7, 6), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    rng = np.random.default_rng(0)
    show_n = min(len(points), 1200)
    idx = rng.choice(len(points), show_n, replace=False)
    pts = points[idx]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c="#5d7895", alpha=0.35, linewidths=0)

    positions = np.asarray([g["pose"][:3] for g in grasps], dtype=np.float32)
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=24, c="#d23b3b", depthshade=False)
    for grasp in grasps[: min(12, len(grasps))]:
        pos = np.asarray(grasp["pose"][:3], dtype=np.float32)
        rot = quat_wxyz_to_matrix(grasp["pose"][3:7])
        axis_len = 0.035
        colors = ["#d23b3b", "#2e8b57", "#4169e1"]
        for axis in range(3):
            direction = rot[:, axis] * axis_len
            ax.plot(
                [pos[0], pos[0] + direction[0]],
                [pos[1], pos[1] + direction[1]],
                [pos[2], pos[2] + direction[2]],
                color=colors[axis],
                linewidth=1.2,
            )

    all_pts = np.concatenate([points, positions], axis=0)
    center = all_pts.mean(axis=0)
    radius = np.max(np.linalg.norm(all_pts - center, axis=1))
    radius = max(float(radius), 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    handles = [
        Line2D([0], [0], marker="o", color="w", label="object point cloud", markerfacecolor="#5d7895", markersize=6),
        Line2D([0], [0], marker="o", color="w", label="generated wrist centers", markerfacecolor="#d23b3b", markersize=6),
        Line2D([0], [0], color="#d23b3b", lw=2, label="wrist x-axis"),
        Line2D([0], [0], color="#2e8b57", lw=2, label="wrist y-axis"),
        Line2D([0], [0], color="#4169e1", lw=2, label="wrist z-axis"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, frameon=True)
    ax.text2D(
        0.02,
        0.02,
        "Pose-level preview: object cloud + generated wrist frames",
        transform=ax.transAxes,
        fontsize=8,
    )
    ax.view_init(elev=24, azim=45)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = resolve_device(args.device)
    runner = EAGGInference(args.checkpoint, device=device, gripper=args.gripper)

    if args.object_id:
        pts = load_object_point_cloud(runner.config["object_models_dir"], args.object_id, runner.config["n_points"])
        cloud_label = args.object_id
    else:
        pts = load_point_cloud(
            args.point_cloud,
            n_points=runner.config["n_points"],
            center=not args.no_center,
            seed=args.seed,
        )
        cloud_label = os.path.basename(args.point_cloud)

    grasps = runner.sample(
        points=pts,
        gripper_name=args.gripper,
        num_samples=args.num_samples,
        steps=args.steps,
        max_batch_size=args.max_batch_size,
        use_igi=not args.no_igi,
    )

    json_path = os.path.join(args.out_dir, f"{args.gripper}_grasps.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "gripper": args.gripper,
                "point_cloud": cloud_label,
                "num_samples": len(grasps),
                "grasps": grasps,
            },
            f,
            indent=2,
        )

    preview_path = None
    if not args.no_preview:
        preview_path = os.path.join(args.out_dir, f"{args.gripper}_preview.png")
        render_preview(pts, grasps, preview_path, f"EAGG pose preview: {args.gripper}")

    print(f"[save] grasp json: {json_path}")
    if preview_path:
        print(f"[save] preview: {preview_path}")


if __name__ == "__main__":
    main()
