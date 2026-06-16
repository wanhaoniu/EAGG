#!/usr/bin/env python
"""Generate a WSG-50 grasp gallery over the bundled demo objects."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from generate_gripper_gallery import choose_render_grasp, load_hand_cloud
from infer_and_visualize import EAGGInference, load_point_cloud as load_inference_cloud, resolve_device
from render_wsg50_grasp_scene import add_mesh, load_object_mesh, load_point_cloud, set_equal_axes, wsg50_meshes


DEFAULT_OBJECTS = [
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "011_banana",
    "024_bowl",
    "025_mug",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a multi-object WSG-50 gallery.")
    parser.add_argument("--checkpoint", default="checkpoints/final/eagg_base.pth")
    parser.add_argument("--objects", default=",".join(DEFAULT_OBJECTS), help="Comma-separated demo object names.")
    parser.add_argument("--point-cloud-dir", default="demo_data/point_clouds")
    parser.add_argument("--object-mesh-dir", default="demo_data/object_meshes")
    parser.add_argument("--mesh-dir", default="isaac_sim_grasping/grippers/wsg_50/meshes")
    parser.add_argument("--out-dir", default="assets/visualizations/multi_object_wsg50")
    parser.add_argument("--out-name", default="multi_object_wsg50_gallery.png")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--selection", choices=["proximity", "first"], default="proximity")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--max-batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--points", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument("--no-igi", action="store_true")
    return parser.parse_args()


def parse_object_names(value: str) -> List[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise ValueError("No demo objects selected.")
    return names


def add_gallery_panel(ax, object_mesh, points: np.ndarray, grasp: dict, mesh_dir: Path, title: str) -> None:
    scene_points = [points]
    if object_mesh is not None:
        add_mesh(ax, object_mesh, (0.30, 0.55, 0.74, 0.58))
        scene_points.append(object_mesh.vertices)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1.6, c="#4f7898", alpha=0.42, linewidths=0)

    for mesh, color in wsg50_meshes(mesh_dir, grasp):
        add_mesh(ax, mesh, color)
        scene_points.append(mesh.vertices)

    set_equal_axes(ax, np.concatenate(scene_points, axis=0))
    ax.view_init(elev=24, azim=58)
    ax.set_title(title, fontsize=8, pad=1.2)
    ax.set_axis_off()


def main() -> None:
    args = parse_args()
    object_names = parse_object_names(args.objects)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    runner = EAGGInference(str(ROOT / args.checkpoint), device=device, gripper="wsg_50")
    hand_cloud = load_hand_cloud(
        ROOT / "data" / "cache" / "hand_cognition",
        "wsg_50",
        n_points=runner.config["n_points"],
        synergy_dim=runner.config["synergy_dim"],
        position_scale=runner.position_scale,
    )

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = ROOT / args.mesh_dir
    panels = []

    for i, object_name in enumerate(object_names):
        cloud_path = ROOT / args.point_cloud_dir / f"{object_name}.xyz"
        mesh_path = ROOT / args.object_mesh_dir / f"{object_name}.stl"
        if not cloud_path.exists():
            raise FileNotFoundError(f"Missing point cloud: {cloud_path}")
        if not mesh_path.exists():
            raise FileNotFoundError(f"Missing object mesh: {mesh_path}")

        torch.manual_seed(args.seed + i)
        points_for_inference = load_inference_cloud(
            str(cloud_path),
            n_points=runner.config["n_points"],
            center=True,
            seed=args.seed + i,
        )
        grasps = runner.sample(
            points=points_for_inference,
            gripper_name="wsg_50",
            num_samples=args.num_samples,
            steps=args.steps,
            max_batch_size=args.max_batch_size,
            use_igi=not args.no_igi,
        )
        if not grasps:
            raise RuntimeError(f"No grasps generated for {object_name}")

        points_for_render, center = load_point_cloud(str(cloud_path), target_points=args.points, seed=args.seed + i)
        object_mesh = load_object_mesh(str(mesh_path), center=center)
        selected_index, selected_grasp = choose_render_grasp(
            points_for_render,
            hand_cloud,
            grasps,
            mode=args.selection,
        )

        json_path = out_dir / f"wsg_50_{object_name}_grasps.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "checkpoint": args.checkpoint,
                    "gripper": "wsg_50",
                    "object_name": object_name,
                    "point_cloud": f"{args.point_cloud_dir}/{object_name}.xyz",
                    "object_mesh": f"{args.object_mesh_dir}/{object_name}.stl",
                    "selected_index": selected_index,
                    "selection": args.selection,
                    "num_samples": len(grasps),
                    "grasps": grasps,
                },
                f,
                indent=2,
            )
        panels.append((object_name, object_mesh, points_for_render, selected_grasp))
        print(f"[save] {json_path.relative_to(ROOT)}")

    cols = 4
    rows = int(np.ceil(len(panels) / cols))
    fig = plt.figure(figsize=(cols * 2.25, rows * 2.0), dpi=args.dpi)
    for i, (object_name, object_mesh, points, grasp) in enumerate(panels, start=1):
        ax = fig.add_subplot(rows, cols, i, projection="3d")
        add_gallery_panel(ax, object_mesh, points, grasp, mesh_dir, object_name)
    fig.tight_layout(pad=0.25)

    out_path = out_dir / args.out_name
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
