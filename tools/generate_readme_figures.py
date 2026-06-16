#!/usr/bin/env python
"""Generate the README visualization figures.

This script reuses ``generate_gripper_gallery.py`` for checkpoint inference,
proximity scoring, URDF-mesh rendering, and top-k candidate export. It then
builds the two README figures:

1. Allegro grasping all bundled demo objects.
2. All released end effectors grasping the mug object.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_OBJECTS = [
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "011_banana",
    "024_bowl",
    "025_mug",
]

DEFAULT_GRIPPERS = [
    "AbilityHand",
    "Allegro",
    "Barrett",
    "DexHand",
    "FreedomHand",
    "HumanHand",
    "franka_panda",
    "jaco_robot",
    "robotiq_3finger",
    "sawyer",
    "wsg_50",
]

DISPLAY_NAMES = {
    "003_cracker_box": "cracker box",
    "004_sugar_box": "sugar box",
    "005_tomato_soup_can": "tomato soup can",
    "006_mustard_bottle": "mustard bottle",
    "011_banana": "banana",
    "024_bowl": "bowl",
    "025_mug": "mug",
    "franka_panda": "Franka Panda",
    "jaco_robot": "Jaco Robot",
    "robotiq_3finger": "Robotiq 3F",
    "wsg_50": "WSG-50",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate README grasp figures.")
    parser.add_argument("--objects", default=",".join(DEFAULT_OBJECTS), help="Comma-separated demo object IDs.")
    parser.add_argument("--mug-object", default="025_mug", help="Object used for cross-end-effector visualization.")
    parser.add_argument("--grippers", default=",".join(DEFAULT_GRIPPERS), help="Comma-separated end-effectors.")
    parser.add_argument("--out-root", default="outputs/readme_figures")
    parser.add_argument("--figure-dir", default="assets/figures")
    parser.add_argument("--checkpoint", default="checkpoints/final/eagg_base.pth")
    parser.add_argument("--checkpoint-mode", choices=["per_gripper", "unified"], default="per_gripper")
    parser.add_argument("--per-gripper-checkpoint-dir", default="checkpoints/per_gripper")
    parser.add_argument("--allow-unified-fallback", action="store_true")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-batch-size", type=int, default=2)
    parser.add_argument("--selection", choices=["proximity", "first"], default="proximity")
    parser.add_argument("--render-mode", choices=["mesh", "point"], default="mesh")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--points", type=int, default=1400)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-hand-faces", type=int, default=0)
    parser.add_argument("--no-igi", action="store_true")
    return parser.parse_args()


def split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def object_paths(object_id: str) -> Tuple[str, str]:
    point_cloud = ROOT / "demo_data" / "point_clouds" / f"{object_id}.xyz"
    object_mesh = ROOT / "demo_data" / "object_meshes" / f"{object_id}.stl"
    if not point_cloud.exists():
        raise FileNotFoundError(point_cloud)
    if not object_mesh.exists():
        raise FileNotFoundError(object_mesh)
    return rel(point_cloud), rel(object_mesh)


def run_gallery(args: argparse.Namespace, grippers: Iterable[str], object_id: str, out_dir: Path) -> None:
    point_cloud, object_mesh = object_paths(object_id)
    command = [
        sys.executable,
        str(ROOT / "tools" / "generate_gripper_gallery.py"),
        "--checkpoint",
        args.checkpoint,
        "--checkpoint-mode",
        args.checkpoint_mode,
        "--per-gripper-checkpoint-dir",
        args.per_gripper_checkpoint_dir,
        "--point-cloud",
        point_cloud,
        "--object-mesh",
        object_mesh,
        "--out-dir",
        rel(out_dir),
        "--grippers",
        ",".join(grippers),
        "--num-samples",
        str(args.num_samples),
        "--selection",
        args.selection,
        "--steps",
        str(args.steps),
        "--max-batch-size",
        str(args.max_batch_size),
        "--device",
        args.device,
        "--points",
        str(args.points),
        "--dpi",
        str(args.dpi),
        "--render-mode",
        args.render_mode,
        "--save-candidates",
        str(args.top_k),
        "--max-hand-faces",
        str(args.max_hand_faces),
    ]
    if args.allow_unified_fallback:
        command.append("--allow-unified-fallback")
    if args.no_igi:
        command.append("--no-igi")

    print("[run]", " ".join(command))
    subprocess.run(command, cwd=str(ROOT), check=True)


def first_rank_image(out_dir: Path, gripper: str, object_id: str) -> Path:
    candidate_dir = out_dir / "candidates" / gripper
    candidates = sorted(candidate_dir.glob(f"{gripper}_{object_id}_rank01_idx*.png"))
    if candidates:
        return candidates[0]
    selected = out_dir / f"{gripper}_{object_id}.png"
    if selected.exists():
        return selected
    raise FileNotFoundError(f"No rank-1 image found for {gripper}/{object_id} under {out_dir}")


def load_gallery_index(index_path: Path) -> List[Dict[str, str]]:
    with index_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_montage(items: List[Tuple[str, Path]], title: str, out_path: Path, cols: int, dpi: int) -> None:
    rows = int(math.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.25), dpi=dpi)
    axes_arr = np.asarray(axes).reshape(rows, cols)
    for ax in axes_arr.flat:
        ax.set_axis_off()

    for ax, (label, image_path) in zip(axes_arr.flat, items):
        ax.imshow(mpimg.imread(image_path))
        ax.set_title(label, fontsize=8, pad=1.5)
        ax.set_axis_off()

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(pad=0.35)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {rel(out_path)}")


def read_result_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "json": rel(path),
        "selected_index": data.get("selected_index"),
        "selected_score": data.get("selected_score"),
        "selection": data.get("selection"),
        "ranking_metric": data.get("ranking_metric"),
        "ranked_candidates": data.get("ranked_candidates", [])[: data.get("num_samples", 0)],
        "candidate_renders": data.get("candidate_renders", []),
        "checkpoint": data.get("checkpoint"),
        "checkpoint_mode": data.get("checkpoint_mode"),
    }


def main() -> None:
    args = parse_args()
    objects = split_csv(args.objects)
    grippers = split_csv(args.grippers)
    out_root = ROOT / args.out_root
    figure_dir = ROOT / args.figure_dir

    allegro_root = out_root / "allegro_cross_object"
    allegro_items: List[Tuple[str, Path]] = []
    summary = {
        "ranking": {
            "selection": args.selection,
            "metric": "hand-to-object proximity score; lower is better",
            "num_samples": args.num_samples,
            "steps": args.steps,
            "top_k": args.top_k,
        },
        "allegro_cross_object": {},
        "mug_cross_gripper": {},
        "figures": {},
    }

    for object_id in objects:
        object_out = allegro_root / object_id
        run_gallery(args, ["Allegro"], object_id, object_out)
        image_path = first_rank_image(object_out, "Allegro", object_id)
        allegro_items.append((DISPLAY_NAMES.get(object_id, object_id), image_path))
        summary["allegro_cross_object"][object_id] = read_result_json(
            object_out / f"Allegro_{object_id}_grasps.json"
        )

    allegro_figure = figure_dir / "readme_allegro_cross_object.png"
    save_montage(
        allegro_items,
        title="Allegro generalization across objects",
        out_path=allegro_figure,
        cols=4,
        dpi=args.dpi,
    )
    summary["figures"]["allegro_cross_object"] = rel(allegro_figure)

    mug_root = out_root / "mug_cross_gripper"
    run_gallery(args, grippers, args.mug_object, mug_root)
    mug_rows = load_gallery_index(mug_root / "index.csv")
    mug_items = []
    for row in mug_rows:
        gripper = row["gripper"]
        image_path = first_rank_image(mug_root, gripper, args.mug_object)
        mug_items.append((DISPLAY_NAMES.get(gripper, gripper), image_path))
        summary["mug_cross_gripper"][gripper] = read_result_json(ROOT / row["json"])

    mug_figure = figure_dir / "readme_mug_cross_gripper.png"
    save_montage(
        mug_items,
        title="Mug generalization across end effectors",
        out_path=mug_figure,
        cols=4,
        dpi=args.dpi,
    )
    summary["figures"]["mug_cross_gripper"] = rel(mug_figure)

    summary_path = out_root / "generation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[save] {rel(summary_path)}")


if __name__ == "__main__":
    main()
