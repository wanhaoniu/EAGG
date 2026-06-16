#!/usr/bin/env python
"""Render high-DPI candidate grasp images from saved gallery JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_gripper_gallery import save_montage, save_single_render, safe_name
from render_wsg50_grasp_scene import load_object_mesh, load_point_cloud
from urdf_mesh_renderer import UrdfGripperMeshRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render high-DPI candidate images from saved grasp JSON.")
    parser.add_argument("--source-root", default="assets/figures/gripper_multi_object/by_object")
    parser.add_argument("--out-root", default="assets/figures/gripper_candidate_pool_hq")
    parser.add_argument("--objects", default="auto", help="Comma-separated object names, or 'auto'.")
    parser.add_argument("--grippers", default="auto", help="Comma-separated gripper names, or 'auto'.")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--points", type=int, default=1800)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--contact-sheet-dpi", type=int, default=150)
    parser.add_argument("--max-hand-faces", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def parse_filter(value: str) -> set[str] | None:
    if value.strip().lower() == "auto":
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def candidate_indices(data: dict, limit: int) -> List[int]:
    grasps = data.get("grasps", [])
    ordered: List[int] = []

    for item in data.get("candidate_renders", []):
        index = item.get("index")
        if isinstance(index, int) and 0 <= index < len(grasps) and index not in ordered:
            ordered.append(index)

    selected = data.get("selected_index")
    if isinstance(selected, int) and 0 <= selected < len(grasps) and selected not in ordered:
        ordered.insert(0, selected)

    for index in range(len(grasps)):
        if index not in ordered:
            ordered.append(index)
    return ordered[: max(0, limit)]


def save_contact_sheet(
    out_path: Path,
    images: Iterable[Tuple[Path, str]],
    dpi: int,
) -> None:
    images = list(images)
    if not images:
        return
    cols = min(3, len(images))
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 2.8), dpi=dpi)
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = list(getattr(axes, "flat", axes))

    for ax, (image_path, title) in zip(axes, images):
        ax.imshow(plt.imread(image_path))
        ax.set_title(title, fontsize=8, pad=2)
        ax.set_axis_off()
    for ax in axes[len(images) :]:
        ax.set_axis_off()

    fig.tight_layout(pad=0.15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    source_root = ROOT / args.source_root
    out_root = ROOT / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    object_filter = parse_filter(args.objects)
    gripper_filter = parse_filter(args.grippers)
    mesh_renderer = UrdfGripperMeshRenderer(ROOT)

    rows: List[Dict[str, str]] = []
    rank1_by_gripper: Dict[str, List[Tuple[Path, str]]] = {}

    for source_dir in sorted(source_root.glob("*_per_gripper")):
        json_files = sorted(source_dir.glob("*_grasps.json"))
        if not json_files:
            continue

        first = json.loads(json_files[0].read_text(encoding="utf-8"))
        object_name = first["object_name"]
        if object_filter is not None and object_name not in object_filter:
            continue

        points, center = load_point_cloud(str(ROOT / first["point_cloud"]), target_points=args.points, seed=args.seed)
        object_mesh_path = ROOT / first.get("object_mesh", "")
        object_mesh = load_object_mesh(str(object_mesh_path), center=center) if object_mesh_path.exists() else None

        out_dir = out_root / "by_object" / source_dir.name
        montage_geometry = {}

        for json_path in json_files:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            gripper = data["gripper"]
            if gripper_filter is not None and gripper not in gripper_filter:
                continue
            if not mesh_renderer.available(gripper):
                print(f"[skip] {gripper}: missing URDF visual mesh")
                continue

            stem = safe_name(gripper)
            grasps = data.get("grasps", [])
            for rank, grasp_index in enumerate(candidate_indices(data, args.candidates), start=1):
                grasp = grasps[grasp_index]
                hand_geometry = mesh_renderer.meshes_for_grasp(
                    gripper,
                    grasp,
                    max_faces=args.max_hand_faces,
                    seed=args.seed + rank * 1000,
                )
                candidate_png = (
                    out_dir
                    / "candidates"
                    / stem
                    / f"{stem}_{object_name}_rank{rank:02d}_idx{grasp_index:03d}.png"
                )
                save_single_render(candidate_png, object_mesh, points, hand_geometry, f"{gripper} r{rank}", args.dpi)
                rows.append(
                    {
                        "object": object_name,
                        "gripper": gripper,
                        "rank": str(rank),
                        "candidate_index": str(grasp_index),
                        "png": relative(candidate_png),
                        "source_json": relative(json_path),
                    }
                )
                if rank == 1:
                    montage_geometry[gripper] = hand_geometry
                    rank1_by_gripper.setdefault(gripper, []).append((candidate_png, object_name))
                print(f"[save] {candidate_png}")

        if montage_geometry:
            save_montage(out_dir / f"all_grippers_{object_name}.png", object_mesh, points, montage_geometry, args.dpi)

    by_gripper_dir = out_root / "by_gripper"
    for gripper, images in sorted(rank1_by_gripper.items()):
        save_contact_sheet(by_gripper_dir / f"{safe_name(gripper)}_candidate_sheet.png", images, args.contact_sheet_dpi)

    with open(out_root / "index.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["object", "gripper", "rank", "candidate_index", "png", "source_json"])
        writer.writeheader()
        writer.writerows(rows)

    with open(out_root / "README.md", "w", encoding="utf-8") as f:
        f.write("# High-DPI Grasp Candidate Pool\n\n")
        f.write("This folder contains high-DPI candidate renders generated from saved grasp JSON files.\n\n")
        f.write("- `by_object/*/candidates/<gripper>/`: ranked candidate images for manual selection.\n")
        f.write("- `by_object/*/all_grippers_*.png`: rank-1 montage for each object.\n")
        f.write("- `by_gripper/*_candidate_sheet.png`: rank-1 contact sheets grouped by end effector.\n")
        f.write("- `index.csv`: candidate image index.\n")


if __name__ == "__main__":
    main()
