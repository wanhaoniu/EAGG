#!/usr/bin/env python
"""Generate per-gripper grasp figures over all bundled demo objects."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run per-gripper galleries over several demo objects.")
    parser.add_argument("--objects", default=",".join(DEFAULT_OBJECTS), help="Comma-separated demo object names.")
    parser.add_argument("--out-root", default="assets/figures/gripper_multi_object")
    parser.add_argument("--checkpoint-mode", choices=["per_gripper", "unified"], default="per_gripper")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--max-batch-size", type=int, default=2)
    parser.add_argument("--selection", choices=["proximity", "first"], default="proximity")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dpi", type=int, default=145)
    parser.add_argument("--render-mode", choices=["mesh", "point"], default="mesh")
    parser.add_argument("--save-candidates", type=int, default=0, help="Save top-K candidate renders for each object/gripper pair.")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def parse_objects(value: str) -> List[str]:
    objects = [item.strip() for item in value.split(",") if item.strip()]
    if not objects:
        raise ValueError("No objects selected.")
    return objects


def run_object_gallery(args: argparse.Namespace, object_name: str, out_dir: Path) -> None:
    point_cloud = ROOT / "demo_data" / "point_clouds" / f"{object_name}.xyz"
    object_mesh = ROOT / "demo_data" / "object_meshes" / f"{object_name}.stl"
    if not point_cloud.exists():
        raise FileNotFoundError(point_cloud)
    if not object_mesh.exists():
        raise FileNotFoundError(object_mesh)

    montage = out_dir / f"all_grippers_{object_name}.png"
    index = out_dir / "index.csv"
    if args.skip_existing and montage.exists() and index.exists():
        print(f"[skip] {object_name}: {out_dir.relative_to(ROOT)}")
        return

    command = [
        sys.executable,
        str(ROOT / "tools" / "generate_gripper_gallery.py"),
        "--checkpoint-mode",
        args.checkpoint_mode,
        "--point-cloud",
        f"demo_data/point_clouds/{object_name}.xyz",
        "--object-mesh",
        f"demo_data/object_meshes/{object_name}.stl",
        "--render-mode",
        args.render_mode,
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
        "--dpi",
        str(args.dpi),
        "--save-candidates",
        str(args.save_candidates),
        "--out-dir",
        out_dir.relative_to(ROOT).as_posix(),
    ]
    print(f"[run] {object_name}: {' '.join(command)}")
    subprocess.run(command, cwd=str(ROOT), check=True)


def read_index(index_path: Path, object_name: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(index_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = dict(row)
            row["object"] = object_name
            rows.append(row)
    return rows


def save_by_gripper_montages(rows: List[Dict[str, str]], out_dir: Path, dpi: int) -> None:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["gripper"], []).append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    for gripper, items in sorted(grouped.items()):
        cols = 2
        rows_count = int(np.ceil(len(items) / cols))
        fig, axes = plt.subplots(rows_count, cols, figsize=(cols * 3.8, rows_count * 1.9), dpi=dpi)
        axes_arr = np.asarray(axes).reshape(rows_count, cols)
        for ax in axes_arr.flat:
            ax.set_axis_off()

        for ax, item in zip(axes_arr.flat, items):
            image_path = ROOT / item["png"]
            img = mpimg.imread(image_path)
            ax.imshow(img)
            ax.set_title(item["object"], fontsize=8, pad=1.5)
            ax.set_axis_off()

        fig.suptitle(gripper, fontsize=11)
        fig.tight_layout(pad=0.25)
        out_path = out_dir / f"{gripper}_multi_object.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[save] {out_path.relative_to(ROOT)}")


def write_global_index(rows: List[Dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["object", "gripper", "png", "json"])
        writer.writeheader()
        writer.writerows(rows)


def write_readme(out_root: Path, objects: List[str], rows: List[Dict[str, str]], save_candidates: int) -> None:
    grippers = sorted({row["gripper"] for row in rows})
    text = [
        "# Multi-Object Per-Gripper Figures",
        "",
        "This folder contains generated grasp visualizations for each bundled per-gripper checkpoint over multiple clean demo objects.",
        "",
        "Objects:",
        "",
    ]
    text.extend(f"- `{name}`" for name in objects)
    text.extend(
        [
            "",
            "Grippers:",
            "",
        ]
    )
    text.extend(f"- `{name}`" for name in grippers)
    text.extend(
        [
            "",
            "Subfolders:",
            "",
            "- `by_object/`: one all-gripper gallery plus per-gripper PNG/JSON files for each object.",
            "- `by_gripper/`: one multi-object montage for each hand or gripper.",
            "- `by_object/*/candidates/`: ranked candidate renders for manual selection, when `--save-candidates` is greater than zero.",
            "- `index.csv`: global index of generated PNG and JSON files.",
            "",
            f"Candidates saved per object/gripper pair: `{save_candidates}`.",
            "",
            "Regenerate with:",
            "",
            "```bash",
            "python tools/generate_all_gripper_object_figures.py --device cuda",
            "```",
            "",
        ]
    )
    (out_root / "README.md").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    args = parse_args()
    objects = parse_objects(args.objects)
    out_root = ROOT / args.out_root
    by_object = out_root / "by_object"
    by_gripper = out_root / "by_gripper"
    by_object.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, str]] = []
    for object_name in objects:
        object_out_dir = by_object / f"{object_name}_per_gripper"
        run_object_gallery(args, object_name, object_out_dir)
        all_rows.extend(read_index(object_out_dir / "index.csv", object_name))

    write_global_index(all_rows, out_root / "index.csv")
    save_by_gripper_montages(all_rows, by_gripper, args.dpi)
    write_readme(out_root, objects, all_rows, args.save_candidates)
    print(f"[save] {out_root.relative_to(ROOT) / 'index.csv'}")
    print(f"[save] {out_root.relative_to(ROOT) / 'README.md'}")


if __name__ == "__main__":
    main()
