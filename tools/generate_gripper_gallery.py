#!/usr/bin/env python
"""Generate grasp visualizations for every bundled end-effector cache."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from infer_and_visualize import EAGGInference, load_point_cloud as load_inference_cloud, resolve_device
from render_wsg50_grasp_scene import load_object_mesh, load_point_cloud, pose_to_matrix, set_equal_axes
from urdf_mesh_renderer import UrdfGripperMeshRenderer


PREFERRED_GRIPPER_ORDER = [
    "AbilityHand",
    "Allegro",
    "AllegroL",
    "Barrett",
    "DexHand",
    "FreedomHand",
    "HumanHand",
    "franka_panda",
    "jaco_robot",
    "robotiq_3finger",
    "sawyer",
    "shadow_hand",
    "wsg_50",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a grasp gallery for bundled end effectors.")
    parser.add_argument("--checkpoint", default="checkpoints/final/eagg_base.pth")
    parser.add_argument("--checkpoint-mode", choices=["per_gripper", "unified"], default="per_gripper")
    parser.add_argument("--per-gripper-checkpoint-dir", default="checkpoints/per_gripper")
    parser.add_argument("--allow-unified-fallback", action="store_true")
    parser.add_argument("--point-cloud", default="demo_data/point_clouds/024_bowl.xyz")
    parser.add_argument("--object-mesh", default="demo_data/object_meshes/024_bowl.stl")
    parser.add_argument("--out-dir", default="assets/gripper_gallery/024_bowl_per_gripper")
    parser.add_argument(
        "--grippers",
        default="auto",
        help="Comma-separated gripper names, or 'auto' to use all bundled hand-cache files.",
    )
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--selection", choices=["proximity", "first"], default="proximity")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--max-batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--points", type=int, default=1400)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--max-hand-faces", type=int, default=0, help="Optional per-hand face cap; 0 keeps full URDF meshes.")
    parser.add_argument("--render-mode", choices=["mesh", "point"], default="mesh")
    parser.add_argument("--save-candidates", type=int, default=0, help="Save the top-K ranked candidate renders for manual selection.")
    parser.add_argument("--no-igi", action="store_true")
    return parser.parse_args()


def discover_cached_grippers(cache_dir: Path) -> List[str]:
    pattern = re.compile(r"(.+)_pts\d+_syn\d+_scale\d+_v\d+\.pt$")
    names = []
    for path in sorted(cache_dir.glob("*_pts*_syn*_scale*_v*.pt")):
        match = pattern.match(path.name)
        if match:
            names.append(match.group(1))
    preferred = [name for name in PREFERRED_GRIPPER_ORDER if name in names]
    remaining = sorted(name for name in names if name not in preferred)
    return preferred + remaining


def discover_per_gripper_checkpoints(checkpoint_dir: Path) -> Dict[str, Path]:
    if not checkpoint_dir.exists():
        return {}
    return {path.stem: path for path in sorted(checkpoint_dir.glob("*.pth"))}


def ordered_grippers(names: Iterable[str]) -> List[str]:
    names = list(dict.fromkeys(names))
    preferred = [name for name in PREFERRED_GRIPPER_ORDER if name in names]
    remaining = sorted(name for name in names if name not in preferred)
    return preferred + remaining


def parse_grippers(
    value: str,
    cache_dir: Path,
    checkpoint_mode: str,
    per_gripper_checkpoints: Dict[str, Path],
    allow_unified_fallback: bool,
) -> List[str]:
    cached = set(discover_cached_grippers(cache_dir))
    if value.strip().lower() == "auto":
        if checkpoint_mode == "per_gripper":
            grippers = ordered_grippers(cached.intersection(per_gripper_checkpoints.keys()))
        else:
            grippers = ordered_grippers(cached)
    else:
        grippers = [g.strip() for g in value.split(",") if g.strip()]
        missing_cache = [g for g in grippers if g not in cached]
        if missing_cache:
            raise ValueError(f"Missing hand cache for requested grippers: {missing_cache}")
        if checkpoint_mode == "per_gripper" and not allow_unified_fallback:
            missing_ckpt = [g for g in grippers if g not in per_gripper_checkpoints]
            if missing_ckpt:
                raise ValueError(
                    "Missing per-gripper checkpoints for "
                    f"{missing_ckpt}. Use --checkpoint-mode unified or --allow-unified-fallback if desired."
                )
    if not grippers:
        raise ValueError("No grippers selected.")
    return grippers


def load_hand_cloud(cache_dir: Path, gripper: str, n_points: int, synergy_dim: int, position_scale: float) -> np.ndarray:
    pattern = f"{gripper}_pts{n_points}_syn{synergy_dim}_scale*_v*.pt"
    hits = sorted(cache_dir.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"Missing hand cache for {gripper} under {cache_dir}")
    item = torch.load(hits[0], map_location="cpu", weights_only=False)
    cloud = np.asarray(item["canonical_cloud"], dtype=np.float32)
    if cloud.ndim == 3:
        cloud = cloud[0]
    return cloud / float(position_scale)


def transform_cloud(local_cloud: np.ndarray, pose: List[float]) -> np.ndarray:
    transform = pose_to_matrix(pose)
    return local_cloud @ transform[:3, :3].T + transform[:3, 3]


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def object_name_from_path(path: str) -> str:
    return safe_name(Path(path).stem)


def score_grasp(points: np.ndarray, local_hand_cloud: np.ndarray, grasp: dict) -> float:
    world_cloud = transform_cloud(local_hand_cloud, grasp["pose"])
    distances = cKDTree(points).query(world_cloud, k=1)[0]
    p05, p25 = np.percentile(distances, [5, 25])
    center_distance = np.linalg.norm(world_cloud.mean(axis=0) - points.mean(axis=0))
    return float(p05 + 0.35 * p25 + 0.05 * center_distance)


def choose_render_grasp(points: np.ndarray, local_hand_cloud: np.ndarray, grasps: List[dict], mode: str) -> Tuple[int, dict]:
    if mode == "first" or len(grasps) == 1:
        return 0, grasps[0]
    scores = [score_grasp(points, local_hand_cloud, grasp) for grasp in grasps]
    index = int(np.argmin(scores))
    return index, grasps[index]


def rank_render_grasps(points: np.ndarray, local_hand_cloud: np.ndarray, grasps: List[dict], mode: str) -> List[Tuple[int, dict, float]]:
    if mode == "first" or len(grasps) == 1:
        return [(i, grasp, float(i)) for i, grasp in enumerate(grasps)]
    scored = [(i, grasp, score_grasp(points, local_hand_cloud, grasp)) for i, grasp in enumerate(grasps)]
    return sorted(scored, key=lambda item: item[2])


def add_render_mesh(ax, mesh, color: Tuple[float, float, float, float]) -> None:
    if len(mesh.faces) == 0:
        return
    triangles = mesh.vertices[mesh.faces]
    collection = Poly3DCollection(
        triangles,
        facecolor=color,
        edgecolor=(0.10, 0.12, 0.14, 0.08),
        linewidths=0.015,
    )
    collection.set_alpha(color[3])
    ax.add_collection3d(collection)


def render_panel(
    ax,
    object_mesh,
    points: np.ndarray,
    hand_geometry,
    gripper: str,
    view: Tuple[float, float] = (24.0, 58.0),
) -> None:
    scene_points = [points]
    if object_mesh is not None:
        add_render_mesh(ax, object_mesh, (0.30, 0.55, 0.74, 0.48))
        scene_points.append(object_mesh.vertices)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2.0, c="#486f91", alpha=0.42, linewidths=0)

    if isinstance(hand_geometry, list):
        for mesh in hand_geometry:
            add_render_mesh(ax, mesh, (0.95, 0.36, 0.10, 0.94))
            scene_points.append(mesh.vertices)
    else:
        hand_cloud = hand_geometry
        ax.scatter(
            hand_cloud[:, 0],
            hand_cloud[:, 1],
            hand_cloud[:, 2],
            s=2.6,
            c="#e25b22",
            alpha=0.76,
            linewidths=0,
            depthshade=False,
        )
        scene_points.append(hand_cloud)

    set_equal_axes(ax, np.concatenate(scene_points, axis=0))
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_title(gripper, fontsize=8, pad=1.5)
    ax.set_axis_off()


def save_single_render(
    out_path: Path,
    object_mesh,
    points: np.ndarray,
    hand_geometry,
    gripper: str,
    dpi: int,
) -> None:
    views = [(24.0, 58.0), (18.0, -122.0)]
    fig = plt.figure(figsize=(7.2, 3.4), dpi=dpi)
    for i, view in enumerate(views, start=1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        render_panel(ax, object_mesh, points, hand_geometry, gripper, view=view)
    fig.tight_layout(pad=0.08)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_montage(
    out_path: Path,
    object_mesh,
    points: np.ndarray,
    rendered: Dict[str, object],
    dpi: int,
) -> None:
    cols = 4
    rows = int(np.ceil(len(rendered) / cols))
    fig = plt.figure(figsize=(cols * 3.0, rows * 2.6), dpi=dpi)
    for i, (gripper, hand_geometry) in enumerate(rendered.items(), start=1):
        ax = fig.add_subplot(rows, cols, i, projection="3d")
        render_panel(ax, object_mesh, points, hand_geometry, gripper)
    fig.tight_layout(pad=0.35)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_hand_geometry(
    args: argparse.Namespace,
    mesh_renderer: UrdfGripperMeshRenderer,
    gripper: str,
    grasp: dict,
    local_hand_cloud: np.ndarray,
    seed: int,
):
    hand_geometry = None
    if args.render_mode == "mesh" and mesh_renderer.available(gripper):
        try:
            hand_geometry = mesh_renderer.meshes_for_grasp(
                gripper,
                grasp,
                max_faces=args.max_hand_faces,
                seed=seed,
            )
        except Exception as exc:
            print(f"[warn] URDF mesh render failed for {gripper}: {exc}. Falling back to hand cache points.")

    if hand_geometry is None:
        hand_geometry = transform_cloud(local_hand_cloud, grasp["pose"])
    return hand_geometry


def build_runner_for_gripper(
    args: argparse.Namespace,
    device: torch.device,
    gripper: str,
    per_gripper_checkpoints: Dict[str, Path],
):
    if args.checkpoint_mode == "per_gripper" and gripper in per_gripper_checkpoints:
        checkpoint_path = per_gripper_checkpoints[gripper]
        return EAGGInference(str(checkpoint_path), device=device, gripper=gripper), checkpoint_path.relative_to(ROOT).as_posix()

    checkpoint_path = ROOT / args.checkpoint
    return EAGGInference(str(checkpoint_path), device=device, gripper=gripper), args.checkpoint


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cache_dir = ROOT / "data" / "cache" / "hand_cognition"
    per_gripper_checkpoints = discover_per_gripper_checkpoints(ROOT / args.per_gripper_checkpoint_dir)
    grippers = parse_grippers(
        args.grippers,
        cache_dir,
        checkpoint_mode=args.checkpoint_mode,
        per_gripper_checkpoints=per_gripper_checkpoints,
        allow_unified_fallback=args.allow_unified_fallback,
    )
    device = resolve_device(args.device)

    points_for_render, center = load_point_cloud(str(ROOT / args.point_cloud), target_points=args.points, seed=args.seed)
    object_mesh_path = ROOT / args.object_mesh if args.object_mesh else None
    object_mesh = load_object_mesh(str(object_mesh_path), center=center) if object_mesh_path and object_mesh_path.exists() else None
    object_name = object_name_from_path(args.point_cloud)
    mesh_renderer = UrdfGripperMeshRenderer(ROOT)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered_geometry: Dict[str, object] = {}
    index_rows = []
    point_cache: Dict[int, np.ndarray] = {}

    for i, gripper in enumerate(grippers):
        torch.manual_seed(args.seed + i)
        runner, checkpoint_label = build_runner_for_gripper(
            args,
            device,
            gripper,
            per_gripper_checkpoints,
        )
        n_points = runner.config["n_points"]
        if n_points not in point_cache:
            point_cache[n_points] = load_inference_cloud(
                str(ROOT / args.point_cloud),
                n_points=n_points,
                center=True,
                seed=args.seed,
            )
        points_for_inference = point_cache[n_points]

        grasps = runner.sample(
            points=points_for_inference,
            gripper_name=gripper,
            num_samples=args.num_samples,
            steps=args.steps,
            max_batch_size=args.max_batch_size,
            use_igi=not args.no_igi,
        )
        if not grasps:
            raise RuntimeError(f"No grasps generated for {gripper}")

        local_hand_cloud = load_hand_cloud(
            cache_dir,
            gripper,
            n_points=runner.config["n_points"],
            synergy_dim=runner.config["synergy_dim"],
            position_scale=runner.position_scale,
        )
        ranked_grasps = rank_render_grasps(
            points_for_render,
            local_hand_cloud,
            grasps,
            mode=args.selection,
        )
        ranked_candidate_rows = [
            {
                "rank": rank,
                "index": candidate_index,
                "score": candidate_score,
            }
            for rank, (candidate_index, _candidate_grasp, candidate_score) in enumerate(ranked_grasps, start=1)
        ]
        selected_index, selected_grasp, selected_score = ranked_grasps[0]
        hand_geometry = build_hand_geometry(
            args,
            mesh_renderer,
            gripper,
            selected_grasp,
            local_hand_cloud,
            seed=args.seed + i,
        )
        rendered_geometry[gripper] = hand_geometry

        stem = safe_name(gripper)
        json_path = out_dir / f"{stem}_{object_name}_grasps.json"
        png_path = out_dir / f"{stem}_{object_name}.png"
        candidate_renders = []
        if args.save_candidates > 0:
            candidate_dir = out_dir / "candidates" / stem
            candidate_dir.mkdir(parents=True, exist_ok=True)
            for rank, (candidate_index, candidate_grasp, candidate_score) in enumerate(
                ranked_grasps[: args.save_candidates],
                start=1,
            ):
                candidate_geometry = hand_geometry if rank == 1 else build_hand_geometry(
                    args,
                    mesh_renderer,
                    gripper,
                    candidate_grasp,
                    local_hand_cloud,
                    seed=args.seed + i + rank * 1000,
                )
                candidate_png = candidate_dir / f"{stem}_{object_name}_rank{rank:02d}_idx{candidate_index:03d}.png"
                save_single_render(candidate_png, object_mesh, points_for_render, candidate_geometry, gripper, args.dpi)
                candidate_renders.append(
                    {
                        "rank": rank,
                        "index": candidate_index,
                        "score": candidate_score,
                        "png": candidate_png.relative_to(ROOT).as_posix(),
                    }
                )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "checkpoint": checkpoint_label,
                    "checkpoint_mode": args.checkpoint_mode,
                    "gripper": gripper,
                    "object_name": object_name,
                    "point_cloud": args.point_cloud,
                    "object_mesh": args.object_mesh,
                    "render_mode": "urdf_mesh" if isinstance(hand_geometry, list) else "hand_cache_points",
                    "selected_index": selected_index,
                    "selected_score": selected_score,
                    "selection": args.selection,
                    "ranking_metric": "hand-to-object proximity score; lower is better",
                    "ranked_candidates": ranked_candidate_rows,
                    "num_samples": len(grasps),
                    "candidate_renders": candidate_renders,
                    "grasps": grasps,
                },
                f,
                indent=2,
            )
        save_single_render(png_path, object_mesh, points_for_render, hand_geometry, gripper, args.dpi)
        index_rows.append((gripper, png_path.relative_to(ROOT).as_posix(), json_path.relative_to(ROOT).as_posix()))
        print(f"[save] {gripper}: {png_path}")
        del runner
        if device.type == "cuda":
            torch.cuda.empty_cache()

    montage_path = out_dir / f"all_grippers_{object_name}.png"
    save_montage(montage_path, object_mesh, points_for_render, rendered_geometry, args.dpi)
    with open(out_dir / "index.csv", "w", encoding="utf-8") as f:
        f.write("gripper,png,json\n")
        for row in index_rows:
            f.write(",".join(row) + "\n")
    print(f"[save] montage: {montage_path}")


if __name__ == "__main__":
    main()
