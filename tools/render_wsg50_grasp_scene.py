#!/usr/bin/env python
"""Render generated WSG-50 grasps with object point cloud and gripper meshes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render WSG-50 grasp meshes from EAGG JSON output.")
    parser.add_argument("--grasp-json", default="assets/quickstart/wsg_50_grasps.json")
    parser.add_argument("--point-cloud", default="demo_data/point_clouds/024_bowl.xyz")
    parser.add_argument("--object-mesh", default="demo_data/object_meshes/024_bowl.stl")
    parser.add_argument("--mesh-dir", default="isaac_sim_grasping/grippers/wsg_50/meshes")
    parser.add_argument("--out", default="assets/quickstart/wsg_50_full_grasp_render.png")
    parser.add_argument("--num-grasps", type=int, default=4)
    parser.add_argument("--points", type=int, default=1400)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def parse_pcd(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PCD file: {path}")
            text = line.decode("utf-8", errors="ignore").strip()
            header_lines.append(text)
            if text.startswith("DATA"):
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
        data = np.loadtxt(path, skiprows=len(header_lines), dtype=np.float32)
        return data[:, :3].astype(np.float32)

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
        dtype_fields.append((name, dt) if count == 1 else (name, dt, (count,)))

    cloud = np.frombuffer(raw, dtype=np.dtype(dtype_fields), count=points)
    return np.stack([cloud["x"], cloud["y"], cloud["z"]], axis=1).astype(np.float32)


def load_point_cloud(path: str, target_points: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    ext = Path(path).suffix.lower()
    if ext == ".pcd":
        pts = parse_pcd(path)
    elif ext in {".xyz", ".txt", ".pts"}:
        pts = np.loadtxt(path, dtype=np.float32)[:, :3]
    elif ext == ".npy":
        pts = np.load(path).astype(np.float32)[:, :3]
    else:
        raise ValueError(f"Unsupported point cloud format: {ext}")

    pts = pts[np.isfinite(pts).all(axis=1)]
    center = pts.mean(axis=0)
    pts = pts - center.reshape(1, 3)
    rng = np.random.default_rng(seed)
    if len(pts) > target_points:
        pts = pts[rng.choice(len(pts), target_points, replace=False)]
    return pts.astype(np.float32), center.astype(np.float32)


def load_object_mesh(path: str, center: np.ndarray) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh = mesh.copy()
    mesh.apply_translation(-center)
    return mesh


def translate(xyz: Iterable[float]) -> np.ndarray:
    t = np.eye(4, dtype=np.float32)
    t[:3, 3] = np.asarray(xyz, dtype=np.float32)
    return t


def rotz(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    t = np.eye(4, dtype=np.float32)
    t[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return t


def quat_wxyz_to_matrix(q: Iterable[float]) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float32)
    n = np.sqrt(w * w + x * x + y * y + z * z) + 1e-8
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def pose_to_matrix(pose: List[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = quat_wxyz_to_matrix(pose[3:7])
    transform[:3, 3] = np.asarray(pose[:3], dtype=np.float32)
    return transform


def load_mesh(path: Path, scale: Tuple[float, float, float], transform: np.ndarray, color: Tuple[float, float, float, float]):
    mesh = trimesh.load(path, force="mesh")
    mesh = mesh.copy()
    mesh.vertices = mesh.vertices * np.asarray(scale, dtype=np.float32)
    mesh.apply_transform(transform)
    return mesh, color


def wsg50_meshes(mesh_dir: Path, grasp: dict):
    pose = pose_to_matrix(grasp["pose"])
    dofs = grasp.get("dofs", [0.02, 0.02])
    left_q = float(dofs[0]) if len(dofs) > 0 else -0.025
    right_q = float(dofs[1]) if len(dofs) > 1 else abs(left_q)

    base = pose
    left = pose @ translate([left_q, 0.0, 0.0])
    left_finger = left @ translate([0.0, 0.0, 0.023])
    right = pose @ rotz(np.pi) @ translate([-right_q, 0.0, 0.0])
    right_finger = right @ translate([0.0, 0.0, 0.023])

    meshes = [
        load_mesh(mesh_dir / "WSG50_110.stl", (1.0, 1.0, 1.0), base, (0.62, 0.68, 0.72, 0.62)),
        load_mesh(mesh_dir / "GUIDE_WSG50_110.stl", (0.001, 0.001, 0.001), left, (0.10, 0.13, 0.16, 0.82)),
        load_mesh(mesh_dir / "WSG-FMF.stl", (0.001, 0.001, 0.001), left_finger, (0.95, 0.39, 0.12, 0.96)),
        load_mesh(mesh_dir / "GUIDE_WSG50_110.stl", (0.001, 0.001, 0.001), right, (0.10, 0.13, 0.16, 0.82)),
        load_mesh(mesh_dir / "WSG-FMF.stl", (0.001, 0.001, 0.001), right_finger, (0.95, 0.39, 0.12, 0.96)),
    ]
    return meshes


def add_mesh(ax, mesh: trimesh.Trimesh, color: Tuple[float, float, float, float]) -> None:
    triangles = mesh.vertices[mesh.faces]
    collection = Poly3DCollection(
        triangles,
        facecolor=color,
        edgecolor=(0.12, 0.14, 0.16, 0.18),
        linewidths=0.04,
    )
    collection.set_alpha(color[3])
    ax.add_collection3d(collection)


def set_equal_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 0.05)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect([1, 1, 1])
    ax.set_proj_type("ortho")


def render(
    points: np.ndarray,
    object_mesh: trimesh.Trimesh | None,
    grasps: List[dict],
    mesh_dir: Path,
    out_path: Path,
    dpi: int,
) -> None:
    count = min(4, len(grasps))
    rows, cols = (2, 2) if count > 1 else (1, 1)
    fig = plt.figure(figsize=(9.0, 7.6), dpi=dpi)

    for i, grasp in enumerate(grasps[:count]):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        if object_mesh is not None:
            add_mesh(ax, object_mesh, (0.30, 0.55, 0.74, 0.68))
            scene_points = [object_mesh.vertices]
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2.3, c="#486f91", alpha=0.48, linewidths=0)
            scene_points = [points]

        for mesh, color in wsg50_meshes(mesh_dir, grasp):
            add_mesh(ax, mesh, color)
            scene_points.append(mesh.vertices)

        wrist = np.asarray(grasp["pose"][:3], dtype=np.float32)
        ax.scatter([wrist[0]], [wrist[1]], [wrist[2]], s=24, c="#d22f27", depthshade=False)
        set_equal_axes(ax, np.concatenate(scene_points, axis=0))
        ax.set_title(f"Generated grasp {i + 1}", fontsize=10)
        ax.view_init(elev=24, azim=58)
        ax.set_axis_off()

    fig.suptitle("EAGG quickstart: clean object mesh + rendered WSG-50 gripper meshes", fontsize=12)
    fig.tight_layout(pad=0.6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    grasp_path = ROOT / args.grasp_json
    cloud_path = ROOT / args.point_cloud
    mesh_dir = ROOT / args.mesh_dir
    object_mesh_path = ROOT / args.object_mesh if args.object_mesh else None
    out_path = ROOT / args.out

    with open(grasp_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    grasps = data["grasps"][: args.num_grasps]
    points, center = load_point_cloud(str(cloud_path), target_points=args.points, seed=args.seed)
    object_mesh = None
    if object_mesh_path is not None and object_mesh_path.exists():
        object_mesh = load_object_mesh(str(object_mesh_path), center=center)
    render(points, object_mesh, grasps, mesh_dir, out_path, args.dpi)
    print(f"[save] full grasp render: {out_path}")


if __name__ == "__main__":
    main()
