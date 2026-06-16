#!/usr/bin/env python
"""URDF visual-mesh utilities for grasp visualization."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import trimesh
from urdfpy import URDF


URDF_PATHS = {
    "AbilityHand": "AbilityHand/AbilityHand.urdf",
    "Allegro": "Allegro/Allegro.urdf",
    "AllegroL": "Allegro/AllegroL.urdf",
    "Barrett": "Barrett/Barrett.urdf",
    "DexHand": "DexHand/DexHand.urdf",
    "FreedomHand": "FreedomHand/FreedomHand.urdf",
    "HumanHand": "HumanHand/HumanHand.urdf",
    "franka_panda": "franka_panda/franka_panda.urdf",
    "jaco_robot": "jaco_robot/jaco_robot.urdf",
    "robotiq_3finger": "robotiq_3finger/robotiq_3finger.urdf",
    "sawyer": "sawyer/sawyer.urdf",
    "shadow_hand": "shadow_hand/shadow_hand.urdf",
    "wsg_50": "wsg_50/wsg_50.urdf",
}


def quat_wxyz_to_matrix(q: Iterable[float]) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64)
    n = np.sqrt(w * w + x * x + y * y + z * z) + 1e-12
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_to_matrix(pose: List[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quat_wxyz_to_matrix(pose[3:7])
    transform[:3, 3] = np.asarray(pose[:3], dtype=np.float64)
    return transform


class UrdfGripperMeshRenderer:
    def __init__(self, release_root: Path):
        self.release_root = Path(release_root)
        self.gripper_root = self.release_root / "isaac_sim_grasping" / "grippers"
        self.synergy_root = self.release_root / "isaac_sim_grasping" / "grippers_synergy"
        self._robots: Dict[str, URDF] = {}
        self._joint_names: Dict[str, List[str]] = {}

    def available(self, gripper: str) -> bool:
        rel = URDF_PATHS.get(gripper)
        return rel is not None and (self.gripper_root / rel).exists()

    def _load_robot(self, gripper: str) -> URDF:
        if gripper not in self._robots:
            rel = URDF_PATHS.get(gripper)
            if rel is None:
                raise KeyError(f"No URDF path registered for {gripper}")
            self._robots[gripper] = URDF.load(str(self.gripper_root / rel))
        return self._robots[gripper]

    def _load_synergy_joint_names(self, gripper: str) -> List[str]:
        if gripper not in self._joint_names:
            path = self.synergy_root / f"{gripper}.pickle"
            names: List[str] = []
            if path.exists():
                with open(path, "rb") as f:
                    data = pickle.load(f)
                raw = data.get("joint_names", [])
                names = [str(x) for x in list(raw)]
            self._joint_names[gripper] = names
        return self._joint_names[gripper]

    @staticmethod
    def _clamp_joint_value(joint, value: float) -> float:
        limit = getattr(joint, "limit", None)
        if limit is None:
            return value
        lower = getattr(limit, "lower", None)
        upper = getattr(limit, "upper", None)
        if lower is not None and np.isfinite(lower):
            value = max(value, float(lower))
        if upper is not None and np.isfinite(upper):
            value = min(value, float(upper))
        return value

    def joint_config(self, gripper: str, dofs: List[float]) -> Dict[str, float]:
        robot = self._load_robot(gripper)
        actuated = {joint.name: joint for joint in robot.actuated_joints}
        cfg: Dict[str, float] = {}

        joint_names = self._load_synergy_joint_names(gripper)
        for name, value in zip(joint_names, dofs):
            if name in actuated:
                cfg[name] = self._clamp_joint_value(actuated[name], float(value))

        for joint, value in zip(robot.actuated_joints, dofs):
            if joint.name not in cfg:
                cfg[joint.name] = self._clamp_joint_value(joint, float(value))
        return cfg

    @staticmethod
    def _limit_faces(meshes: List[trimesh.Trimesh], max_faces: int, seed: int) -> List[trimesh.Trimesh]:
        if max_faces <= 0:
            return meshes
        total = sum(len(mesh.faces) for mesh in meshes)
        if total <= max_faces:
            return meshes

        rng = np.random.default_rng(seed)
        limited = []
        for mesh in meshes:
            keep = max(16, int(round(len(mesh.faces) * max_faces / total)))
            keep = min(keep, len(mesh.faces))
            if keep >= len(mesh.faces):
                limited.append(mesh)
                continue
            idx = np.sort(rng.choice(len(mesh.faces), keep, replace=False))
            limited.append(
                trimesh.Trimesh(
                    vertices=mesh.vertices.copy(),
                    faces=mesh.faces[idx].copy(),
                    process=False,
                )
            )
        return limited

    def meshes_for_grasp(
        self,
        gripper: str,
        grasp: dict,
        max_faces: int = 0,
        seed: int = 0,
    ) -> List[trimesh.Trimesh]:
        robot = self._load_robot(gripper)
        cfg = self.joint_config(gripper, grasp.get("dofs", []))
        world_from_hand = pose_to_matrix(grasp["pose"])
        fk = robot.visual_trimesh_fk(cfg=cfg)

        meshes = []
        for mesh, hand_from_link in fk.items():
            posed = mesh.copy()
            posed.apply_transform(world_from_hand @ hand_from_link)
            meshes.append(posed)
        return self._limit_faces(meshes, max_faces=max_faces, seed=seed)

