#!/usr/bin/env python
"""Build hand-cognition cache files from bundled URDF and synergy assets."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.hand_cognition_dataset import HandCognitionDataset  # noqa: E402


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


def parse_grippers(values: list[str]) -> list[str]:
    requested: list[str] = []
    for value in values:
        requested.extend(part.strip() for part in value.split(",") if part.strip())

    if not requested or any(name.lower() == "all" for name in requested):
        return list(HAND_URDFS)

    unknown = [name for name in requested if name not in HAND_URDFS]
    if unknown:
        available = ", ".join(HAND_URDFS)
        raise SystemExit(f"Unknown gripper(s): {', '.join(unknown)}. Available: {available}")

    deduped: list[str] = []
    for name in requested:
        if name not in deduped:
            deduped.append(name)
    return deduped


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def cache_name(name: str, n_points: int, synergy_dim: int, position_scale: float) -> str:
    return f"{name}_pts{n_points}_syn{synergy_dim}_scale{int(position_scale)}_v2.pt"


def build_configs(grippers: list[str]) -> list[dict[str, str]]:
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


def relative_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build EAGG hand-cognition cache files from URDF meshes and synergy PCA files."
    )
    parser.add_argument(
        "--grippers",
        nargs="+",
        default=["all"],
        help="Names to build, comma-separated names, or 'all'.",
    )
    parser.add_argument("--cache-dir", default="data/cache/hand_cognition")
    parser.add_argument("--n-points", type=int, default=1024)
    parser.add_argument("--synergy-dim", type=int, default=4)
    parser.add_argument("--position-scale", type=float, default=10.0)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Regenerate selected caches even when matching files already exist.",
    )
    args = parser.parse_args()

    grippers = parse_grippers(args.grippers)
    cache_dir = resolve_path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    expected = {
        name: cache_dir / cache_name(name, args.n_points, args.synergy_dim, args.position_scale)
        for name in grippers
    }

    to_build = [name for name, path in expected.items() if args.rebuild or not path.exists()]
    if not to_build:
        print("[HandCognition] All requested cache files already exist:")
        for name in grippers:
            print(f"  - {name}: {relative_to_root(expected[name])}")
        return

    build_cache_dir = cache_dir
    if args.rebuild:
        build_cache_dir = cache_dir / ".rebuild_tmp"
        if build_cache_dir.exists():
            shutil.rmtree(build_cache_dir)
        build_cache_dir.mkdir(parents=True, exist_ok=True)

    print("[HandCognition] Building cache files for:", ", ".join(to_build))
    HandCognitionDataset(
        build_configs(to_build),
        n_points=args.n_points,
        samples_per_epoch=1,
        pos_scale=args.position_scale,
        synergy_dim=args.synergy_dim,
        cache_dir=str(build_cache_dir),
        augment=False,
    )

    if args.rebuild:
        for name in to_build:
            generated = build_cache_dir / cache_name(
                name, args.n_points, args.synergy_dim, args.position_scale
            )
            target = expected[name]
            if not generated.exists():
                raise FileNotFoundError(f"Expected generated cache was not created: {generated}")
            shutil.copy2(generated, target)
        shutil.rmtree(build_cache_dir)

    print("[HandCognition] Cache ready:")
    for name in grippers:
        print(f"  - {name}: {relative_to_root(expected[name])}")


if __name__ == "__main__":
    main()
