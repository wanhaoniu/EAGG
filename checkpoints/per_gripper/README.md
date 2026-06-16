# Per-Gripper Checkpoints

This directory contains one generator checkpoint per bundled end effector where
a per-gripper checkpoint is available:

- `AbilityHand.pth`
- `Allegro.pth`
- `Barrett.pth`
- `DexHand.pth`
- `FreedomHand.pth`
- `HumanHand.pth`
- `franka_panda.pth`
- `jaco_robot.pth`
- `robotiq_3finger.pth`
- `sawyer.pth`
- `wsg_50.pth`

`tools/generate_gripper_gallery.py` uses these checkpoints by default in
`--checkpoint-mode per_gripper`. Use `--checkpoint-mode unified` to render with
`checkpoints/final/eagg_base.pth` instead.

