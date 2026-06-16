"""Default configuration for EAGG.

The paths are relative to the repository root. The repository includes the
final inference checkpoint, the hand-cognition checkpoint/cache, and
synergy/config files for the supported end effectors. Grasp datasets and object
model libraries should be placed under ``data/`` when running training.
"""

config = {
    # Prediction mode used by the released checkpoint.
    "prediction_mode": "x",
    "loss_mode": "x",

    # Checkpoints.
    "checkpoint_path": "checkpoints/final/eagg_base.pth",
    "pretrained_hand_model_path": "checkpoints/final/eagg_hand_cognition.pth",
    "freeze_hand": True,

    # Data layout.
    "dataset_root": "data/graspit_grasps",
    "object_models_dir": "data/Object_Models",
    "cache_dir": "data/cache",
    "hand_cache_dir": "data/cache/hand_cognition",
    "custom_synergy_dir": "isaac_sim_grasping/grippers_synergy",
    "hand_config_dir": "isaac_sim_grasping/usd2urdf",

    # Representative object IDs. Replace or extend these with the objects
    # available in your local MGG-style dataset before full training.
    "target_object_id": [
        "003_cracker_box",
        "004_sugar_box",
        "005_tomato_soup_can",
        "006_mustard_bottle",
        "011_banana",
        "024_bowl",
        "025_mug",
    ],
    "train_object_ids": [
        "003_cracker_box",
        "004_sugar_box",
        "005_tomato_soup_can",
        "006_mustard_bottle",
        "011_banana",
        "024_bowl",
        "025_mug",
    ],
    "test_object_ids": [
        "003_cracker_box",
        "004_sugar_box",
        "005_tomato_soup_can",
        "006_mustard_bottle",
        "011_banana",
        "024_bowl",
        "025_mug",
    ],

    # End effector sets supported by the included hand cache and synergies.
    "supported_grippers": [
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
    ],
    "train_grippers": [
        "AbilityHand",
        "Allegro",
        "AllegroL",
        "Barrett",
        "DexHand",
        "FreedomHand",
        "franka_panda",
        "robotiq_3finger",
        "HumanHand",
        "shadow_hand",
        "wsg_50",
        "sawyer",
    ],
    "test_grippers": ["jaco_robot"],

    # Model.
    "synergy_dim": 4,
    "synergy_clip": 5.0,
    "n_points": 1024,
    "min_fall_time": 3.0,
    "position_scale": 10.0,
    "embed_dim": 256,
    "num_heads": 8,
    "depth": 8,
    "hand_node_feat_dim": 33,

    # Loss weights.
    "loss_weight_syn": 1.0,
    "loss_weight_pos": 10.0,
    "loss_weight_rot": 1.0,

    # Training defaults.
    "batch_size": 420,
    "learning_rate": 2e-4,
    "epochs": 10,
    "weight_decay": 0.0,
    "grad_clip_norm": 10.0,
    "val_ratio": 0.02,
    "use_curriculum": True,
    "ramp_up_epochs": 5,
    "t_max_start": 0.30,
    "t_max_end": 0.98,
    "data_ratio_start": 0.10,
    "data_ratio_end": 1.0,
    "save_every_epochs": 0,

    # Runtime.
    "device": "cuda",
    "seed": 42,
    "num_workers": 8,
    "log_interval": 50,
    "use_amp": True,
    "compile_model": False,
    "use_cache": True,
}
