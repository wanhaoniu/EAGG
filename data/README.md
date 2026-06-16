# Data Layout

The hand-cognition cache is stored under `data/cache/hand_cognition/`. The
`*_pts1024_syn4_scale10_v2.pt` files are required by inference, visualization,
and training; they are derived from the bundled URDF visual meshes and synergy
PCA files. The release includes them, but missing or edited caches can be
rebuilt from the repository root with:

```bash
python tools/build_hand_cognition_cache.py --grippers all
```

Place the MGG grasp data and object models in the following layout before
running the training script:

```text
EAGG_open_source/
  data/
    graspit_grasps/
      Allegro/
        Allegro-003_cracker_box.json
        ...
      franka_panda/
        franka_panda-024_bowl.json
        ...
    Object_Models/
      024_bowl/
        points.xyz
        meshes/
          model.obj
```

If the downloaded MGG archive already contains `graspit_grasps/` and
`Object_Models/`, replace `MGG_ROOT` with the extracted MGG root directory and
copy those two directories into this `data/` directory:

```bash
cp -r MGG_ROOT/graspit_grasps ./
cp -r MGG_ROOT/Object_Models ./
```

The dataset loader first looks for `points.xyz` or
`points_sampled_<N>.xyz` under each object folder. If those files are absent,
it samples points from mesh files under `meshes/` with extensions `.obj`,
`.stl`, `.dae`, or `.ply`.

Each grasp JSON is expected to contain:

```json
{
  "object_id": "024_bowl",
  "pose": [[x, y, z, qw, qx, qy, qz]],
  "final_dofs": [[...]],
  "fall_time": [5.0]
}
```

The inference demo can run on the included point cloud in
`demo_data/point_clouds/` and does not require the full dataset.
