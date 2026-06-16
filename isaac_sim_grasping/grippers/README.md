# End-Effector URDF And Mesh Files

This directory includes the main URDF file and referenced visual mesh assets
for each supported end-effector name used by the configuration and synergy
files:

- `AbilityHand/AbilityHand.urdf`
- `Allegro/Allegro.urdf`
- `Allegro/AllegroL.urdf`
- `Barrett/Barrett.urdf`
- `DexHand/DexHand.urdf`
- `FreedomHand/FreedomHand.urdf`
- `HumanHand/HumanHand.urdf`
- `franka_panda/franka_panda.urdf`
- `jaco_robot/jaco_robot.urdf`
- `robotiq_3finger/robotiq_3finger.urdf`
- `sawyer/sawyer.urdf`
- `shadow_hand/shadow_hand.urdf`
- `wsg_50/wsg_50.urdf`

The bundled inference demo uses the preprocessed hand cache in
`data/cache/hand_cognition/`. The gallery renderer uses the URDF visual meshes
and the decoded joint vectors to show the generated hand/gripper action.
