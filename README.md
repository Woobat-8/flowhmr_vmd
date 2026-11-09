# FlowHMR VMD Export Fork

For the full upstream project background, setup, and original documentation, see the original FlowHMR README:
https://github.com/flowhmr/flowhmr/blob/main/README.md

This repository is a fork of FlowHMR focused on adding a VMD export backend for standard MMD humanoid animation output.

## What this fork adds

This project keeps the original FlowHMR motion reconstruction pipeline and adds a separate export path for MikuMikuDance-compatible VMD files.

The scope of the modifications is intentionally narrow:

- preserve the existing FlowHMR video-to-SMPL-H reconstruction pipeline
- keep existing `.npz` output behavior intact
- keep existing FBX export behavior intact
- add a new MMD/VMD export backend based on the reconstructed SMPL-H motion
- map SMPL-H motion into conventional Japanese-named MMD bones
- write valid `.vmd` animation files for import into standard MMD humanoid models

## Core functionality

This fork still supports the normal FlowHMR workflow:

1. run inference from a video
2. reconstruct 3D human motion in SMPL-H space
3. save the motion as standard FlowHMR output
4. optionally export the same motion as VMD for MMD

The VMD export is designed to be an additional output format, not a replacement for the original FlowHMR outputs.

## Output files

The project keeps the usual FlowHMR output layout under `output/<video_name>/`, for example:

- `flowhmr_latest.npz`
- `flowhmr_latest.fbx`
- `flowhmr_latest.vmd`

The VMD file is generated from the same reconstructed motion data used by the existing FlowHMR exports.

## Typical usage

From the project root:

```bash
python run_v2m_demo_generation.py <video_or_directory>
```

The repository’s existing inference and output pipeline remains the primary entry point. The VMD export is added as an additional backend alongside the current NPZ/FBX generation logic.

## Project-specific modifications

The main additions are:

- a VMD writer that serializes standard VMD 0002 binary animation data
- SMPL-H to MMD retargeting for conventional humanoid bone names
- coordinate and rotation conversion from FlowHMR/SMPL-H space to MMD conventions
- root motion and torso distribution logic appropriate for MMD skeletons
- tests for VMD serialization and retargeting behavior
- documentation for export usage, supported bones, and limitations

## Important limitations

This is a compatibility-focused export layer, not a full PMX-authoring or model-specific retargeting system.

- It targets the conventional standard MMD humanoid skeleton
- supplementary or model-specific bones are optional and handled conservatively
- IK is not assumed to be universally valid without PMX-specific rig information
- VMD compatibility is intended to be broadly usable, but real-world MMD character behavior still depends on the target model rig

## Notes

This repository is meant to extend the original FlowHMR project without changing its core reconstruction behavior. For installation details, model downloads, training, and the upstream project documentation, refer to the original FlowHMR README above.
