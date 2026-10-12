# Utils

Shared helpers used by the training, simulation, and visualization code in `NewCode`.
Visualization-specific rendering code lives in `NewCode/Visualization`.

## Files

- `runtime.py`
  - Common runtime utilities: seeding, device/AMP setup, checkpoint loading, memmap DB checks, dataloaders, optimizer helpers, and normalization stats.

- `training.py`
  - Shared training helpers: common CLI arguments, data loader preparation, checkpoint save/resume, config/history writing, and best-checkpoint loading.

- `model_loading.py`
  - Restores trained models from checkpoints.
  - Supports `BasicVQVAE375to291`, `LMM375to291`, and `MVAE375to291`.

- `realtime.py`
  - Helpers for realtime scripts: parse style names, sample contiguous same-style segments, load memmap arrays, and write reconstructed/reference or generated/reference NPZ outputs plus metadata.

- `motion375.py`
  - Decodes the 375-dim motion format (`84 traj + 291 motion`) into trajectory context, root motion, and selected 24-joint poses.

- `quat.py`
  - NumPy quaternion math utilities, including conversions, vector rotation, FK, IK, and velocity-aware FK.

- `bvh.py`
  - BVH read/write helpers used by preprocessing and export code.

- `run_utils.sh`
  - Small shell helpers reused by the `run_*.sh` wrapper scripts for dependency checks, path checks, and argument assembly.

- `vq_target_cache.py`
  - Caches per-frame VQ codebook indices and restores quantized latents for FlowMatch training, so the frozen VQ encoder does not need to run on every step.

- `motion_eval.py`
  - Shared evaluation helpers for `basic_vqvae`, `lmm`, and `mvae`.
  - Reconstructs normalized batches, denormalizes motion for metrics, and computes MPJPE, FID, foot sliding, and jitteriness.

- `fid_feature_extractor.py`
  - Classifier-style feature extractor used for motion FID.
  - Includes model definition, checkpoint loading, and input conversion helpers for 375-d motion clips.

- `__init__.py`
  - Package export entry point for the main utility modules.
