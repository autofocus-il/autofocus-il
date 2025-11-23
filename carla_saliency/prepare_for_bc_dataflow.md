# `GazePreprocessor.prepare_for_bc` Data Flow

This document breaks down how `GazePreprocessor.prepare_for_bc` (defined in `data_utils/data_loader_robomimic.py`) transforms a batch of image and gaze sequences into behavior cloning (BC) training inputs. It follows the execution order inside the method and highlights the helper utilities involved.

## High-Level Goal

Given a sequence of observations (`obs_image_seq`) and gaze coordinates (`gaze_seq`), the function returns:

- `obs_image`: stacked frames formatted for the policy encoder (`[B, stacked_channels, H, W]`).
- `gaze_heatmaps`: stack-aligned gaze heatmaps (`[B, frame_stack, H, W]`).
- `center_idx`: the temporal index used as the anchor for both stacks.

The method concentrates on the last time step of each sequence (causal training) and optionally performs temporal aggregation of gaze information across the stack.

## Inputs

- `obs_image_seq` (`torch.Tensor`): shape `[B, L, H, W, C]` (batch, sequence length, height, width, channels). Can degrade to `[B, H, W, C]` if only a single frame exists.
- `gaze_seq` (`torch.Tensor`): shape `[B, L, P*2]` or `[B, L, P, 2]`, where `P` is `maxpoints` gaze positions per frame.
- `frame_stack` (`int`): number of frames to include in the temporal window `S` (taken from config `data.frame_stack`).
- `grayscale` (`bool`): converts RGB frames to a single channel if `True`.
- `aggregate_stack` (`bool`): toggles temporal smoothing via `build_stack_heatmaps` (True = causal aggregation, False = per-frame Gaussian masks only).
- `temporal_window` (`Optional[int]`, configurable): when set on the preprocessor it determines how many past frames are available for aggregation (can exceed `frame_stack`).

## Step-by-Step Flow

1. **Select anchor frame (`center_idx`)**
   - Computes `center_idx = obs_image_seq.shape[1] - 1` when a time dimension exists, otherwise `0`.
   - This choice makes the BC learner observe past frames up to the most recent one, matching causal behavior.

2. **Extract image window**
   - Calls `extract_image_stack_around_center(obs_image_seq, center_idx, frame_stack)`.
   - Internally uses `_gather_last_s_frames` to gather `S` frames ending at `center_idx`, padding the left boundary by repeating the earliest available frame if the sequence is shorter than `frame_stack`.
   - Resulting tensor: `[B, S, H, W, C]` (or original tensor if already in the right shape).

3. **Format images for the encoder**
   - Passes the stacked images through `_format_obs_image`.
   - Converts `uint8` inputs to `[0,1]` floats, optionally converts RGB to grayscale with luminance weights, and reshapes to channels-first layout: `[B, S*C', H, W]`, where `C'` is `1` (grayscale) or `3` (RGB).

4. **Prepare gaze heatmaps**
   - If `aggregate_stack` is `True` (default):
     - Calls `build_stack_heatmaps(gaze_seq, frame_stack, center_idx)`.
     - This helper slices a gaze window whose length is `temporal_window` (if provided) or `frame_stack`, and performs causal temporal aggregation while only returning the last `frame_stack` steps to stay aligned with the image stack:
       - Builds per-frame delta maps (`_build_delta_from_gaze_stack`).
       - For each target frame `s` (aligned to the last `frame_stack` indices), blends contributions from the configured temporal window `j ≤ target` (or the whole window if `temporal_use_future=True`).
       - Applies Gaussian blur kernels with time-varying `sigma` (`temporal_gamma * temporal_beta^{-k}`) and alpha decay (`alpha^k`) based on frame distance `k`.
       - Normalizes each aggregated heatmap to `[0,1]`.
     - Output shape: `[B, S, H, W]`.
   - If `aggregate_stack` is `False`:
     - Gathers the gaze window via `extract_gaze_stack_around_center`.
     - Generates independent per-frame Gaussian masks with `forward`.
     - Squeezes the singleton channel dimension and, for single-frame outputs with `frame_stack > 1`, repeats the mask across the stack for robustness.

5. **Return values**
   - `obs_image`: ready for the policy encoder.
   - `gaze_heatmaps`: aligned with the image stack (either temporally aggregated or raw).
   - `center_idx`: index of the newest frame used, enabling downstream modules to stay synchronized if needed.

## Interactions with Other Components

- `GazePreprocessor` is instantiated inside `RobomimicBCSequenceDataset` (see `data_utils/data_loader_robomimic.py`) with configuration parameters from `configs/*.yaml`.
- During training (e.g., `train/common/data.py`), batches from the dataset call `prepare_for_bc` to produce tensors used by the policy network and gaze-regularization losses.
- Temporal hyperparameters (`temporal_alpha`, `temporal_beta`, `temporal_gamma`, `temporal_flag`, `temporal_use_future`, `temporal_window`, `mask_sigma`, `max_points`) are configurable via Hydra (e.g., `configs/playground.yaml`).

## Summary

`prepare_for_bc` orchestrates the final preprocessing step before feeding data into the BC learner:

- Aligns image and gaze sequences around the latest available observation.
- Normalizes and reshapes image stacks for convolutional encoders.
- Produces gaze heatmaps that either carry temporal context (default) or per-frame masks, controlled via configuration.

This encapsulation keeps training scripts simple while centralizing all temporal gaze handling logic inside the `GazePreprocessor` utility.
