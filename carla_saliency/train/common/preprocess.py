#!/usr/bin/env python3
"""
Shared preprocessing utilities for image and gaze tensors
"""

import torch
from einops import rearrange


def _gather_last_s_frames(
    seq: torch.Tensor, center_idx: int, stack_len: int
) -> torch.Tensor:
    """
    Gather the last `stack_len` frames ending at `center_idx` from a batched sequence.

    Args:
        seq: Tensor shaped [B, L, ...]
        center_idx: int index along L to end the window on
        stack_len: number of frames to gather (S)

    Returns:
        Tensor shaped [B, S, ...] where S == stack_len
    """
    assert seq.dim() >= 2, f"Expected at least 2D tensor [B, L, ...], got {seq.shape}"
    B, L = seq.shape[0], seq.shape[1]
    # Build indices [center_idx - (S-1), ..., center_idx], clamped to [0, L-1]
    start = center_idx - (stack_len - 1)
    idxs = [min(max(i, 0), L - 1) for i in range(start, center_idx + 1)]
    # In case L < stack_len and center_idx < stack_len-1, pad by front-clamping
    while len(idxs) < stack_len:
        idxs.insert(0, idxs[0])
    index_tensor = torch.tensor(idxs, device=seq.device, dtype=torch.long)  # [S]
    # Gather along time dimension
    gathered = seq.index_select(dim=1, index=index_tensor)  # [B, S, ...]
    return gathered


def format_obs_image(
    obs_image: torch.Tensor, frame_stack: int, grayscale: bool
) -> torch.Tensor:
    # Normalize if uint8
    if obs_image.dtype == torch.uint8:
        obs_image = obs_image.float() / 255.0

    if obs_image.dim() == 5 and obs_image.shape[1] == frame_stack:
        # [B, stack, H, W, C] -> [B, stack*C, H, W] (with optional grayscale)
        B, stack, H, W, C = obs_image.shape
        obs_image = rearrange(obs_image, "b s h w c -> b s c h w")
        if grayscale and C == 3:
            obs_image = (
                0.299 * obs_image[:, :, 0:1]
                + 0.587 * obs_image[:, :, 1:2]
                + 0.114 * obs_image[:, :, 2:3]
            )
        obs_image = rearrange(obs_image, "b s c h w -> b (s c) h w")
        return obs_image

    if obs_image.dim() == 4:
        if obs_image.shape[1] == frame_stack and obs_image.shape[-1] in [1, 3]:
            B, stack, H, W, C = obs_image.shape
            obs_image = rearrange(obs_image, "b s h w c -> b s c h w")
            if grayscale and C == 3:
                obs_image = (
                    0.299 * obs_image[:, :, 0:1]
                    + 0.587 * obs_image[:, :, 1:2]
                    + 0.114 * obs_image[:, :, 2:3]
                )
            obs_image = rearrange(obs_image, "b s c h w -> b (s c) h w")
            return obs_image
        if obs_image.shape[-1] in [1, 3]:
            obs_image = rearrange(obs_image, "b h w c -> b c h w")
            if grayscale and obs_image.shape[1] == 3:
                obs_image = (
                    0.299 * obs_image[:, 0:1]
                    + 0.587 * obs_image[:, 1:2]
                    + 0.114 * obs_image[:, 2:3]
                )
            return obs_image

    return obs_image


def format_gaze_coords(gaze_coords: torch.Tensor) -> torch.Tensor:
    # Select last frame if stacked
    if gaze_coords.dim() == 3:
        # Preserve temporal/stack information by default for downstream stack-aware processing.
        # Callers that truly want only the last step should slice explicitly.
        # For backward compatibility in old paths that expect [B, P], keep a shallow view helper here:
        # return_last = False
        # if return_last:
        #     return gaze_coords[:, -1]
        # Otherwise, return as-is [B, S_or_L, ...]
        return gaze_coords
    return gaze_coords


def extract_image_stack_around_center(
    images_seq: torch.Tensor, center_idx: int, frame_stack: int
) -> torch.Tensor:
    """
    Convert a fetched image sequence [B, L, H, W, C] into a stack window [B, S, H, W, C]
    ending at `center_idx`.
    """
    if images_seq.dim() != 5:
        # Already [B, H, W, C] or in channels-first; return as-is
        return images_seq
    return _gather_last_s_frames(
        images_seq, center_idx=center_idx, stack_len=frame_stack
    )


def extract_gaze_stack_around_center(
    gaze_seq: torch.Tensor, center_idx: int, frame_stack: int
) -> torch.Tensor:
    """
    Convert a fetched gaze sequence [B, L, P*2] or [B, L, P, 2] into
    [B, S, P*2] or [B, S, P, 2] ending at `center_idx`.
    """
    if gaze_seq.dim() < 3:
        return gaze_seq
    return _gather_last_s_frames(gaze_seq, center_idx=center_idx, stack_len=frame_stack)
