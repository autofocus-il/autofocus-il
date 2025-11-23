#!/usr/bin/env python3
"""
Robomimic SequenceDataset Integration for GABRIL-CARLA
Provides dataset/dataloader functionality using robomimic's efficient SequenceDataset
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Import robomimic (external dependency or vendored under carla_saliency.robomimic)
try:
    from robomimic.utils.dataset import SequenceDataset  # type: ignore
except Exception:  # fallback to vendored copy within repo
    from robomimic.utils.dataset import SequenceDataset  # type: ignore

# ============================================================================
# Custom Gaze Preprocessor for Robomimic
# ============================================================================


class GazePreprocessor(nn.Module):
    """
    GPU-based gaze heatmap generation compatible with robomimic
    """

    def __init__(
        self,
        img_height: int = 180,
        img_width: int = 320,
        gaze_sigma: float = 30.0,
        maxpoints: int = 5,
        device: str = "cuda",
        temporal_alpha: float = 0.7,
        temporal_beta: float = 0.8,
        temporal_gamma: float = 1.0,
        temporal_use_future: bool = True,
        temporal_window: Optional[int] = None,
    ):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.maxpoints = maxpoints
        self.gaze_sigma = gaze_sigma
        self.device = device
        # Temporal hyperparameters
        self.temporal_alpha = float(temporal_alpha)
        self.temporal_beta = float(temporal_beta)
        self.temporal_gamma = float(temporal_gamma)
        self.temporal_use_future = bool(temporal_use_future)
        self.temporal_window = (
            int(temporal_window) if temporal_window is not None else None
        )

        # Pre-compute Gaussian kernel for base forward()
        kernel_size = int(4 * gaze_sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size

        # Create 1D Gaussian kernel
        x = torch.arange(kernel_size).float() - kernel_size // 2
        kernel_1d = torch.exp(-(x**2) / (2 * gaze_sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        self.register_buffer("kernel_1d", kernel_1d.to(device))
        # Cache for variable-sigma kernels (for temporal aggregation)
        self._sigma_kernel_cache: dict[float, torch.Tensor] = {}

    def forward(self, gaze_coords: torch.Tensor) -> torch.Tensor:
        """
        Generate gaze heatmaps from coordinates

        Args:
            gaze_coords: [B, T, maxpoints*2] or [B, T, maxpoints, 2] tensor

        Returns:
            heatmaps: [B, T, 1, H, W] tensor
        """
        if gaze_coords.dim() == 3 and gaze_coords.shape[-1] == self.maxpoints * 2:
            # [B, T, maxpoints*2] -> [B, T, maxpoints, 2]
            gaze_coords = rearrange(
                gaze_coords, "b t (p c) -> b t p c", p=self.maxpoints, c=2
            )
        elif gaze_coords.dim() == 2:
            # [T, maxpoints*2] -> [1, T, maxpoints, 2]
            gaze_coords = rearrange(
                gaze_coords, "t (p c) -> 1 t p c", p=self.maxpoints, c=2
            )

        B, T, P, _ = gaze_coords.shape
        H, W = self.img_height, self.img_width
        device = gaze_coords.device

        # Valid points mask: (x>=0 and y>=0)
        valid_mask = (gaze_coords[..., 0] >= 0) & (
            gaze_coords[..., 1] >= 0
        )  # [B, T, P]

        # Scale coords to pixel indices
        x_coords = (gaze_coords[..., 0].clamp(0, 1) * (W - 1)).long().clamp(0, W - 1)
        y_coords = (gaze_coords[..., 1].clamp(0, 1) * (H - 1)).long().clamp(0, H - 1)

        # Uniform weights per valid gaze point within each (B, T), independent of gaze_coeff
        weights = valid_mask.float()

        # Build delta maps for all (B,T) at once via scatter_add
        num_samples = B * T
        delta = torch.zeros(num_samples, H * W, device=device, dtype=torch.float32)
        # Compute linear indices per pixel
        lin_idx = rearrange(y_coords * W + x_coords, "b t p -> (b t) p")  # [B*T, P]
        w_flat = rearrange(weights, "b t p -> (b t) p")
        delta.scatter_add_(dim=1, index=lin_idx, src=w_flat)
        delta = rearrange(delta, "(b t) (h w) -> b t 1 h w", b=B, t=T, h=H, w=W)

        # Apply separable Gaussian blur in a vectorized manner
        padding = self.kernel_size // 2
        kernel = rearrange(self.kernel_1d, "l -> 1 1 l 1")
        # Flatten batch and time for convolution
        delta_bt = rearrange(delta, "b t c h w -> (b t) c h w")
        blurred = F.conv2d(delta_bt, kernel, padding=(0, padding))
        kernel_t = rearrange(kernel, "a b h w -> a b w h")
        blurred = F.conv2d(blurred, kernel_t, padding=(padding, 0))
        # Normalize each heatmap to [0, 1] via min-max
        min_vals = blurred.amin(dim=(2, 3), keepdim=True)
        max_vals = blurred.amax(dim=(2, 3), keepdim=True)
        normalized = (blurred - min_vals) / (max_vals - min_vals + 1e-8)
        heatmaps = rearrange(normalized, "(b t) c h w -> b t c h w", b=B, t=T)

        return heatmaps

    # ---------------------------------------------------------------------
    # Stack-aware helpers and unified APIs for training scripts
    # ---------------------------------------------------------------------
    @staticmethod
    def _gather_last_s_frames(
        seq: torch.Tensor, center_idx: int, stack_len: int
    ) -> torch.Tensor:
        """
        Generic utility: from [B, L, ...] gather a window [B, S, ...] that ends at center_idx.
        Pads by clamping at boundaries to preserve length S.
        """
        assert seq.dim() >= 2, f"Expected [B, L, ...], got {seq.shape}"
        B, L = seq.shape[0], seq.shape[1]
        start = center_idx - (stack_len - 1)
        idxs = [min(max(i, 0), L - 1) for i in range(start, center_idx + 1)]
        while len(idxs) < stack_len:
            idxs.insert(0, idxs[0])
        index_tensor = torch.tensor(idxs, device=seq.device, dtype=torch.long)
        return seq.index_select(dim=1, index=index_tensor)

    @staticmethod
    def extract_image_stack_around_center(
        images_seq: torch.Tensor, center_idx: int, frame_stack: int
    ) -> torch.Tensor:
        """
        From [B, L, H, W, C], build [B, S, H, W, C] ending at center_idx.
        If input is already [B, H, W, C], return as-is.
        """
        if images_seq.dim() != 5:
            return images_seq
        return GazePreprocessor._gather_last_s_frames(
            images_seq, center_idx=center_idx, stack_len=frame_stack
        )

    @staticmethod
    def extract_gaze_stack_around_center(
        gaze_seq: torch.Tensor, center_idx: int, frame_stack: int
    ) -> torch.Tensor:
        """
        From [B, L, P*2] or [B, L, P, 2], build [B, S, P*2] or [B, S, P, 2].
        If input has no time dimension, return as-is.
        """
        if gaze_seq.dim() < 3:
            return gaze_seq
        return GazePreprocessor._gather_last_s_frames(
            gaze_seq, center_idx=center_idx, stack_len=frame_stack
        )

    @staticmethod
    def _format_obs_image(
        images: torch.Tensor, frame_stack: int, grayscale: bool
    ) -> torch.Tensor:
        """
        Format images for encoder input. Accepts [B, S, H, W, C] or [B, H, W, C].
        Returns channels-first tensor [B, C_img, H, W] where C_img = S * (1 or 3).
        """
        from einops import rearrange as _rearr

        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        # [B, S, H, W, C]
        if images.dim() == 5 and images.shape[1] == frame_stack:
            B, S, H, W, C = images.shape
            x = _rearr(images, "b s h w c -> b s c h w")
            if grayscale and C == 3:
                x = 0.299 * x[:, :, 0:1] + 0.587 * x[:, :, 1:2] + 0.114 * x[:, :, 2:3]
            x = _rearr(x, "b s c h w -> b (s c) h w")
            return x
        # [B, H, W, C]
        if images.dim() == 4 and images.shape[-1] in [1, 3]:
            x = _rearr(images, "b h w c -> b c h w")
            if grayscale and x.shape[1] == 3:
                x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
            return x
        return images

    def build_stack_heatmaps(
        self, gaze_seq: torch.Tensor, frame_stack: int, center_idx: int
    ) -> torch.Tensor:
        """
        Build per-stack heatmaps with causal aggregation along stack S.

        Args:
            gaze_seq: [B, L, P*2] or [B, L, P, 2]
            frame_stack: S
            center_idx: center time to end the stack window on

        Returns:
            heatmaps_stack: [B, S, H, W] in [0,1]
        """
        # 1) Extract [B, W, ...] where W can differ from S when temporal_window is set
        window = (
            max(frame_stack, self.temporal_window)
            if self.temporal_window is not None
            else frame_stack
        )
        gaze_stack = self.extract_gaze_stack_around_center(
            gaze_seq, center_idx=center_idx, frame_stack=window
        )

        # Implement variable-sigma Gaussian per temporal offset k with alpha^k decay
        B = gaze_stack.shape[0]
        S = frame_stack
        W = gaze_stack.shape[1]
        H, ImgW = self.img_height, self.img_width

        # 1) Build per-step delta maps: [B, W, 1, H, ImgW]
        delta_stack = self._build_delta_from_gaze_stack(gaze_stack, H, ImgW)

        # 2) For each target time s (aligned with the last S frames), aggregate over temporal neighbors j
        agg_stack = torch.zeros(
            B, S, 1, H, ImgW, device=delta_stack.device, dtype=delta_stack.dtype
        )
        alpha = float(self.temporal_alpha)
        beta = float(self.temporal_beta)
        gamma = float(self.temporal_gamma)
        use_future = bool(self.temporal_use_future)

        target_indices = list(range(max(0, W - S), W))
        if len(target_indices) < S:
            while len(target_indices) < S:
                target_indices.insert(0, target_indices[0])

        for s, target_idx in enumerate(target_indices[:S]):
            acc = torch.zeros(
                B, 1, H, ImgW, device=delta_stack.device, dtype=delta_stack.dtype
            )
            if use_future:
                j_range = range(0, W)
            else:
                lower = max(0, target_idx - (window - 1))
                j_range = range(lower, target_idx + 1)
            for j in j_range:
                # temporal offset k relative to target index
                k = abs(target_idx - j) if use_future else (target_idx - j)
                # sigma_k = gamma * beta^{-k}
                sigma_k = float(gamma) * (float(beta) ** float(-k))
                k1d = self._get_or_make_kernel_1d(
                    sigma_k, device=delta_stack.device, dtype=delta_stack.dtype
                )
                kx = k1d.view(1, 1, -1, 1)
                ky = kx.permute(0, 1, 3, 2)
                cur = delta_stack[:, j]  # [B,1,H,ImgW]
                cur = F.conv2d(cur, kx, padding=(0, kx.shape[2] // 2))
                cur = F.conv2d(cur, ky, padding=(ky.shape[3] // 2, 0))
                if k > 0:
                    cur = cur * (alpha**k)
                acc = acc + cur
            amin = acc.amin(dim=(-2, -1), keepdim=True)
            amax = acc.amax(dim=(-2, -1), keepdim=True)
            acc = (acc - amin) / (amax - amin + 1e-8)
            agg_stack[:, s] = acc

        return agg_stack.squeeze(2)

    def _get_or_make_kernel_1d(
        self, sigma: float, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Return cached 1D Gaussian kernel for a given sigma (on correct device / dtype)."""
        # guard sigma
        sigma = max(1e-3, float(sigma))
        key = float(sigma)
        k = self._sigma_kernel_cache.get(key, None)
        if k is None:
            size = int(4 * sigma + 1)
            if size % 2 == 0:
                size += 1
            x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
            k = torch.exp(-(x**2) / (2 * sigma**2))
            k = (k / (k.sum() + 1e-8)).to(dtype)
            self._sigma_kernel_cache[key] = k
        else:
            if k.device != device or k.dtype != dtype:
                k = k.to(device=device, dtype=dtype)
                self._sigma_kernel_cache[key] = k
        return k

    def _build_delta_from_gaze_stack(
        self, gaze_stack: torch.Tensor, H: int, W: int
    ) -> torch.Tensor:
        """Vectorized delta map construction for [B,S,P*2] or [B,S,P,2] -> [B,S,1,H,W]."""
        from einops import rearrange as _rearr

        if gaze_stack.dim() == 3 and gaze_stack.shape[-1] == self.maxpoints * 2:
            gaze_stack = _rearr(
                gaze_stack, "b s (p c) -> b s p c", p=self.maxpoints, c=2
            )
        B, S, P, _ = gaze_stack.shape
        device = gaze_stack.device
        valid_mask = (gaze_stack[..., 0] >= 0) & (gaze_stack[..., 1] >= 0)
        x_coords = (gaze_stack[..., 0].clamp(0, 1) * (W - 1)).long().clamp(0, W - 1)
        y_coords = (gaze_stack[..., 1].clamp(0, 1) * (H - 1)).long().clamp(0, H - 1)
        weights = valid_mask.float()
        num_samples = B * S
        delta = torch.zeros(num_samples, H * W, device=device, dtype=torch.float32)
        lin_idx = (y_coords * W + x_coords).view(num_samples, P)
        w_flat = weights.view(num_samples, P)
        delta.scatter_add_(dim=1, index=lin_idx, src=w_flat)
        delta = _rearr(delta, "(b s) (h w) -> b s 1 h w", b=B, s=S, h=H, w=W)
        return delta

    def prepare_for_bc(
        self,
        obs_image_seq: torch.Tensor,
        gaze_seq: torch.Tensor,
        frame_stack: int,
        grayscale: bool = False,
        aggregate_stack: bool = True,
    ):
        """
        One-call API for BC training to get encoder-ready images and stack-aggregated gaze heatmaps.

        Args:
            obs_image_seq: [B, L, H, W, C]
            gaze_seq: [B, L, P*2] or [B, L, P, 2]
            frame_stack: S
            seq_length: configured sequence length L (used to pick center)
            grayscale: whether to convert RGB to 1-channel

        Returns:
            obs_image: [B, S*C', H, W]
            gaze_heatmaps: [B, S, H, W]
            center_idx: int
        """
        # Determine center index: use last available step (ignore external seq_length)
        center_idx = (obs_image_seq.shape[1] - 1) if obs_image_seq.dim() > 1 else 0

        # Build image stack window and format to channels-first
        imgs_stack = self.extract_image_stack_around_center(
            obs_image_seq, center_idx=center_idx, frame_stack=frame_stack
        )
        obs_image = self._format_obs_image(
            imgs_stack, frame_stack=frame_stack, grayscale=grayscale
        )

        # Build gaze heatmaps along stack
        if aggregate_stack:
            gaze_heatmaps = self.build_stack_heatmaps(
                gaze_seq, frame_stack=frame_stack, center_idx=center_idx
            )
        else:
            # Base per-stack heatmaps without causal aggregation
            gaze_stack = self.extract_gaze_stack_around_center(
                gaze_seq, center_idx=center_idx, frame_stack=frame_stack
            )
            base_stack = self.forward(gaze_stack)  # [B, S, 1, H, W]
            if base_stack.dim() == 5:
                gaze_heatmaps = base_stack.squeeze(2)  # [B, S, H, W]
            else:
                # Robustness: if [B,1,H,W], tile to S then squeeze
                gaze_heatmaps = base_stack
                if (
                    gaze_heatmaps.dim() == 4
                    and gaze_heatmaps.shape[1] == 1
                    and frame_stack > 1
                ):
                    gaze_heatmaps = gaze_heatmaps.repeat(1, frame_stack, 1, 1)
        return obs_image, gaze_heatmaps, center_idx

    def prepare_for_gaze_predictor(
        self,
        obs_image_seq: torch.Tensor,
        gaze_seq: torch.Tensor,
        frame_stack: int,
        grayscale: bool = False,
    ):
        """
        One-call API for gaze predictor training.
        Builds an image stack [B, S, H, W, C] and aggregates gaze along stack to [B, 1, H, W]
        using forward_temporal centered at the last stack frame.
        """
        center_idx = (obs_image_seq.shape[1] - 1) if obs_image_seq.dim() > 1 else 0
        imgs_stack = self.extract_image_stack_around_center(
            obs_image_seq, center_idx=center_idx, frame_stack=frame_stack
        )
        obs_image = self._format_obs_image(
            imgs_stack, frame_stack=frame_stack, grayscale=grayscale
        )

        # Extract gaze stack and apply causal aggregation along stack to the last step only
        gaze_stack_agg = self.build_stack_heatmaps(
            gaze_seq, frame_stack=frame_stack, center_idx=center_idx
        )  # [B,S,H,W]
        last = gaze_stack_agg[:, -1]  # [B,H,W]
        return obs_image, last.unsqueeze(1), center_idx  # [B,1,H,W]
