#!/usr/bin/env python3
"""
Training data visualization: directly read one demo from a robomimic HDF5,
do per-frame temporal aggregation with GazePreprocessor, and export a GIF
with three panels: image, heatmap, and overlay. Configuration is driven by
Hydra YAML (configs/train_data_viz.yaml).
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, List
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import hydra
from omegaconf import DictConfig
import h5py


@hydra.main(version_base=None, config_path="../configs", config_name="train_data_viz")
def main(cfg: DictConfig):
    """
    可视化：直接读取 HDF5 的某个 demo，全时序逐帧做因果聚合（基于栈窗口的 build_stack_heatmaps），
    生成包含 Image / Heatmap / Overlay 三列的 GIF。保持 YAML 配置体系；viz.mode 默认 full_demo。
    """
    import sys

    sys.path.append("/path/to/project_root")

    from carla_saliency.data_utils import GazePreprocessor

    # Device and config
    device = str(getattr(cfg.training, "device", "cuda:0"))
    viz_mode = getattr(cfg.viz, "mode", "full_demo")
    assert (
        viz_mode == "full_demo"
    ), "当前脚本仅支持 viz.mode=full_demo，可在 YAML 中设置"

    hdf5_path = getattr(cfg.data, "hdf5_path", None)
    if hdf5_path is None:
        raise ValueError("cfg.data.hdf5_path 未设置")
    gaze_key = getattr(cfg.data, "gaze_key", "gaze_coords")

    # 选择 demo：优先使用 cfg.viz.demo_key，其次使用 cfg.viz.demo_index（默认 0）
    demo_key_cfg = getattr(cfg.viz, "demo_key", None)
    demo_index = int(getattr(cfg.viz, "demo_index", 0))

    # 初始化凝视预处理器（用于生成基础热图与stack聚合）
    gaze_preprocessor = GazePreprocessor(
        img_height=int(getattr(cfg.data, "img_height", 180)),
        img_width=int(getattr(cfg.data, "img_width", 320)),
        gaze_sigma=float(getattr(cfg.gaze, "mask_sigma", 30.0)),
        gaze_coeff=float(getattr(cfg.gaze, "mask_coeff", 0.8)),
        maxpoints=int(getattr(cfg.gaze, "max_points", 5)),
        device=device,
        temporal_alpha=float(getattr(cfg.gaze, "temporal_alpha", 0.7)),
        temporal_beta=1.0,
        temporal_gamma=1.0,
        temporal_use_future=bool(getattr(cfg.gaze, "temporal_use_future", False)),
        temporal_window=int(getattr(cfg.gaze, "temporal_window", None)),
    )

    # 读取 HDF5 的指定 demo
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            raise KeyError(f"HDF5 缺少 'data' 组: {hdf5_path}")
        data_grp = f["data"]
        demos = sorted(list(data_grp.keys()))
        if len(demos) == 0:
            raise RuntimeError("HDF5 中没有任何 demo")

        if demo_key_cfg is not None and str(demo_key_cfg) != "":
            # Accept either explicit key or a numeric index (int or numeric string)
            if isinstance(demo_key_cfg, int) or (
                isinstance(demo_key_cfg, str) and demo_key_cfg.isdigit()
            ):
                demo_index = int(demo_key_cfg)
                demo_index = max(0, min(demo_index, len(demos) - 1))
                demo_key = demos[demo_index]
            else:
                if demo_key_cfg in data_grp:
                    demo_key = str(demo_key_cfg)
                else:
                    # Fallback: try using as index if convertible
                    try:
                        demo_index = int(str(demo_key_cfg))
                        demo_index = max(0, min(demo_index, len(demos) - 1))
                        demo_key = demos[demo_index]
                    except Exception:
                        raise KeyError(f"指定的 demo_key 不存在: {demo_key_cfg}")
        else:
            demo_index = max(0, min(int(demo_index), len(demos) - 1))
            demo_key = demos[demo_index]

        demo = data_grp[demo_key]
        if "obs" not in demo:
            raise KeyError(f"demo 缺少 'obs' 组: {demo_key}")
        obs = demo["obs"]

        # 图像与凝视数据
        if "image" not in obs:
            raise KeyError(f"obs 中缺少 'image': {demo_key}")
        if gaze_key not in obs:
            raise KeyError(f"obs 中缺少 '{gaze_key}': {demo_key}")

        # numpy arrays
        imgs_np = np.array(obs["image"])  # [T, H, W, C]
        gaze_np = np.array(obs[gaze_key])  # [T, P*2] 或 [T, P, 2]

    T = int(imgs_np.shape[0])
    frames_img: List[torch.Tensor] = []  # each [C,H,W]
    frames_hm: List[torch.Tensor] = []  # each [1,H,W]

    # 组装全时序 gaze 到张量，置到 device
    gzs_torch = torch.from_numpy(gaze_np)
    if gzs_torch.dim() == 2:
        gzs_torch = gzs_torch.unsqueeze(0)  # [1, T, P*2]
    gzs_torch = gzs_torch.to(device)

    stack_window = int(
        getattr(
            cfg.viz, "stack_window", max(1, int(getattr(cfg.data, "frame_stack", 2)))
        )
    )
    with torch.no_grad():
        for t in range(T):
            # 图像处理 -> [C,H,W], 归一化到 [0,1]
            img_t = torch.from_numpy(imgs_np[t]).float()
            if img_t.max() > 1.0:
                img_t = img_t / 255.0
            if img_t.dim() == 2:
                img_t = img_t.unsqueeze(0)  # [1,H,W]
            elif img_t.dim() == 3:
                img_t = img_t.permute(2, 0, 1)  # [C,H,W]
            else:
                raise ValueError(f"Unexpected image shape at t={t}: {img_t.shape}")
            if getattr(cfg.model, "grayscale", False) and img_t.shape[0] == 3:
                r, gch, b = img_t[0:1], img_t[1:2], img_t[2:3]
                img_t = 0.299 * r + 0.587 * gch + 0.114 * b
            frames_img.append(img_t.cpu())

            # 基于 stack 窗口的因果聚合：取中心 t，窗口大小 S
            # 返回 [1, S, H, W]，每个 s 都是因果聚合后的热图；取最后一帧作为 t 时刻的聚合热图
            hm_stack = gaze_preprocessor.build_stack_heatmaps(
                gzs_torch, frame_stack=stack_window, center_idx=t
            )  # [1, S, H, W]
            hm_t = hm_stack[:, -1]  # [1, H, W]
            frames_hm.append(hm_t.detach().cpu())

    # 堆叠为可视化批次形状
    obs_vis = torch.stack(frames_img, dim=0).unsqueeze(0)  # [1,T,C,H,W]
    hm_vis = torch.stack(frames_hm, dim=0).unsqueeze(0)  # [1,T,1,H,W]

    # 可视化配置与导出 GIF
    visualizer = TrainingDataVisualizer(
        output_dir=getattr(cfg.logging, "viz_dir", "outputs/training_viz"),
        max_frames=getattr(cfg.viz, "max_frames", 999),
        fps=getattr(cfg.viz, "fps", 20),
        overlay_alpha=getattr(cfg.viz, "alpha", 0.5),
    )

    visualizer.visualize_batch(
        obs_vis,
        hm_vis,
        batch_idx=demo_index if demo_key_cfg is None else 0,
        epoch=0,
        save_individual=False,
    )

    print("Done. Visualization saved.")


import io


class TrainingDataVisualizer:
    """
    Visualizes training data from BC trainer including images, heatmaps and overlays
    """

    def __init__(
        self,
        output_dir: str = "outputs/training_viz",
        max_frames: int = 999,
        fps: int = 20,
        overlay_alpha: float = 0.4,
    ):
        """
        Initialize the visualizer

        Args:
            output_dir: Directory to save output GIFs
            max_frames: Maximum number of frames to include in GIF
            fps: Frames per second for GIF
            overlay_alpha: Alpha value for heatmap overlay (0=transparent, 1=opaque)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_frames = max_frames
        self.fps = fps
        self.overlay_alpha = overlay_alpha
        self.frame_duration = 1000 // fps  # Duration per frame in milliseconds

    def visualize_batch(
        self,
        obs_images: torch.Tensor,
        gaze_heatmaps: torch.Tensor,
        batch_idx: int = 0,
        epoch: int = 0,
        save_individual: bool = True,
    ) -> Optional[Path]:
        """
        Visualize a batch of training data

        Args:
            obs_images: Input images tensor [B, C, H, W] or [B, S, C, H, W]
            gaze_heatmaps: Gaze heatmap tensor [B, C, H, W] or [B, S, 1, H, W]
            batch_idx: Index of current batch
            epoch: Current training epoch
            save_individual: Whether to save individual frames as images

        Returns:
            Path to saved GIF file if successful
        """
        # Move tensors to CPU and convert to numpy
        obs_images = obs_images.detach().cpu()
        gaze_heatmaps = gaze_heatmaps.detach().cpu()

        # Handle frame stacking - extract individual frames from all samples in batch
        frames_img = []
        frames_hm = []

        # Process all samples in the batch
        batch_size = obs_images.shape[0]
        for sample_idx in range(min(batch_size, self.max_frames)):
            obs_img = obs_images[sample_idx]  # [C, H, W] or [S, C, H, W]
            gaze_hm = gaze_heatmaps[sample_idx]  # [C, H, W] or [S, 1, H, W]

            # Check if we have stacked frames
            if obs_img.dim() == 3:
                # Single frame or stacked frames in channel dimension
                C = obs_img.shape[0]

                # Determine number of stacked frames
                if C % 3 == 0:
                    # RGB stacked frames
                    num_stack = C // 3
                    for i in range(num_stack):
                        frame = obs_img[i * 3 : (i + 1) * 3]  # Extract RGB channels
                        frames_img.append(frame)
                elif C == 1 or C == 2:
                    # Grayscale (possibly stacked)
                    for i in range(C):
                        frame = obs_img[i : i + 1].repeat(
                            3, 1, 1
                        )  # Convert to RGB for visualization
                        frames_img.append(frame)
                else:
                    # Unknown format, take as is
                    frames_img.append(
                        obs_img[:3] if C >= 3 else obs_img.repeat(3, 1, 1)
                    )

                # Process heatmaps
                hm_C = gaze_hm.shape[0]
                for i in range(min(hm_C, num_stack if C % 3 == 0 else 1)):
                    frames_hm.append(gaze_hm[i : i + 1])

            else:
                # Explicit time/stack dimension [S, C, H, W]
                S = obs_img.shape[0]
                for i in range(S):
                    frame = obs_img[i]
                    if frame.shape[0] == 1:
                        frame = frame.repeat(3, 1, 1)  # Grayscale to RGB
                    elif frame.shape[0] > 3:
                        frame = frame[:3]  # Take first 3 channels
                    frames_img.append(frame)

                    if i < gaze_hm.shape[0]:
                        frames_hm.append(
                            gaze_hm[i, 0:1]
                            if gaze_hm.dim() == 4
                            else gaze_hm[i : i + 1]
                        )

        # Ensure we have matching number of frames
        min_frames = min(len(frames_img), len(frames_hm), self.max_frames)
        frames_img = frames_img[:min_frames]
        frames_hm = frames_hm[:min_frames]

        if len(frames_img) == 0:
            print(f"Warning: No frames to visualize for batch {batch_idx}")
            return None

        # Create visualizations
        pil_images = []
        individual_dir = (
            self.output_dir / f"epoch_{epoch:04d}" / f"batch_{batch_idx:06d}"
            if save_individual
            else None
        )
        if individual_dir:
            individual_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx, (img_frame, hm_frame) in enumerate(zip(frames_img, frames_hm)):
            # Create composite visualization
            fig = self._create_composite_figure(img_frame, hm_frame, frame_idx)

            # Convert matplotlib figure to PIL Image
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0.1)
            buf.seek(0)
            pil_img = Image.open(buf).copy()
            pil_images.append(pil_img)
            buf.close()
            plt.close(fig)

            # Save individual frame if requested
            if individual_dir:
                pil_img.save(individual_dir / f"frame_{frame_idx:04d}.png")

        # Save as GIF
        gif_path = self.output_dir / f"epoch_{epoch:04d}_batch_{batch_idx:06d}.gif"
        if len(pil_images) > 1:
            pil_images[0].save(
                gif_path,
                save_all=True,
                append_images=pil_images[1:],
                duration=self.frame_duration,
                loop=0,
            )
        else:
            # Single frame - save as static image
            pil_images[0].save(gif_path)

        print(f"Saved visualization to {gif_path} ({len(pil_images)} frames)")
        return gif_path

    def _create_composite_figure(
        self, img_tensor: torch.Tensor, hm_tensor: torch.Tensor, frame_idx: int
    ) -> plt.Figure:
        """
        Create a composite figure with image, heatmap, and overlay

        Args:
            img_tensor: Image tensor [C, H, W]
            hm_tensor: Heatmap tensor [1, H, W]
            frame_idx: Current frame index

        Returns:
            Matplotlib figure with 3 subplots
        """
        # Convert tensors to numpy
        img_np = img_tensor.numpy()
        hm_np = hm_tensor.numpy()

        # Normalize image to [0, 1] if needed
        if img_np.min() < 0:
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        elif img_np.max() > 1:
            img_np = img_np / 255.0

        # Ensure image is in HWC format for display
        if img_np.shape[0] in [1, 3]:
            img_np = np.transpose(img_np, (1, 2, 0))

        # Squeeze heatmap to 2D
        if hm_np.shape[0] == 1:
            hm_np = hm_np[0]

        # Normalize heatmap to [0, 1]
        if hm_np.max() > hm_np.min():
            hm_np = (hm_np - hm_np.min()) / (hm_np.max() - hm_np.min())

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. Original Image
        axes[0].imshow(
            img_np if img_np.shape[-1] == 3 else img_np[:, :, 0],
            cmap="gray" if img_np.shape[-1] == 1 else None,
        )
        axes[0].set_title(f"Input Image (Frame {frame_idx})")
        axes[0].axis("off")

        # 2. Gaze Heatmap
        im = axes[1].imshow(hm_np, cmap="hot", vmin=0, vmax=1)
        axes[1].set_title(f"Gaze Heatmap")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        # 3. Overlay
        axes[2].imshow(
            img_np if img_np.shape[-1] == 3 else img_np[:, :, 0],
            cmap="gray" if img_np.shape[-1] == 1 else None,
        )
        # Apply colormap to heatmap for overlay
        hm_colored = cm.hot(hm_np)
        # Create alpha mask based on heatmap intensity
        alpha_mask = hm_np * self.overlay_alpha
        hm_colored[:, :, 3] = alpha_mask
        axes[2].imshow(hm_colored)
        axes[2].set_title(f"Overlay (alpha={self.overlay_alpha})")
        axes[2].axis("off")

        plt.suptitle(f"Training Data Visualization", fontsize=14)
        plt.tight_layout()

        return fig

    def create_summary_gif(
        self,
        all_images: List[torch.Tensor],
        all_heatmaps: List[torch.Tensor],
        output_name: str = "training_summary.gif",
    ) -> Path:
        """
        Create a summary GIF from multiple batches

        Args:
            all_images: List of image tensors from different batches
            all_heatmaps: List of heatmap tensors from different batches
            output_name: Name of output GIF file

        Returns:
            Path to saved GIF file
        """
        pil_images = []

        for batch_idx, (imgs, hms) in enumerate(zip(all_images, all_heatmaps)):
            # Take first sample from each batch
            img = imgs[0] if imgs.dim() > 3 else imgs
            hm = hms[0] if hms.dim() > 3 else hms

            # Extract first frame if stacked
            if img.shape[0] > 3:
                img = img[:3]
            elif img.shape[0] == 1:
                img = img.repeat(3, 1, 1)

            if hm.shape[0] > 1:
                hm = hm[0:1]

            # Create visualization
            fig = self._create_composite_figure(img.cpu(), hm.cpu(), batch_idx)

            # Convert to PIL
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0.1)
            buf.seek(0)
            pil_img = Image.open(buf).copy()
            pil_images.append(pil_img)
            buf.close()
            plt.close(fig)

            if len(pil_images) >= self.max_frames:
                break

        # Save GIF
        gif_path = self.output_dir / output_name
        if len(pil_images) > 1:
            pil_images[0].save(
                gif_path,
                save_all=True,
                append_images=pil_images[1:],
                duration=self.frame_duration,
                loop=0,
            )
        else:
            pil_images[0].save(gif_path)

        print(f"Saved summary GIF to {gif_path} ({len(pil_images)} frames)")
        return gif_path


# Hook function to be called from train_bc.py
def visualize_training_batch(
    obs_images: torch.Tensor,
    gaze_heatmaps: torch.Tensor,
    epoch: int = 0,
    batch_idx: int = 0,
    output_dir: str = "outputs/training_viz",
    max_viz_per_epoch: int = 5,
) -> Optional[Path]:
    """
    Simple hook function to visualize training data
    Can be called from train_bc.py's compute_loss method

    Args:
        obs_images: Input images [B, C, H, W]
        gaze_heatmaps: Gaze heatmaps [B, C, H, W]
        epoch: Current epoch
        batch_idx: Current batch index
        output_dir: Output directory
        max_viz_per_epoch: Maximum visualizations per epoch

    Returns:
        Path to saved visualization or None
    """
    # Only visualize first few batches of each epoch
    if batch_idx >= max_viz_per_epoch:
        return None

    visualizer = TrainingDataVisualizer(
        output_dir=output_dir, max_frames=999, fps=20, overlay_alpha=0.5
    )

    return visualizer.visualize_batch(
        obs_images,
        gaze_heatmaps,
        batch_idx=batch_idx,
        epoch=epoch,
        save_individual=False,
    )


def visualize_from_training_data(
    data_path: str = "/path/to/dataset/bench2drive220_robomimic.hdf5",
    output_dir: str = "outputs/training_viz_real",
    num_demos: int = 3,
    frames_per_demo: int = 20,
    use_pseudo_gaze: bool = False,
    gaze_key: str = "gaze_coords",
    random_demos: bool = False,
    seed: int = None,
):
    """
    Load actual training data and visualize it

    Args:
        data_path: Path to the HDF5 dataset file
        output_dir: Output directory for visualizations
        num_demos: Number of demos to visualize
        frames_per_demo: Number of frames to visualize per demo
        use_pseudo_gaze: Whether to use pseudo gaze data
        gaze_key: Which gaze key to use (gaze_coords, gaze_coords_gaze_pseudo, etc.)
        random_demos: If True, randomly select demos instead of sequential
        seed: Random seed for reproducible demo selection
    """
    import sys

    sys.path.append("/path/to/project_root")

    from carla_saliency.data_utils import GazePreprocessor
    from carla_saliency.train.common.preprocess import format_obs_image
    from einops import repeat, rearrange

    print(f"Loading data from {data_path}...")

    # Initialize preprocessor
    gaze_preprocessor = GazePreprocessor(
        img_height=180,
        img_width=320,
        gaze_sigma=15.0,
        gaze_coeff=0.8,
        maxpoints=5,
        device="cpu",
    )

    # Initialize visualizer
    visualizer = TrainingDataVisualizer(
        output_dir=output_dir, max_frames=frames_per_demo, fps=5, overlay_alpha=0.5
    )

    # Load HDF5 file
    import h5py

    data_file = Path(data_path)

    if not data_file.exists():
        print(f"Data file not found: {data_file}")
        print("Falling back to synthetic data test...")
        return test_with_synthetic_data()

    with h5py.File(data_file, "r") as f:
        print(f"Dataset keys: {list(f.keys())}")

        # Get list of demos and filter valid ones
        if "data" in f:
            all_demo_keys = sorted(
                [k for k in f["data"].keys() if k.startswith("demo_")]
            )

            # Check which demos actually have data
            demo_keys = []
            for demo_key in all_demo_keys:
                try:
                    # Try to access the image data to verify it exists
                    _ = f[f"data/{demo_key}/obs/image"].shape
                    demo_keys.append(demo_key)
                except KeyError:
                    print(
                        f"Warning: {demo_key} found in keys but has no data, skipping..."
                    )

            print(
                f"Found {len(demo_keys)} valid demos (out of {len(all_demo_keys)} total)"
            )

            # Select demos - either random or sequential
            if random_demos:
                import random

                if seed is not None:
                    random.seed(seed)
                    print(f"Using random seed: {seed}")

                # Randomly select demos
                selected_indices = random.sample(
                    range(len(demo_keys)), min(num_demos, len(demo_keys))
                )
                selected_demos = [demo_keys[i] for i in selected_indices]
                print(f"Randomly selected demos: {', '.join(selected_demos)}")
            else:
                # Sequential selection (original behavior)
                selected_demos = demo_keys[: min(num_demos, len(demo_keys))]
                print(f"Sequential demos: {', '.join(selected_demos)}")

            # Process selected demos
            for demo_idx, demo_key in enumerate(selected_demos):
                print(
                    f"\nProcessing {demo_key} ({demo_idx+1}/{len(selected_demos)})..."
                )

                # Load image data for this demo
                image_data = f[f"data/{demo_key}/obs/image"][:]  # [T, H, W, C]

                # Determine which gaze key to use
                if use_pseudo_gaze:
                    gaze_key = "gaze_coords_gaze_pseudo"

                # Load gaze coordinates
                gaze_path = f"data/{demo_key}/obs/{gaze_key}"
                if gaze_path in f:
                    gaze_data = f[gaze_path][:]  # [T, 10] for 5 points with x,y coords
                    print(f"Using gaze key: {gaze_key}, shape: {gaze_data.shape}")
                else:
                    print(f"Warning: {gaze_path} not found, using zeros")
                    gaze_data = np.zeros((len(image_data), 10), dtype=np.float32)

                # Process frames
                total_frames = len(image_data)
                # For frame stacking, we need pairs of frames
                max_frame_pairs = (total_frames - 1) // 2
                frames_to_viz = min(frames_per_demo, max_frame_pairs)

                # Create batch of frames for visualization
                batch_images = []  # [B, 2, H, W, C]
                batch_gazes_seq = []  # [B, 2, 10] (two-frame gaze coords per pair)

                # Collect frame pairs for visualization
                for pair_idx in range(frames_to_viz):
                    frame_idx = pair_idx * 2
                    if frame_idx + 1 >= total_frames:
                        break

                    # Stack 2 consecutive frames
                    stacked_image = np.stack(
                        [image_data[frame_idx], image_data[frame_idx + 1]], axis=0
                    )  # [2, H, W, C]

                    # Stack corresponding gaze coords for the two frames in the pair
                    stacked_gaze = np.stack(
                        [
                            gaze_data[frame_idx],  # previous frame gaze
                            gaze_data[frame_idx + 1],  # current frame gaze
                        ],
                        axis=0,
                    )  # [2, 10]

                    batch_images.append(stacked_image)
                    batch_gazes_seq.append(stacked_gaze)

                if len(batch_images) == 0:
                    print(f"No frames to process for {demo_key}")
                    continue

                # Convert to tensors
                batch_images = torch.from_numpy(
                    np.array(batch_images)
                )  # [B, 2, H, W, C]
                batch_gazes_seq = torch.from_numpy(
                    np.array(batch_gazes_seq)
                )  # [B, 2, 10]

                # Format images for model input (stack=2)
                batch_images = format_obs_image(
                    batch_images, frame_stack=2, grayscale=False
                )

                # Reshape gaze coords [B, 2, 10] -> [B, 2, 5, 2]
                batch_gazes_seq = rearrange(
                    batch_gazes_seq, "b s (p c) -> b s p c", p=5, c=2
                )

                # Generate gaze heatmaps with causal aggregation over the pair (center at last frame)
                with torch.no_grad():
                    # Causally aggregate within the pair window
                    gaze_heatmaps = gaze_preprocessor.build_stack_heatmaps(
                        batch_gazes_seq, frame_stack=2, center_idx=1
                    )  # [B, 2, H, W]

                # Visualize this demo
                gif_path = visualizer.visualize_batch(
                    batch_images,
                    gaze_heatmaps,
                    batch_idx=demo_idx,
                    epoch=0,
                    save_individual=False,
                )

                if gif_path:
                    print(f"✓ Saved visualization to: {gif_path}")
        else:
            print("Error: 'data' group not found in HDF5 file")
            return False

    print(f"\n✓ All visualizations saved to: {output_dir}")
    return True


def test_with_synthetic_data():
    """Test with synthetic data when real data is not available"""
    print("\nTesting TrainingDataVisualizer with synthetic data...")

    # Create synthetic data
    batch_size = 2
    frame_stack = 2
    height, width = 180, 320

    # Stacked RGB images [B, stack*3, H, W]
    obs_images = torch.rand(batch_size, frame_stack * 3, height, width)

    # Stacked heatmaps [B, stack, H, W]
    gaze_heatmaps = torch.zeros(batch_size, frame_stack, height, width)

    # Add some Gaussian-like patterns
    for b in range(batch_size):
        for s in range(frame_stack):
            cx, cy = np.random.randint(50, width - 50), np.random.randint(
                50, height - 50
            )
            y, x = np.ogrid[:height, :width]
            mask = ((x - cx) ** 2 + (y - cy) ** 2) < 30**2
            gaze_heatmaps[b, s, mask] = 1.0

    # Apply Gaussian blur to heatmaps
    from scipy.ndimage import gaussian_filter

    for b in range(batch_size):
        for s in range(frame_stack):
            gaze_heatmaps[b, s] = torch.from_numpy(
                gaussian_filter(gaze_heatmaps[b, s].numpy(), sigma=10)
            )

    # Test visualizer
    visualizer = TrainingDataVisualizer(
        output_dir="outputs/test_training_viz", max_frames=20, fps=5, overlay_alpha=0.4
    )

    gif_path = visualizer.visualize_batch(
        obs_images, gaze_heatmaps, batch_idx=0, epoch=0, save_individual=True
    )

    if gif_path and gif_path.exists():
        print(f"✓ Test successful! GIF saved to: {gif_path}")
        return True
    else:
        print("✗ Test failed!")
        return False


if __name__ == "__main__":
    # Prefer Hydra-based visualization using dataloader and config
    main()
