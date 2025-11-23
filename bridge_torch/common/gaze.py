from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def get_gaze_mask(
    spatial_features: torch.Tensor,
    beta: float = 1.0,
    out_hw: Tuple[int, int] | None = None,
) -> torch.Tensor:
    """Compute a saliency map from last spatial feature maps.

    Align implementation with carla_saliency: sum of absolute values across channels -> softmax with beta as temperature -> upsample -> normalize to [0,1].

    Args:
        spatial_features: (B, C, H, W) encoder spatial activations
        beta: softmax temperature (smaller is sharper; consistent with counterpart)
        out_hw: optional output size (H_out, W_out)

    Returns:
        (B, 1, H*, W*), range [0,1]
    """

    # 1) Channel aggregation (same as carla_saliency: sum after abs across channels)
    z = spatial_features.abs().sum(dim=1)  # (B, H, W)

    # 2) Temperature softmax (flatten by pixel -> softmax -> restore shape)
    B, Hf, Wf = z.shape
    zf = z.view(B, Hf * Wf)
    # Consistent with counterpart: use z / beta as input for temperature softmax
    beta_safe = float(beta) if isinstance(beta, (int, float)) else 1.0
    zf = torch.nn.functional.softmax(zf / max(beta_safe, 1e-8), dim=-1)
    sal = zf.view(B, 1, Hf, Wf)  # (B,1,Hf,Wf)

    # 3) Upsample to target resolution (if specified)
    if out_hw is not None:
        Ho, Wo = int(out_hw[0]), int(out_hw[1])
        if Ho > 0 and Wo > 0 and (Ho != Hf or Wo != Wf):
            sal = F.interpolate(sal, size=(Ho, Wo), mode="bicubic", align_corners=False)

    # 4) Robust normalization to [0,1] again
    flat = sal.view(B, 1, -1)
    mx = flat.max(dim=-1).values.view(B, 1, 1, 1)
    mn = flat.min(dim=-1).values.view(B, 1, 1, 1)
    sal = (sal - mn) / (mx - mn + 1e-8)
    sal = sal.clamp_(0.0, 1.0)
    return sal
