import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from einops import rearrange


def get_gaze_mask(z_oreo, beta, target_size):
    z_oreo_flat = z_oreo.abs().sum(dim=1)
    # apply softmax on the last two dimensions of the tensor
    z_oreo_shape = z_oreo_flat.shape
    z_oreo_flat = rearrange(z_oreo_flat, "b h w -> b (h w)")

    z_oreo_softmax = torch.nn.functional.softmax(z_oreo_flat / beta, dim=-1)
    z_oreo_softmax = rearrange(
        z_oreo_softmax, "b (h w) -> b h w", h=z_oreo_shape[-2], w=z_oreo_shape[-1]
    )

    # return z_oreo_softmax

    # scale up z_oreo to the size of xx
    z_oreo_mask = torch.nn.functional.interpolate(
        rearrange(z_oreo_softmax, "b h w -> b 1 h w"), size=target_size, mode="bicubic"
    )

    # normalize z_oreo_mask
    flattened = rearrange(z_oreo_mask, "b c h w -> b c (h w)")
    mx = flattened.max(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
    mn = flattened.min(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
    gaze_mask = (z_oreo_mask - mn) / (mx - mn)

    # masked_xx = xx * gaze_mask
    # plot_once(xx_clean, masked_xx, 2, beta)

    return gaze_mask


def apply_gmd_dropout(z, g, test_mode=False):
    dropout_prob = 0.7
    B, C, H, W = z.shape
    A = torch.rand([B, 1, H, W], device=z.device)

    if g.dim() == 3:
        g = g.unsqueeze(1)
    K = torch.nn.functional.interpolate(g, size=(H, W), mode="bicubic")
    if K.shape[1] != 1:
        K = K.mean(dim=1, keepdim=True)

    denom = K.max(dim=(-1, -2), keepdim=True)[0] - K.min(dim=(-1, -2), keepdim=True)[0]
    K = (K - K.min(dim=(-1, -2), keepdim=True)[0]) / (denom + 1e-8)

    K = dropout_prob * K + (1 - dropout_prob)

    # for masked, set K to 1
    zero_mask = (g.abs().sum(dim=(1, 2, 3)) == 0).float().view(B, 1, 1, 1)
    K = K * (1 - zero_mask) + 1.0 * zero_mask

    if test_mode:
        return z * K
    else:
        M = (A < K).float()
        return z * M


temp_flag = False


def plot_once(xx_clean, masked_xx, count, beta):
    global temp_flag
    os.makedirs("figs", exist_ok=True)
    if not temp_flag:
        # save 5 random images
        idxs = np.random.choice(range(xx_clean.shape[0]), count)
        for idx in idxs:
            plt.imshow(masked_xx[idx, 0].cpu(), cmap="gray")
            plt.savefig(
                "figs/obs_masked_{}_beta_{}.png".format(idx, beta), bbox_inches="tight"
            )
            plt.clf()

            plt.imshow(xx_clean[idx, 0].cpu(), cmap="gray")
            plt.savefig(
                "figs/obs_clean_{}_beta_{}.png".format(idx, beta), bbox_inches="tight"
            )
            plt.clf()
        temp_flag = True
