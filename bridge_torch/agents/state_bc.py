from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

# 复用现有的 MLP 实现和 Agent 基础设施
from .bc import MLP


class StateGaussianPolicy(nn.Module):
    """仅基于 proprio 的高斯策略（无视觉编码器）。

    - 输入: obs["proprio"] (B, P) 或更高维度，内部会展平到 (B, P_flat)
    - 输出: 字典 {"mu": (B, A), "std": (B, A), "z_map": None}
      其中 z_map 保持为 None 以兼容通用 BCAgent 的 saliency 分支（自动跳过）。
    """

    def __init__(
        self,
        mlp: MLP,
        action_dim: int,
        tanh_squash_distribution: bool = False,
        state_dependent_std: bool = False,
        fixed_std: list[float] | None = None,
        use_proprio: bool = True,
    ):
        super().__init__()
        self.mlp = mlp
        self.mu = nn.Linear(self.mlp.out_dim, action_dim)
        self.tanh_squash = bool(tanh_squash_distribution)
        self.state_std = bool(state_dependent_std)
        if self.state_std:
            self.log_std = nn.Linear(self.mlp.out_dim, action_dim)
        else:
            fs = torch.tensor(
                fixed_std if fixed_std is not None else [1.0] * action_dim,
                dtype=torch.float32,
            )
            self.register_buffer("fixed_std", fs)
        self.use_proprio = use_proprio
        
    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "proprio" not in obs:
            raise KeyError("StateGaussianPolicy expects obs['proprio'] as input")
        prop = obs["proprio"]
        if prop.dim() > 2:
            prop = prop.flatten(1)
        h = self.mlp(prop)
        mu = self.mu(h)
        if self.state_std:
            std = torch.exp(self.log_std(h)).clamp(min=1e-4, max=10.0)
        else:
            std = self.fixed_std.expand_as(mu)
        return {"mu": mu, "std": std, "z_map": None} # z_map is None for state-only

    def dist(self, out: Dict[str, torch.Tensor]):
        mu, std = out["mu"], out["std"]
        return torch.distributions.Independent(torch.distributions.Normal(mu, std), 1)


