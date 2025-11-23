from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from common.gaze import get_gaze_mask

import torch
import torch.nn as nn

class MLP(nn.Module): 
    def __init__(self, input_dim: int, hidden_dims=(256, 256, 256), dropout_rate=0.0, activate_final=True): 
        super().__init__()
        layers = []
        last = input_dim
        for i, h in enumerate(hidden_dims):
            # 线性层
            layers.append(nn.Linear(last, h))
            # LayerNorm
            layers.append(nn.LayerNorm(h))
            # 激活
            layers.append(nn.ReLU(inplace=True))
            # Dropout（如果需要）
            if dropout_rate and dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            last = h

        if activate_final:
            self.net = nn.Sequential(*layers)
            self.out_dim = last
        else:
            # 去掉最后的 ReLU（以及可能的 Dropout）
            # 注意：如果要保留最后一层 Linear+Norm，而去掉激活，就删掉最后两层
            if dropout_rate and dropout_rate > 0:
                self.net = nn.Sequential(*layers[:-2])  # 去掉 ReLU + Dropout
            else:
                self.net = nn.Sequential(*layers[:-1])  # 去掉 ReLU
            self.out_dim = last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class GaussianPolicy(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        mlp: MLP,
        action_dim: int,
        tanh_squash_distribution: bool = False,
        state_dependent_std: bool = False,
        fixed_std: list[float] | None = None,
        use_proprio: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.mlp = mlp
        self.mu = nn.Linear(self.mlp.out_dim, action_dim)
        self.tanh_squash = tanh_squash_distribution
        self.state_std = state_dependent_std
        self.use_proprio = use_proprio
        if self.state_std:
            self.log_std = nn.Linear(self.mlp.out_dim, action_dim)
        else:
            fs = torch.tensor(
                fixed_std if fixed_std is not None else [1.0] * action_dim,
                dtype=torch.float32,
            )
            self.register_buffer("fixed_std", fs)

    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = obs["image"]  # (B, C, H, W)
        z_pool = self.encoder(x)
        z = z_pool.flatten(1)
        if self.use_proprio and ("proprio" in obs):
            prop = obs["proprio"]
            if prop.dim() > 2:
                prop = prop.flatten(1)
            z = torch.cat([z, prop], dim=-1)
        h = self.mlp(z)
        mu = self.mu(h)
        if self.state_std:
            std = torch.exp(self.log_std(h)).clamp(min=1e-4, max=10.0)
        else:
            std = self.fixed_std.expand_as(mu)
        return {"mu": mu, "std": std, "z_map": self.encoder.z_map}

    def dist(self, out: Dict[str, torch.Tensor]):
        mu, std = out["mu"], out["std"]
        return torch.distributions.Independent(torch.distributions.Normal(mu, std), 1)


@dataclass
class BCAgentConfig:
    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    # Scheduler
    warmup_steps: int = 1000
    decay_steps: int = 1000000
    scheduler: Dict[str, Any] = field(default_factory=dict)
    # Saliency regularization
    saliency: Dict[str, Any] = field(default_factory=dict)


class BCAgent:
    def __init__(self, model: GaussianPolicy, cfg: BCAgentConfig, action_dim: int, device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        # Scheduler selectable via config.scheduler.type
        self.scheduler = None
        sched_type = cfg.scheduler["type"]
        if sched_type == "constant":
            self.scheduler = None
        else:
            scheds: list[torch.optim.lr_scheduler._LRScheduler] = []
            if cfg.warmup_steps and cfg.warmup_steps > 0:
                scheds.append(torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1e-6, end_factor=1.0, total_iters=cfg.warmup_steps))
            if cfg.decay_steps and cfg.decay_steps > 0:
                scheds.append(torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.decay_steps, eta_min=0.0))
            if scheds:
                if len(scheds) == 1:
                    self.scheduler = scheds[0]
                else:
                    self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, scheds, milestones=[cfg.warmup_steps])
        self.step = 0

        # Saliency-variant config (optional, from YAML)
        self._saliency_enabled = cfg.saliency.enabled
        self._saliency_weight = cfg.saliency.weight
        self._saliency_beta = cfg.saliency.beta
            
    def update(self, batch: Dict[str, torch.Tensor]):
        self.model.train()
        base = self.model.module if hasattr(self.model, "module") else self.model
        # Move to device with channels_last for 4D tensors
        obs: Dict[str, torch.Tensor] = {}
        for k, v in batch["observations"].items():
            if not isinstance(v, torch.Tensor):
                continue
            t = v.to(self.device, non_blocking=True)
            if t.dim() == 4:  # (B,C,H,W)
                t = t.contiguous(memory_format=torch.channels_last)
            obs[k] = t
        actions = batch["actions"].to(self.device, non_blocking=True)
        out = base(obs)
        dist = base.dist(out)
        log_prob = dist.log_prob(actions)
        actor_loss = -log_prob.mean()
        reg_loss = torch.tensor(0.0, device=self.device)
        # Saliency regularization (Reg variant)
        # print(f"self._saliency_enabled: {self._saliency_enabled}")
        # print(f"obs: {obs.keys()}")
        if self._saliency_enabled and ("saliency" in obs):
            z_map = out["z_map"]  # (B,C,Hf,Wf)
            # print(f"z_map.shape: {z_map.shape}")
            # print(f"obs['saliency'].shape: {obs['saliency'].shape}")
            if z_map is not None and z_map.dim() == 4:
                target = obs["saliency"]  # (B,1,H,W)
                H, W = int(target.shape[-2]), int(target.shape[-1])
                pred = get_gaze_mask(z_map, self._saliency_beta, (H, W))  # (B,1,H,W)
                reg_loss = F.mse_loss(pred, target)
        loss = actor_loss + self._saliency_weight * reg_loss
        with torch.no_grad():
            mse = F.mse_loss(out["mu"], actions, reduction="none").sum(-1).mean()
            mean_std = out["std"].mean(dim=1).mean()
            max_std = out["std"].max()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.step += 1

        return {
            "actor_loss": float(actor_loss.item()),
            "saliency_reg": float(reg_loss.item()) if reg_loss is not None else 0.0,
            "total_loss": float(loss.item()),
            "mse": mse.item(),
            "log_probs": log_prob.mean().item(),
            "mean_std": mean_std.item(),
            "max_std": max_std.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    @torch.no_grad()
    def get_debug_metrics(self, batch: Dict[str, torch.Tensor]):
        self.model.eval()
        base = self.model.module if hasattr(self.model, "module") else self.model
        obs: Dict[str, torch.Tensor] = {}
        for k, v in batch["observations"].items():
            if not isinstance(v, torch.Tensor):
                continue
            t = v.to(self.device, non_blocking=True)
            if t.dim() == 4:
                t = t.contiguous(memory_format=torch.channels_last)
            obs[k] = t
        actions = batch["actions"].to(self.device, non_blocking=True)
        out = base(obs)
        dist = base.dist(out)
        log_prob = dist.log_prob(actions)
        mse = F.mse_loss(out["mu"], actions, reduction="none").sum(-1).mean()
        return {"mse": mse.item(), "log_probs": log_prob.mean().item()}

    @torch.no_grad()
    def sample_actions(self, obs: Dict[str, torch.Tensor], argmax: bool = False):
        self.model.eval()
        base = self.model.module if hasattr(self.model, "module") else self.model
        obs_dev: Dict[str, torch.Tensor] = {}
        for k, v in obs.items():
            if not isinstance(v, torch.Tensor):
                continue
            t = v.to(self.device, non_blocking=True)
            if t.dim() == 4:
                t = t.contiguous(memory_format=torch.channels_last)
            obs_dev[k] = t
        out = base(obs_dev)
        dist = base.dist(out)
        if argmax:
            return out["mu"].cpu()
        return dist.sample().cpu()
