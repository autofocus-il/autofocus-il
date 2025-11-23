#!/usr/bin/env python3
"""
Common optimizer and scheduler builders for training scripts
"""

from typing import Tuple, Any, Iterable

import torch


def build_optimizer(params_or_model: Any, cfg) -> torch.optim.Optimizer:
    """Build optimizer from cfg for given params or model.

    Args:
        params_or_model: iterable of params or module with .parameters()
        cfg: cfg.optimizer namespace (expects type, lr, weight_decay)
    """
    params: Iterable
    if hasattr(params_or_model, "parameters"):
        params = params_or_model.parameters()
    else:
        params = params_or_model

    if cfg.type == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.type == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer type: {cfg.type}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_loader_len: int,
    training_cfg,
    scheduler_cfg,
    grad_accum_steps: int,
) -> Tuple[Any, bool]:
    """Build LR scheduler and return (scheduler, batch_scheduler_update).

    batch_scheduler_update indicates whether step per-batch is needed.
    """
    if scheduler_cfg.type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_cfg.step_size, gamma=scheduler_cfg.gamma
        )
        return scheduler, False

    if scheduler_cfg.type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=training_cfg.epochs, eta_min=scheduler_cfg.eta_min
        )
        return scheduler, False

    if scheduler_cfg.type == "cosine_warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_cfg.T_0,
            T_mult=scheduler_cfg.T_mult,
            eta_min=scheduler_cfg.eta_min,
        )
        return scheduler, True

    if scheduler_cfg.type == "cosine_warmup":
        import math
        from torch.optim.lr_scheduler import LambdaLR

        total_steps = max(
            1, (train_loader_len * training_cfg.epochs) // max(1, grad_accum_steps)
        )
        warmup_steps = scheduler_cfg.warmup_steps

        # Determine the base LR from the optimizer (not from cfg)
        # Prefer optimizer.defaults['lr'] when available; fallback to current group lr
        if hasattr(optimizer, "defaults") and "lr" in optimizer.defaults:
            base_lr = optimizer.defaults["lr"]
        else:
            base_lr = optimizer.param_groups[0].get(
                "initial_lr", optimizer.param_groups[0]["lr"]
            )

        def lr_lambda(step):
            if step < warmup_steps:
                return step / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            # Return multiplicative factor relative to base LR
            min_lr_ratio = scheduler_cfg.eta_min / max(1e-12, base_lr)
            return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

        scheduler = LambdaLR(optimizer, lr_lambda)
        return scheduler, True

    if scheduler_cfg.type == "onecycle":
        total_steps = training_cfg.epochs * train_loader_len
        # Use optimizer's base LR as the target max_lr
        if hasattr(optimizer, "defaults") and "lr" in optimizer.defaults:
            max_lr = optimizer.defaults["lr"]
        else:
            max_lr = optimizer.param_groups[0].get(
                "initial_lr", optimizer.param_groups[0]["lr"]
            )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=scheduler_cfg.pct_start,
            anneal_strategy=scheduler_cfg.anneal_strategy,
            div_factor=scheduler_cfg.div_factor,
            final_div_factor=scheduler_cfg.final_div_factor,
        )
        return scheduler, True

    if scheduler_cfg.type == "none":
        return None, False

    raise ValueError(f"Unknown scheduler type: {scheduler_cfg.type}")
