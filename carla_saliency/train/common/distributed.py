#!/usr/bin/env python3
"""
Common utilities for (optional) DistributedDataParallel setup and rank0 printing
"""

from typing import Tuple
import os
import torch


def init_ddp(cfg_training) -> Tuple[bool, int, int, int, torch.device]:
    """Initialize DDP using either config flag or torchrun env vars.

    Priority: If torchrun env vars are present (LOCAL_RANK/WORLD_SIZE), enable DDP
    regardless of the Hydra config. Otherwise, fall back to the config flag.

    Returns: (is_distributed, rank, world_size, local_rank, device)
    """
    is_distributed = False
    rank = 0
    world_size = 1
    local_rank = 0

    env_has_dist = all(k in os.environ for k in ("LOCAL_RANK", "RANK", "WORLD_SIZE"))

    # Use torchrun-provided env if available
    if env_has_dist:
        import torch.distributed as dist

        backend = (
            getattr(cfg_training.distributed, "backend", "nccl")
            if hasattr(cfg_training, "distributed")
            else "nccl"
        )
        if not (dist.is_available() and dist.is_initialized()):
            dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.setDevice = torch.cuda.set_device  # alias for safety
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        is_distributed = True
        return is_distributed, rank, world_size, local_rank, device

    # If config requests DDP but no env provided, fall back to single process (safe default)
    device = torch.device(cfg_training.device if torch.cuda.is_available() else "cpu")
    return is_distributed, rank, world_size, local_rank, device


def rank0_print(rank: int, message: str):
    if rank == 0:
        print(message)


def wrap_ddp(module: torch.nn.Module, cfg_training, local_rank: int):
    """Wrap a module with DDP using cfg training.distributed settings."""
    from torch.nn.parallel import DistributedDataParallel as DDP

    find_unused = (
        bool(getattr(cfg_training.distributed, "find_unused_parameters", False))
        if hasattr(cfg_training, "distributed")
        else False
    )
    broadcast_buffers = (
        bool(getattr(cfg_training.distributed, "broadcast_buffers", True))
        if hasattr(cfg_training, "distributed")
        else True
    )
    grad_as_bucket_view = (
        bool(getattr(cfg_training.distributed, "gradient_as_bucket_view", True))
        if hasattr(cfg_training, "distributed")
        else True
    )
    return DDP(
        module,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=find_unused,
        broadcast_buffers=broadcast_buffers,
        gradient_as_bucket_view=grad_as_bucket_view,
    )


def destroy_distributed_if_initialized():
    """Gracefully destroy the default process group if initialized."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
