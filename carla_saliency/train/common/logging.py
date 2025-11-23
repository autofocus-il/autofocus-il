#!/usr/bin/env python3
"""
Experiment logger: save dir, tensorboard SummaryWriter, checkpoint & params.json helpers
"""

from pathlib import Path
import datetime
import json
from typing import Optional, Dict, Any

from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    def __init__(self, cfg, task: str, rank: int):
        self.cfg = cfg
        self.rank = rank
        now_time = datetime.datetime.now()

        save_tag = self._build_save_tag()
        self.save_dir = "{}_{}".format(now_time.strftime("%Y_%m_%d_%H_%M_%S"), save_tag)

        self.log_dir = Path(cfg.logging.log_dir) / task / self.save_dir
        self.ckpt_dir = Path(cfg.logging.checkpoint_dir) / task / self.save_dir

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.writer: Optional[SummaryWriter] = (
            SummaryWriter(self.log_dir) if rank == 0 else None
        )

    def _build_save_tag(self) -> str:
        return str(self.cfg.tag)

    def rank0_print(self, msg: str):
        if self.rank == 0:
            print(msg)

    def add_scalars(self, metrics: Dict[str, float], step: int):
        if self.writer is None:
            return
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)
        self.writer.flush()

    def save_checkpoint(self, module_or_state_dict, epoch: int, final: bool = False):
        path = self.ckpt_dir / f"model_ep{epoch}.torch"
        import torch

        if hasattr(module_or_state_dict, "state_dict"):
            state = module_or_state_dict.state_dict()
        else:
            state = module_or_state_dict
        torch.save(state, path)
        self.rank0_print(f"Saved checkpoint: {path}")
        if final:
            torch.save(state, self.ckpt_dir / "model.torch")

    def save_params_json(self, params: Dict[str, Any]):
        if not getattr(self.cfg.logging, "save_params", True):
            return
        with open(self.ckpt_dir / "params.json", "w") as f:
            json.dump(params, f, indent=2)
