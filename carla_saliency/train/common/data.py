from typing import Optional
from torch.utils.data import DataLoader


def build_dataloader(
    dataset, cfg_data, sampler: Optional[object], grad_accum_steps: int = 1
):
    loader = DataLoader(
        dataset,
        batch_size=cfg_data.batch_size // max(1, grad_accum_steps),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg_data.num_workers,
        pin_memory=True,
        persistent_workers=True if cfg_data.num_workers > 0 else False,
        prefetch_factor=cfg_data.prefetch_factor if cfg_data.num_workers > 0 else None,
    )
    return loader
