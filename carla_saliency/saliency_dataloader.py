import hashlib

import numpy as np

import torch
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.dataset import SequenceDataset

def build_obs_specs(cfg_data):
    """Initialize ObsUtils with the chosen gaze key."""
    gaze_key = cfg_data.gaze_key
    obs_modality_specs = {
        'obs': {
            'rgb': ['image'],
            'low_dim': [gaze_key],
        },
        'goal': {'rgb': [], 'low_dim': []},
    }
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

def build_dataset(cfg_data, gaze_ratio: float = 1.0):
    gaze_key = cfg_data.gaze_key
    obs_keys = ['image', gaze_key]
    dataset_keys = ['actions', 'rewards', 'dones']
    action_keys = ['actions']
    action_config = {'actions': {'normalization': None} }

    dataset = SaliencySequenceDataset(
        hdf5_path=cfg_data.hdf5_path,
        obs_keys=obs_keys,
        dataset_keys=dataset_keys,
        action_keys=action_keys,
        action_config=action_config,
        frame_stack=cfg_data.frame_stack,
        seq_length=1,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=cfg_data.cache_mode,
        hdf5_cache_getitem=cfg_data.cache_getitem,
        hdf5_use_swmr=True,
        hdf5_normalize_obs=False,
        load_next_obs=False,
        filter_by_attribute=None,
        # Limit number of demos loaded according to config
        demo_limit=cfg_data.num_episodes,
        gaze_ratio=gaze_ratio,
    )
    return dataset

class SaliencySequenceDataset(SequenceDataset):
    """SequenceDataset wrapper that also precomputes per-sample gaze usage flags."""
    def __init__(self, hdf5_path, obs_keys, dataset_keys, action_keys=None, action_config=None, gaze_ratio: float = 1.0, **kwargs):
        # Set default action_keys and action_config if not provided
        if action_keys is None:
            action_keys = ['actions']
        if action_config is None:
            action_config = {
                'actions': {
                    'normalization': None
                }
            }
        self.gaze_ratio = float(gaze_ratio)
        super().__init__(
            hdf5_path=hdf5_path, 
            obs_keys=obs_keys, 
            dataset_keys=dataset_keys,
            action_keys=action_keys,
            action_config=action_config,
            **kwargs
        )
        self._is_valid_gaze = self._build_is_valid_gaze_flags()

    def _build_is_valid_gaze_flags(self) -> torch.Tensor:
        torch.manual_seed(42)
        num_samples = len(self)
        num_true = int(round(num_samples * self.gaze_ratio))

        arr = torch.zeros(num_samples, dtype=torch.bool)
        arr[:num_true] = True

        idx = torch.randperm(num_samples)
        arr = arr[idx]
        return arr



    def __getitem__(self, index):
        meta = super().__getitem__(index)
        meta['is_valid_gaze'] = self._is_valid_gaze[index]
        return meta
