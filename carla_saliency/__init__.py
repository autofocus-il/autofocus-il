"""
VLM-GAZE: Vision-Language Model based Gaze Prediction for GABRIL-CARLA
"""

__version__ = "1.0.0"

# Expose vendored robomimic under top-level 'robomimic' if needed
# This allows 'import robomimic' to resolve to 'carla_saliency.robomimic' when the
# external package is not installed but the vendored copy exists.
try:  # pragma: no cover
    import robomimic  # noqa: F401
except Exception:  # fallback to vendored
    import sys as _sys
    import importlib as _importlib
    _vendored_spec = _importlib.util.find_spec('carla_saliency.robomimic')
    if _vendored_spec is not None:
        _module = _importlib.import_module('carla_saliency.robomimic')
        _sys.modules['robomimic'] = _module

# Make key modules available at package level
from .data_utils.data_loader_robomimic import SaliencySequenceDataset, GazePreprocessor
from .models.linear_models import AutoEncoder, Encoder, Decoder
from .models.gaze_predictor import UNet

__all__ = [
    'SaliencySequenceDataset',
    'GazePreprocessor', 
    'AutoEncoder',
    'Encoder',
    'Decoder',
    'UNet'
]