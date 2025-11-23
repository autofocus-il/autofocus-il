import datetime
import tempfile
from copy import copy
from socket import gethostname

from ml_collections import ConfigDict

try:
    import wandb
except Exception as e:  # pragma: no cover
    wandb = None


def _recursive_flatten_dict(d: dict):
    keys, values = [], []
    for key, value in d.items():
        if isinstance(value, dict):
            sub_keys, sub_values = _recursive_flatten_dict(value)
            keys += [f"{key}/{k}" for k in sub_keys]
            values += sub_values
        else:
            keys.append(key)
            values.append(value)
    return keys, values


class WandBLogger(object):
    @staticmethod
    def get_default_config():
        config = ConfigDict()
        config.project = "bridgedata_torch"
        config.entity = None
        config.exp_descriptor = ""
        config.unique_identifier = ""
        return config

    def __init__(self, wandb_config, variant, wandb_output_dir=None, debug=False):
        self.config = wandb_config
        if self.config.unique_identifier == "":
            self.config.unique_identifier = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.config.experiment_id = self.experiment_id = (
            f"{self.config.exp_descriptor}_{self.config.unique_identifier}"
        )

        if wandb_output_dir is None:
            wandb_output_dir = tempfile.mkdtemp()

        self._variant = copy(variant)
        if "hostname" not in self._variant:
            self._variant["hostname"] = gethostname()

        if wandb is None:
            self.run = None
            self._disabled = True
            return

        mode = "disabled" if debug else "online"
        self.run = wandb.init(
            config=self._variant,
            project=self.config.project,
            entity=self.config.entity,
            dir=wandb_output_dir,
            id=self.config.experiment_id,
            save_code=True,
            mode=mode,
        )
        self._disabled = False

    def log(self, data: dict, step: int | None = None):
        if self._disabled or wandb is None:
            return
        data_flat = _recursive_flatten_dict(data)
        data = {k: v for k, v in zip(*data_flat)}
        wandb.log(data, step=step)

