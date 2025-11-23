"""
Converts data from the BridgeData raw format to numpy format (bdv2).

This script mirrors the functionality of bridge_data_v2/data_processing/bridgedata_raw_to_numpy.py
but removes dependencies on TensorFlow and absl. It uses argparse, numpy, PIL, and tqdm.

Directory structure assumption for input data (same as original):

    bridgedata_raw/
        rss/
            toykitchen2/
                set_table/
                    00/
                        2022-01-01_00-00-00/
                            collection_metadata.json
                            config.json
                            diagnostics.png
                            raw/
                                traj_group0/
                                    traj0/
                                        obs_dict.pkl
                                        policy_out.pkl
                                        agent_data.pkl
                                        images0/
                                            im_0.jpg
                                            im_1.jpg
                                            ...

Output structure mirrors the input tree at a depth controlled by --depth and
produces train/val/out.npy under the mapped output path.

Written originally by Kevin Black (kvablack@berkeley.edu). TensorFlow/absl-free
port authored for bdv2 by maintaining parity with the original logic.
"""

import argparse
import copy
import glob
import logging
import os
import pickle
import random
import re
from collections import defaultdict
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

import numpy as np
import tqdm
from PIL import Image


# Parsed arguments will be stored here for module-wide access
ARGS: argparse.Namespace


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _squash_image(path: str) -> np.ndarray:
    """Load and resize an image to ARGS.im_size x ARGS.im_size using PIL.

    Falls back to Image.ANTIALIAS if Image.Resampling is unavailable.
    """
    im = Image.open(path)
    try:
        resample = Image.Resampling.LANCZOS  # Pillow>=9
    except AttributeError:
        resample = Image.ANTIALIAS  # type: ignore[attr-defined]
    im = im.resize((ARGS.im_size, ARGS.im_size), resample)
    out = np.asarray(im).astype(np.uint8)
    return out


def _squash_saliency_map(arr: np.ndarray) -> np.ndarray:
    """Resize a single-channel float32 saliency map (H,W) to (im_size, im_size).

    Keeps dtype float32 and value range [0,1]. Returns HWC with channel=1.
    """
    assert arr.ndim == 2, f"Expected (H,W) float32, got {arr.shape}"
    try:
        resample = Image.Resampling.BILINEAR  # Pillow>=9
    except AttributeError:
        resample = Image.BILINEAR  # type: ignore[attr-defined]
    im = Image.fromarray(arr, mode="F")
    im = im.resize((ARGS.im_size, ARGS.im_size), resample)
    out = np.asarray(im).astype(np.float32)
    return out[..., None]  # (H,W,1)


def _process_saliency(traj_path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Read saliency_map.pkl at trajectory root and build obs/next lists.

    - Expects shape (T, 1, H, W) float32 in [0,1]
    - Returns obs = [0..T-2], next = [1..T-1], each element shaped (H',W',1)
    """
    fp = os.path.join(traj_path, "saliency_map.pkl")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Missing saliency_map.pkl at {traj_path}")
    with open(fp, "rb") as f:
        sal = pickle.load(f)
    if not isinstance(sal, np.ndarray) or sal.ndim != 4 or sal.shape[1] != 1:
        raise ValueError(f"Unexpected saliency format at {fp}: shape={getattr(sal, 'shape', None)}")
    T, _, H, W = sal.shape
    obs_l: List[np.ndarray] = []
    next_l: List[np.ndarray] = []
    for t in range(T):
        hw = sal[t, 0]  # (H,W)
        obs_l.append(_squash_saliency_map(hw))
    # align like images
    obs_out = obs_l[:-1]
    next_out = obs_l[1:]
    return obs_out, next_out


def _process_images(traj_path: str) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]]]:
    """Process images at a trajectory level.

    Identifies directories named like "images0", "images1", ... (excluding depth images)
    and constructs observations and next_observations as aligned lists.
    """
    # 严格匹配形如 images<数字> 的目录，忽略文件和其他命名（如 images0_saliency.gif）
    names: List[Tuple[int, str]] = []
    for x in os.listdir(traj_path):
        p = os.path.join(traj_path, x)
        if not os.path.isdir(p):
            continue
        m = re.match(r"^images(\d+)$", x)
        if m and ("depth" not in x):
            names.append((int(m.group(1)), x))
    image_dir_names = [name for _, name in sorted(names, key=lambda t: t[0])]
    image_paths = [os.path.join(traj_path, x) for x in image_dir_names]

    images_out: Dict[str, List[np.ndarray]] = defaultdict(list)

    if not image_paths:
        raise FileNotFoundError(f"No image directories found under {traj_path}")

    tlen = len(glob.glob(os.path.join(image_paths[0], "im_*.jpg")))

    for i, name in enumerate(image_dir_names):
        for t in range(tlen):
            img = _squash_image(os.path.join(image_paths[i], f"im_{t}.jpg"))
            images_out[name].append(img)

    images_out = dict(images_out)

    obs: Dict[str, List[np.ndarray]] = {}
    next_obs: Dict[str, List[np.ndarray]] = {}

    for name in image_dir_names:
        obs[name] = images_out[name][:-1]
        next_obs[name] = images_out[name][1:]

    return obs, next_obs


def _process_state(traj_path: str) -> Tuple[List[Any], List[Any]]:
    fp = os.path.join(traj_path, "obs_dict.pkl")
    with open(fp, "rb") as f:
        x = pickle.load(f)
    return x["full_state"][:-1], x["full_state"][1:]


def _process_time(traj_path: str) -> Tuple[List[Any], List[Any]]:
    fp = os.path.join(traj_path, "obs_dict.pkl")
    with open(fp, "rb") as f:
        x = pickle.load(f)
    return x["time_stamp"][:-1], x["time_stamp"][1:]


def _process_actions(traj_path: str) -> List[Any]:
    fp = os.path.join(traj_path, "policy_out.pkl")
    with open(fp, "rb") as f:
        act_list = pickle.load(f)
    if isinstance(act_list[0], dict):
        act_list = [x["actions"] for x in act_list]
    return act_list


def _process_data_collection(path: str, train_ratio: float = 0.9) -> Tuple[List[dict], List[dict], List[int], List[int]]:
    """Processes one dated data-collection directory containing raw trajectories.

    Returns train dicts, val dicts, train rewards, val rewards.
    """
    if "lmdb" in path:
        logging.warning(f"Skipping {path} because it appears to be an lmdb directory")
        return [], [], [], []

    all_dicts_train: List[dict] = []
    all_dicts_val: List[dict] = []
    all_rews_train: List[int] = []
    all_rews_val: List[int] = []

    # Data collected prior to 7-23 has a delay of 1, otherwise a delay of 0
    try:
        date_time = datetime.strptime(path.split("/")[-1], "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        # If directory does not match the expected date format, skip it
        logging.info(f"Skipping non-dated directory: {path}")
        return [], [], [], []

    latency_shift = date_time < datetime(2021, 7, 23)

    search_path = os.path.join(path, "raw", "traj_group*", "traj*")
    all_traj = glob.glob(search_path)
    if not all_traj:
        logging.info(f"No trajectories found in {search_path}")
        return [], [], [], []

    random.shuffle(all_traj)

    num_traj = len(all_traj)
    for itraj, traj_path in tqdm.tqdm(enumerate(all_traj), total=num_traj):
        try:
            output: Dict[str, Any] = {}

            listing = os.listdir(traj_path)

            assert "obs_dict.pkl" in listing, f"Missing obs_dict.pkl in {traj_path}: {listing}"
            assert "policy_out.pkl" in listing, f"Missing policy_out.pkl in {traj_path}: {listing}"

            obs_images, next_obs_images = _process_images(traj_path)
            if getattr(ARGS, "saliency", False):
                try:
                    sal_obs, sal_next = _process_saliency(traj_path)
                except Exception as e:
                    logging.error(f"Saliency load failed for {traj_path}: {e}")
                    sal_obs, sal_next = None, None
            actions = _process_actions(traj_path)
            state, next_state = _process_state(traj_path)
            time_stamp, next_time_stamp = _process_time(traj_path)
            terminals = [0] * len(actions)

            if "lang.txt" in listing:
                with open(os.path.join(traj_path, "lang.txt")) as f:
                    lang = [l.strip() for l in f if "confidence" not in l]
            else:
                lang = [""]

            output["observations"] = obs_images
            output["observations"]["state"] = state
            output["observations"]["time_stamp"] = time_stamp
            output["next_observations"] = next_obs_images
            output["next_observations"]["state"] = next_state
            output["next_observations"]["time_stamp"] = next_time_stamp
            if getattr(ARGS, "saliency", False) and sal_obs is not None and sal_next is not None:
                # store as HWC float32 with C=1 to be consistent with images HWC
                output["observations"]["saliency"] = sal_obs
                output["next_observations"]["saliency"] = sal_next

            # Convert dict-of-lists to list-of-dicts per timestep
            output["observations"] = [
                dict(zip(output["observations"], t)) for t in zip(*output["observations"].values())
            ]
            output["next_observations"] = [
                dict(zip(output["next_observations"], t)) for t in zip(*output["next_observations"].values())
            ]

            output["actions"] = actions
            output["terminals"] = terminals
            output["language"] = lang

            # Shift according to camera latency, if applicable
            if latency_shift:
                output["observations"] = output["observations"][1:]
                output["next_observations"] = output["next_observations"][1:]
                output["actions"] = output["actions"][:-1]
                output["terminals"] = terminals[:-1]

            labeled_rew = copy.deepcopy(output["terminals"])[:]
            if len(labeled_rew) >= 2:
                labeled_rew[-2:] = [1, 1]

            traj_len = len(output["observations"])
            assert len(output["next_observations"]) == traj_len
            assert len(output["actions"]) == traj_len
            assert len(output["terminals"]) == traj_len
            assert len(labeled_rew) == traj_len

            if itraj < int(num_traj * train_ratio):
                all_dicts_train.append(output)
                all_rews_train.append(labeled_rew)
            else:
                all_dicts_val.append(output)
                all_rews_val.append(labeled_rew)

        except FileNotFoundError as e:
            logging.error(e)
            continue
        except AssertionError as e:
            logging.error(e)
            continue

    return all_dicts_train, all_dicts_val, all_rews_train, all_rews_val


def _make_numpy(path: str, train_proportion: float) -> None:
    """Create output numpy files for one path containing dated directories."""
    dirname = os.path.abspath(path)
    # Replicate the original mapping logic for output path
    tail_parts = dirname.split(os.sep)[-(max(ARGS.depth - 1, 1)) :]
    outpath = os.path.join(ARGS.output_path, *tail_parts)

    if os.path.exists(outpath):
        if ARGS.overwrite:
            logging.info(f"Deleting existing directory: {outpath}")
            import shutil

            shutil.rmtree(outpath)
        else:
            logging.info(f"Skipping existing directory: {outpath}")
            return

    outpath_train = os.path.join(outpath, "train")
    outpath_val = os.path.join(outpath, "val")
    os.makedirs(outpath_train, exist_ok=True)
    os.makedirs(outpath_val, exist_ok=True)

    lst_train: List[dict] = []
    lst_val: List[dict] = []
    rew_train_l: List[int] = []
    rew_val_l: List[int] = []

    for dated_folder in os.listdir(path):
        curr_train, curr_val, rew_train, rew_val = _process_data_collection(
            os.path.join(path, dated_folder), train_ratio=train_proportion
        )
        lst_train.extend(curr_train)
        lst_val.extend(curr_val)
        rew_train_l.extend(rew_train)
        rew_val_l.extend(rew_val)

    if len(lst_train) == 0 or len(lst_val) == 0:
        logging.info(f"No data produced for {path}; skipping save.")
        return

    # Save using numpy's save (object arrays will be pickled)
    with open(os.path.join(outpath_train, "out.npy"), "wb") as f:
        np.save(f, lst_train)
    with open(os.path.join(outpath_val, "out.npy"), "wb") as f:
        np.save(f, lst_val)

    # Keeping parity with original: reward npy files were deprecated


def _main() -> None:
    parser = argparse.ArgumentParser(description="Convert BridgeData raw to numpy (no TF/absl)")
    parser.add_argument("--input_path", type=str, required=True, help="Input path root")
    parser.add_argument("--output_path", type=str, required=True, help="Output path root")
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help=(
            "Number of directories deep to traverse to the dated directory. "
            "Looks for {input_path}/dir_1/dir_2/.../dir_{depth-1}/2022-01-01_00-00-00/..."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument(
        "--train_proportion",
        type=float,
        default=0.9,
        help="Proportion of data to use for training (rather than val)",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes")
    parser.add_argument("--im_size", type=int, default=128, help="Image size for squashing")
    parser.add_argument("--saliency", action="store_true", help="Also load and export saliency maps if present")

    args = parser.parse_args()

    global ARGS
    ARGS = args

    _configure_logging()

    assert ARGS.depth >= 1, "--depth must be >= 1"

    # Each path is a directory that contains dated directories
    # Replicate the original glob construction using repeated "*" path segments
    glob_pattern_root = os.path.join(ARGS.input_path, *("*" * (ARGS.depth - 1)))
    paths = glob.glob(glob_pattern_root)

    worker_fn = partial(_make_numpy, train_proportion=ARGS.train_proportion)

    if ARGS.num_workers and ARGS.num_workers > 1:
        with Pool(ARGS.num_workers) as p:
            list(tqdm.tqdm(p.imap(worker_fn, paths), total=len(paths)))
    else:
        # Fallback to serial processing
        for pth in tqdm.tqdm(paths):
            worker_fn(pth)


if __name__ == "__main__":
    _main()


