#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, io, json, glob, pickle, sys, warnings, tempfile, re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from datasets import Dataset as HFDataset, Features as HFFeatures, Value, Sequence
from huggingface_hub import HfApi

# LeRobot (only version number; do not build LeRobotDataset locally to avoid repo_id validation)
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.datasets.video_utils import VideoFrame, encode_video_frames

from safetensors.torch import save_file
import torch
import av

# ---------------- helpers ----------------
def _save_meta_bundle(info: dict, stats_json: dict, epi_idx: dict, meta_dir: Path):
    """Write meta/info.json, meta/stats.json, meta/episode_data_index.safetensors"""
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))
    (meta_dir / "stats.json").write_text(json.dumps(stats_json, indent=2))
    save_file({k: torch.tensor(v, dtype=torch.int64) for k, v in epi_idx.items()},
              meta_dir / "episode_data_index.safetensors")

def _upload_folder(repo_id: str, local: Path, remote: str, revision="main"):
    HfApi().upload_folder(folder_path=str(local), path_in_repo=remote,
                          repo_id=repo_id, repo_type="dataset", revision=revision)

def _vcodec_alias(vcodec: str) -> str:
    """Map shorthand to ffmpeg encoder name"""
    m = {"h264": "libx264", "hevc": "libx265", "libsvtav1": "libsvtav1"}
    return m.get(vcodec, vcodec)

# ---------------- BDV2 discovery & loading ----------------
def discover_collections(root: Path) -> List[Tuple[str, Path]]:
    pairs = []
    if not root.exists():
        raise FileNotFoundError(f"bdv2_root does not exist: {root}")
    for task in sorted(os.listdir(root)):
        tdir = root / task
        if not tdir.is_dir(): continue
        for sub in sorted(os.listdir(tdir)):
            cdir = tdir / sub
            if (cdir / "raw").is_dir():
                pairs.append((task, cdir))
    return pairs

def load_traj(traj_dir: Path) -> Dict[str, Any]:
    with open(traj_dir / "obs_dict.pkl", "rb") as f:
        try: obs = pickle.load(f)
        except Exception: obs = pickle.load(f, encoding="latin1")
    with open(traj_dir / "policy_out.pkl", "rb") as f:
        try: pol = pickle.load(f)
        except Exception: pol = pickle.load(f, encoding="latin1")
    return {"obs": obs, "pol": pol}

def _sorted_imgs(img_dir: Path) -> List[str]:
    files = glob.glob(str(img_dir / "*"))
    if not files: return []
    def extract_number(filepath):
        filename = Path(filepath).stem
        nums = re.findall(r"\d+", filename)
        return int(nums[-1]) if nums else 0
    return sorted(files, key=extract_number)

def read_frames_from(obs: Dict[str, Any], raw_root: Path, traj_dir: Path) -> List[np.ndarray]:
    """Compatible with three types: relative path list (relative to raw or traj), images0 directory, inline array"""
    rels = obs.get("images0_files", []) or obs.get("images0_paths", [])
    if rels:
        frames = []
        for rel in rels:
            cand1 = raw_root / rel
            cand2 = traj_dir / rel
            p = cand1 if cand1.exists() else cand2
            with Image.open(p) as im:
                frames.append(np.array(im.convert("RGB"), np.uint8))
        return frames
    img_dir = traj_dir / "images0"
    if img_dir.is_dir():
        frames = []
        for p in _sorted_imgs(img_dir):
            with Image.open(p) as im:
                frames.append(np.array(im.convert("RGB"), np.uint8))
        return frames
    arr = obs.get("images0", None)
    if arr is not None:
        arr = np.asarray(arr)
        assert arr.ndim == 4, f"images0 expect (T,H,W,C), got {arr.shape}"
        return [arr[t].astype(np.uint8) for t in range(arr.shape[0])]
    raise FileNotFoundError("No images found (images0_* / images0/ / inline images0).")

def parse_actions_and_xforms(policy_out) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """Support list[dict{actions(7,), new_robot_transform(4x4), delta_robot_transform(4x4), policy_type}]"""
    if isinstance(policy_out, list) and policy_out and isinstance(policy_out[0], dict):
        acts, newTs, dTs = [], [], []
        ptype = policy_out[0].get("policy_type", None)
        for step in policy_out:
            acts.append(np.asarray(step["actions"], np.float32))
            if "new_robot_transform" in step:
                newTs.append(np.asarray(step["new_robot_transform"], np.float64).reshape(16).astype(np.float32))
            if "delta_robot_transform" in step:
                dTs.append(np.asarray(step["delta_robot_transform"], np.float64).reshape(16).astype(np.float32))
        actions = np.stack(acts, 0)
        new_T = np.stack(newTs, 0) if len(newTs)==len(acts) and newTs else None
        delta_T = np.stack(dTs, 0) if len(dTs)==len(acts) and dTs else None
        return actions, new_T, delta_T, ptype
    if isinstance(policy_out, dict) and "actions" in policy_out:
        return np.asarray(policy_out["actions"], np.float32), None, None, policy_out.get("policy_type")
    return np.asarray(policy_out, np.float32), None, None, None

# ---------------- core conversion ----------------
def convert_bdv2_to_lerobot(
    bdv2_root: Path,
    out_root: Path,
    fps: int = 10,
    vcodec: str = "libsvtav1",
    test_run: bool = False,
) -> Tuple[Dict[str, List[int]], Dict[str, Any], Dict[str, Any]]:
    # Target directory structure (v2.x)
    data_dir   = out_root / "data" / "chunk-000"
    videos_dir = out_root / "videos" / "chunk-000" / "observation.images.cam0"
    meta_dir   = out_root / "meta"
    data_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    epi_idx = {"from": [], "to": []}
    cursor, epi = 0, 0
    policy_types = set()

    # Task and episode meta info
    tasks_list: List[str] = []
    task_to_idx: Dict[str, int] = {}
    episodes_meta: List[dict] = []

    # Statistics/feature shape collection
    dims_seen: Dict[str, int] = {}
    img_hw: Optional[Tuple[int,int]] = None
    action_dim_first: Optional[int] = None

    # Collect all trajectory directories
    all_trajs = []
    for task, cdir in discover_collections(bdv2_root):
        raw_root = cdir / "raw"
        for grp in sorted(os.listdir(raw_root)):
            gdir = raw_root / grp
            if not gdir.is_dir(): continue
            for traj in sorted(os.listdir(gdir)):
                tdir = gdir / traj
                if not tdir.is_dir(): continue
                all_trajs.append((task, raw_root, tdir))
    if test_run:
        all_trajs = all_trajs[:3]

    vcodec_real = _vcodec_alias(vcodec)

    for task, raw_root, tdir in tqdm(all_trajs, desc="Converting trajectories", unit="traj"):
        try:
            data = load_traj(tdir)
            obs, pol = data["obs"], data["pol"]

            frames = read_frames_from(obs, raw_root=raw_root, traj_dir=tdir)
            if img_hw is None and len(frames):
                h, w = frames[0].shape[:2]
                img_hw = (int(h), int(w))

            actions, new_T, delta_T, ptype = parse_actions_and_xforms(pol)
            if ptype: policy_types.add(ptype)

            env_done = list(obs.get("env_done", []))
            t_stamps = list(obs.get("time_stamp", []))
            T = min(len(frames), len(actions),
                    len(env_done) if env_done else 10**9,
                    len(t_stamps) if t_stamps else 10**9)
            if T == 0:
                continue

            frames, actions = frames[:T], actions[:T]
            if action_dim_first is None:
                action_dim_first = int(np.asarray(actions[0]).shape[-1])
            if new_T is not None: new_T = new_T[:T]
            if delta_T is not None: delta_T = delta_T[:T]
            if env_done: env_done = env_done[:T]
            if t_stamps:
                t0 = t_stamps[0]
                t_stamps = [float(x - t0) for x in t_stamps[:T]]

            # Task index
            if task not in task_to_idx:
                task_to_idx[task] = len(tasks_list)
                tasks_list.append(task)
            task_idx = task_to_idx[task]

            episodes_meta.append({"episode_index": epi, "tasks": [task], "length": T})

            # ---- Encode video (one segment per trajectory) ----
            vname = f"episode_{epi:06d}.mp4"
            vpath = videos_dir / vname
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                for i, frame in enumerate(frames):
                    Image.fromarray(frame).save(temp_path / f"frame_{i:06d}.png", compress_level=1)
                encode_video_frames(temp_path, vpath, fps=fps, vcodec=vcodec_real,
                                    overwrite=True, log_level=av.logging.ERROR)

            epi_idx["from"].append(cursor)

            # ---- Frame-by-frame table entry (episode level parquet) ----
            epi_cols: Dict[str, list] = {
                "task": [], "task_index": [], "episode_index": [], "frame_index": [],
                "timestamp": [], "next.done": [],
                "action": [], "observation.images.cam0": [],
            }

            rel_video_path = str(Path("videos") / "chunk-000" / "observation.images.cam0" / vname)

            def _push_opt(col, val):
                epi_cols.setdefault(col, []).append(val)
                if isinstance(val, np.ndarray) and val.ndim == 1:
                    dims_seen[col] = int(val.shape[0])

            for t in range(T):
                ts = float(t) / float(fps)
                epi_cols["task"].append(task)
                epi_cols["task_index"].append(task_idx)
                epi_cols["episode_index"].append(epi)
                epi_cols["frame_index"].append(t)
                epi_cols["timestamp"].append(ts)
                epi_cols["next.done"].append(bool(env_done[t]) if env_done else (t == T-1))
                epi_cols["action"].append(np.asarray(actions[t], np.float32))
                epi_cols["observation.images.cam0"].append({"path": rel_video_path, "timestamp": ts})

                # Additional observations (flatten/scalar)
                def _get(name, arr, get_t=lambda z:z[t]):
                    if arr is None: return None
                    if name == "observation.eef_transform":
                        return np.asarray(get_t(arr), np.float64).reshape(16).astype(np.float32)
                    if name == "observation.task_stage":
                        return np.array([int(get_t(arr))], np.int32)
                    if name in ("observation.t_get_obs","observation.time_stamp"):
                        return np.array([float(get_t(arr))], np.float32)
                    return np.asarray(get_t(arr), np.float32)

                for k_src, k_dst in [
                    ("qpos","observation.qpos"),
                    ("qvel","observation.qvel"),
                    ("joint_effort","observation.joint_effort"),
                    ("state","observation.state"),
                    ("full_state","observation.full_state"),
                    ("desired_state","observation.desired_state"),
                    ("high_bound","observation.high_bound"),
                    ("low_bound","observation.low_bound"),
                    ("eef_transform","observation.eef_transform"),
                    ("task_stage","observation.task_stage"),
                    ("t_get_obs","observation.t_get_obs"),
                    ("time_stamp","observation.time_stamp"),
                ]:
                    v = _get(k_dst, obs.get(k_src, None))
                    if v is not None:
                        _push_opt(k_dst, v)

                if new_T is not None:
                    _push_opt("action.new_robot_transform", new_T[t])
                if delta_T is not None:
                    _push_opt("action.delta_robot_transform", delta_T[t])

                cursor += 1

            # features (determined by actual existing columns in this episode)
            a_dim = int(np.asarray(epi_cols["action"][0]).shape[-1])
            features: Dict[str, Any] = {
                "task": Value("string"),
                "task_index": Value("int32"),
                "episode_index": Value("int32"),
                "frame_index": Value("int32"),
                "timestamp": Value("float32"),
                "next.done": Value("bool"),
                "action": Sequence(length=a_dim, feature=Value("float32")),
                "observation.images.cam0": VideoFrame(),
            }

            def _add_seq(name, dim, as_int=False, scalar=False):
                if name in epi_cols and len(epi_cols[name]) > 0:
                    if scalar:
                        features[name] = Value("int32" if as_int else "float32")
                    else:
                        features[name] = Sequence(length=dim, feature=Value("float32"))

            # Dynamic columns (write only if exists)
            for nm, dim, as_int, scalar in [
                ("observation.qpos",6, False, False),
                ("observation.qvel",6, False, False),
                ("observation.joint_effort",6, False, False),
                ("observation.state",7, False, False),
                ("observation.full_state",7, False, False),
                ("observation.desired_state",7, False, False),
                ("observation.high_bound",5, False, False),
                ("observation.low_bound",5, False, False),
                ("observation.eef_transform",16, False, False),
                ("observation.task_stage",1, True,  True),
                ("observation.t_get_obs",1, False, True),
                ("observation.time_stamp",1, False, True),
                ("action.new_robot_transform",16, False, False),
                ("action.delta_robot_transform",16, False, False),
            ]:
                _add_seq(nm, dim, as_int=as_int, scalar=scalar)

            epi_ds = HFDataset.from_dict(epi_cols, features=HFFeatures(features)).with_format(None)
            epi_ds.to_parquet(str(data_dir / f"episode_{epi:06d}.parquet"))

            epi_idx["to"].append(cursor)
            epi += 1

        except Exception as e:
            print(f"[WARN] skip {tdir}: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()
            continue

    if cursor == 0:
        raise RuntimeError("No samples found under bdv2_root.")

    total_episodes = len(episodes_meta)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": int(fps),
        "video": True,
        "encoding": {"vcodec": vcodec_real, "pix_fmt": "yuv420p", "g": 2, "crf": 30},
        "policy_types": sorted(list(policy_types)),
        "splits": {"train": f"0:{total_episodes}"},
        "chunks_size": 1000,
        "data_path":  "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/observation.images.cam0/episode_{episode_index:06d}.mp4",
        "total_episodes": total_episodes,
        "total_frames": int(cursor),
        "total_tasks": len(tasks_list),
        "total_chunks": 1,
    }
    if img_hw:
        info["cameras"] = {"observation.images.cam0": {"resolution": [img_hw[0], img_hw[1]]}}
    if action_dim_first:
        info["action_dim"] = int(action_dim_first)

    extra_meta = {"tasks": tasks_list, "episodes": episodes_meta, "dims_seen": dims_seen}
    return epi_idx, info, extra_meta

# ---------------- end2end: save + push ----------------
def build_and_push(
    bdv2_root: Path,
    repo_id: str,
    out_root: Path,
    fps: int = 10,
    vcodec: str = "libsvtav1",
    private: bool = False,
    push: bool = True,
    overwrite: bool = False,
    test_run: bool = False,
):
    if overwrite and out_root.exists():
        import shutil
        shutil.rmtree(out_root)
        print(f"Deleted existing output directory: {out_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Convert (save episode by episode + return info/indices/meta)
    epi_idx, info, extra_meta = convert_bdv2_to_lerobot(
        bdv2_root, out_root, fps=fps, vcodec=vcodec, test_run=test_run
    )

    # 2) Statistics (minimal available: write meta/stats.json)
    stats_json = {
        "num_episodes": len(extra_meta.get("episodes", [])),
        "num_frames": info.get("total_frames", 0),
        "dims": extra_meta.get("dims_seen", {}),
        "action_dim": info.get("action_dim", None),
    }

    # 3) Save meta (info.json / stats.json / episode_data_index.safetensors)
    meta_dir = out_root / "meta"
    _save_meta_bundle(info, stats_json, epi_idx, meta_dir)

    # 3.1) Save tasks.jsonl and episodes.jsonl
    tasks_path = meta_dir / "tasks.jsonl"
    episodes_path = meta_dir / "episodes.jsonl"
    with open(tasks_path, "w") as f:
        for i, name in enumerate(extra_meta.get("tasks", [])):
            f.write(json.dumps({"task_index": i, "task": name}) + "\n")
    with open(episodes_path, "w") as f:
        for ep in extra_meta.get("episodes", []):
            f.write(json.dumps(ep) + "\n")

    # 4) Push (data / videos / meta)
    if push:
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        _upload_folder(repo_id, out_root / "data",   "data")
        _upload_folder(repo_id, out_root / "videos", "videos")
        _upload_folder(repo_id, out_root / "meta",   "meta")
        print(f"[OK] uploaded to hub: {repo_id}")

    print(f"[OK] local dataset at: {out_root}")

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--bdv2_root", type=Path, required=True)
    p.add_argument("--repo_id", type=str, required=True)
    p.add_argument("--out_root", type=Path, default=Path("/tmp/bdv2_to_lerobot"))
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--vcodec", type=str, default="libsvtav1",
                   choices=["libsvtav1","h264","hevc"],
                   help="Video encoder: libsvtav1/h264/hevc (will map to actual ffmpeg encoder name)")
    p.add_argument("--private", action="store_true")
    p.add_argument("--no_push", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="Delete existing output directory and restart")
    p.add_argument("--test_run", action="store_true", help="Test run, process only first 3 trajectories")
    args = p.parse_args()

    build_and_push(
        bdv2_root=args.bdv2_root,
        repo_id=args.repo_id,
        out_root=args.out_root,
        fps=args.fps,
        vcodec=args.vcodec,
        private=args.private,
        push=not args.no_push,
        overwrite=args.overwrite,
        test_run=args.test_run,
    )

if __name__ == "__main__":
    main()
