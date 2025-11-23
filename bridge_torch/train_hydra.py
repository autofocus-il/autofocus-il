import os
import json
import signal
import sys
from functools import partial
import logging

import numpy as np
import torch
from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from torch.amp import autocast, GradScaler

from models.encoders import ResNetV1Bridge
from agents.bc import BCAgent, BCAgentConfig, MLP as MLP_BC, GaussianPolicy
from agents.state_bc import StateGaussianPolicy
from data.bridge_numpy import build_bridge_dataset, make_dataloader
from torch.utils.data.distributed import DistributedSampler
from common.wandb import WandBLogger
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


def _prep_device(device_flag: str) -> torch.device:
    if device_flag == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def save_checkpoint(save_dir: str, agent, step: int):
    _ensure_dir(save_dir)
    path = os.path.join(save_dir, f"ckpt_{step}.pt")
    model_state = agent.model.module.state_dict() if isinstance(agent.model, DDP) else agent.model.state_dict()
    payload = {"model": model_state, "step": agent.step}
    if hasattr(agent, "target_model") and agent.target_model is not None:
        tm_state = agent.target_model.module.state_dict() if isinstance(agent.target_model, DDP) else agent.target_model.state_dict()
        payload["target_model"] = tm_state
    torch.save(payload, path)
    return path


def _ddp_init_if_needed(device: torch.device):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Prioritize IPv4 loopback to avoid AF_INET6 issues on some systems where localhost resolves to IPv6
        ma = os.environ.get("MASTER_ADDR", "")
        if (not ma) or (ma.lower() == "localhost") or (ma == "::1"):
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ.setdefault("MASTER_PORT", "29500")
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
        return True, dist.get_rank(), dist.get_world_size(), local_rank
    return False, 0, 1, 0


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0

def basic_env_setup(cfg: DictConfig):
    pass

@hydra_main(config_path="./conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Basic logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    device = _prep_device(cfg.device)

    # Enable backend accelerations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # Suppress prints on non-main processes
    if not is_main_process():
        os.environ['WANDB_SILENT'] = 'true'
        _orig_print = print
        def _quiet(*args, **kwargs):
            if any(keyword in str(args) for keyword in ['rank', 'W0909', 'I0909', 'wandb']):
                return
            return _orig_print(*args, **kwargs)
        import builtins
        builtins.print = _quiet

    # Graceful shutdown flag and signal handlers
    stop_requested = {"val": False}
    def _signal_handler(signum, frame):
        stop_requested["val"] = True
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # init (optional) DDP
    ddp_cfg = cfg.ddp
    ddp_enabled = ddp_cfg.enabled
    ddp_active, rank, world_size, local_rank = _ddp_init_if_needed(device)
    if ddp_enabled and not ddp_active and torch.cuda.device_count() > 1:
        logging.warning("DDP enabled in config but process not launched with torchrun. Proceeding single-process.")

    # set up wandb (only on main process)
    wandb_logger = None
    save_root: str | None = None
    if is_main_process():
        wb_cfg = cfg.wandb
        wb_enabled = wb_cfg.enabled
        project_name = wb_cfg.project
        
        # Auto-generate experiment descriptor based on configuration parameters
        def generate_exp_descriptor(cfg):
            parts = []
            # Get task name
            parts.append(cfg.bridgedata.task_name)
            
            # Proprioception usage
            use_proprio = cfg.algo.model.use_proprio
            if use_proprio:
                if cfg.algo.encoder == "none":
                    parts.append("only_proprio")
                else:
                    parts.append("proprio")
            # Observation horizon
            obs_horizon = cfg.algo.data.obs_horizon
            parts.append(f"s{obs_horizon}")
            # Saliency configuration
            saliency_enabled = cfg.saliency.enabled
            
            if cfg.algo.encoder != "none":
                if saliency_enabled:
                    parts.append("saliency")
                else:
                    parts.append("no_saliency")

            # Encoder architecture
            parts.append(cfg.algo.encoder)
            
            return "_".join(parts)
        
        # Use custom name if provided, otherwise auto-generate
        custom_name = str(cfg.name).strip()
        if custom_name:
            exp_descriptor = custom_name
        else:
            exp_descriptor = generate_exp_descriptor(cfg)
        
        wandb_config = WandBLogger.get_default_config()
        cfg_plain = OmegaConf.to_container(cfg, resolve=True)
        wandb_config.update({"project": project_name, "exp_descriptor": exp_descriptor})
        if wb_enabled:
            wandb_logger = WandBLogger(wandb_config=wandb_config, variant=cfg_plain, debug=cfg.debug)
        else:
            wandb_logger = None

        unique_id = wandb_logger.config.unique_identifier if wandb_logger else None
        if not unique_id:
            from datetime import datetime as _dt
            unique_id = _dt.now().strftime("%Y%m%d_%H%M%S")

        save_root = os.path.join(str(cfg.save_dir), project_name, f"{exp_descriptor}_{unique_id}")
        _ensure_dir(save_root)
        snap = OmegaConf.to_container(cfg, resolve=True)
        with open(os.path.join(save_root, "config.json"), "w") as f:
            json.dump(snap, f, indent=2, ensure_ascii=False)
        logging.info("Config saved to %s", save_root)
        if wandb_logger and getattr(wandb_logger, "run", None):
            logging.info("WandB run initialized: %s", wandb_logger.run.url)

    # Build numpy datasets (required)
    train_seed = int(cfg.seed) + (rank if ddp_active else 0)
    val_seed = int(cfg.seed)

    per_rank_bs = int(cfg.batch_size // (world_size if ddp_active else 1))
    per_rank_val_bs = int(cfg.val_batch_size // (world_size if ddp_active else 1))
    per_rank_bs = max(per_rank_bs, 1)
    per_rank_val_bs = max(per_rank_val_bs, 1)

    bdcfg = cfg.bridgedata
    # Build torch-style datasets
    data_cfg = cfg.algo.data
    data_kwargs = OmegaConf.to_container(data_cfg, resolve=True) if isinstance(data_cfg, DictConfig) else dict(data_cfg)
    # inject saliency.alpha
    if cfg.saliency and cfg.saliency.alpha:
        data_kwargs.setdefault("saliency_alpha", float(cfg.saliency.alpha))

    # Determine if we need to load images based on encoder configuration
    enc_name = cfg.algo.encoder
    load_images = enc_name.lower() != "none"
    if is_main_process():
        logging.info("Loading images: %s (encoder: %s)", load_images, enc_name)

    train_ds = build_bridge_dataset(
        task_globs=bdcfg.include,
        data_root=str(cfg.data_path),
        split="train",
        seed=train_seed,
        bridgedata_cfg=bdcfg,
        dataset_kwargs=data_kwargs,
        load_images=load_images,
    )
    val_ds = build_bridge_dataset(
        task_globs=bdcfg.include,
        data_root=str(cfg.data_path),
        split="val",
        seed=val_seed,
        bridgedata_cfg=bdcfg,
        dataset_kwargs=data_kwargs,
        load_images=load_images,
    )

    # DataLoaders
    dl_cfg = cfg.dataloader
    dl_kwargs = OmegaConf.to_container(dl_cfg, resolve=True) if isinstance(dl_cfg, DictConfig) else dict(dl_cfg)
    train_sampler = None
    val_sampler = None
    if ddp_enabled and ddp_active:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    train_loader = make_dataloader(train_ds, batch_size=per_rank_bs, shuffle=(train_sampler is None), sampler=train_sampler, drop_last=False, **dl_kwargs)
    val_loader = make_dataloader(val_ds, batch_size=per_rank_val_bs, shuffle=False, sampler=val_sampler, drop_last=False, **dl_kwargs)

    # Peek one batch to infer shapes
    warmup_batch = next(iter(train_loader))
    obs_img = warmup_batch["observations"].get("image")
    obs_shape = obs_img.shape if obs_img is not None else None
    action_dim = int(warmup_batch["actions"].shape[-1])
    goal_img = warmup_batch.get("goals", {}).get("image") if isinstance(warmup_batch.get("goals"), dict) else None
    goal_shape = goal_img.shape if goal_img is not None else None
    if is_main_process():
        logging.info("Obs batch shape: %s, action_dim: %s, goal shape: %s", obs_shape, action_dim, goal_shape)

    # Build encoder if needed (state_bc needs no encoder)
    enc = None
    enc_name = cfg.algo.encoder
    if isinstance(enc_name, str) and enc_name.lower() != "none":
        enc = ResNetV1Bridge(arch=enc_name).to(device)

    # torch.compile on encoder if enabled (and not DDP-active)
    compile_enabled = cfg.compile.enabled
    compile_kwargs = cfg.compile.kwargs
    if enc is not None:
        if compile_enabled and hasattr(torch, 'compile') and not (ddp_enabled and ddp_active):
            enc = torch.compile(enc, **compile_kwargs)
            if is_main_process():
                logging.info("Encoder compiled with torch.compile")
        elif compile_enabled and (ddp_enabled and ddp_active) and is_main_process():
            logging.info("DDP active: skip compiling encoder; will compile full model instead")

    # Warm-up forward to initialize lazy shapes (use channels_last for conv perf)
    enc_feat = 0
    if enc is not None and obs_img is not None:
        _warm = warmup_batch["observations"]["image"].to(device)
        if _warm.dim() == 4:
            _warm = _warm.contiguous(memory_format=torch.channels_last)
        enc_feat = enc.adapt_to_input_channels(_warm)
    # Model/policy configuration (flattened Hydra schema)
    # model_cfg_obj = cfg.get("model", {}) if isinstance(cfg, DictConfig) else getattr(cfg, "model", {})
    # model_cfg = OmegaConf.to_container(model_cfg_obj, resolve=True) if isinstance(model_cfg_obj, DictConfig) else (dict(model_cfg_obj) if isinstance(model_cfg_obj, dict) else {})
    model_cfg = cfg.algo.model
    use_proprio_flag = bool(getattr(model_cfg, "use_proprio", False))
    # state_bc scenario forces use of proprio
    if isinstance(enc_name, str) and enc_name.lower() == "none":
        use_proprio_flag = True
    prop_dim = int(warmup_batch["observations"]["proprio"].shape[1]) if ("proprio" in warmup_batch["observations"]) else 0
    # print(f"warmup_batch['observations']: {warmup_batch['observations'].keys()}")
    # print(f"model_cfg: {model_cfg}")

    sal_cfg = cfg.saliency
    # Build BC agent/model
    print(f"enc_feat: {enc_feat}, prop_dim: {prop_dim}")
    mlp = MLP_BC(input_dim=(enc_feat + prop_dim) if (enc is not None) else prop_dim,
                 hidden_dims=tuple(model_cfg.hidden_dims), dropout_rate=float(model_cfg.dropout_rate))
    policy_kwargs = {
        "tanh_squash_distribution": bool(model_cfg.tanh_squash_distribution),
        "state_dependent_std": bool(model_cfg.state_dependent_std),
        "fixed_std": list(model_cfg.fixed_std) or None,
        "use_proprio": use_proprio_flag,
    }
    agent_cfg = BCAgentConfig(
        learning_rate=cfg.algo.optimizer.lr,
        weight_decay=cfg.algo.optimizer.weight_decay,
        warmup_steps=0.1*cfg.num_steps,
        decay_steps=cfg.num_steps, 
        scheduler=cfg.algo.scheduler,
        saliency=sal_cfg,
    )
    # Choose visual/state policy based on whether encoder exists
    if enc is None:
        agent = BCAgent(
            model=StateGaussianPolicy(
                mlp, action_dim=action_dim, tanh_squash_distribution=policy_kwargs["tanh_squash_distribution"],
                state_dependent_std=policy_kwargs["state_dependent_std"], fixed_std=policy_kwargs["fixed_std"],
            ),
            cfg=agent_cfg,
            action_dim=action_dim,
            device=device,
        )
    else:
        agent = BCAgent(
            model=GaussianPolicy(
                enc, mlp, action_dim=action_dim, **policy_kwargs
            ),
            cfg=agent_cfg,
            action_dim=action_dim,
            device=device,
        )

    # DDP wrap for full model if enabled
    if ddp_enabled and ddp_active:
        agent.model = DDP(
            agent.model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=bool(cfg.ddp.find_unused_parameters),
        )

    # AMP
    amp_cfg = cfg.amp
    amp_enabled = bool(amp_cfg.enabled)
    amp_dtype_name = str(amp_cfg.dtype).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in {"bf16", "bfloat16"} else torch.float16
    scaler = None
    if amp_enabled and device.type == "cuda":
        if amp_dtype is torch.float16:
            scaler = GradScaler(
                enabled=True,
                growth_factor=float(amp_cfg.growth_factor),
                backoff_factor=float(amp_cfg.backoff_factor),
                growth_interval=int(amp_cfg.growth_interval),
            )
            if is_main_process():
                logging.info("AMP FP16 enabled with GradScaler")
        else:
            scaler = None
            if is_main_process():
                logging.info("AMP BF16 enabled (no GradScaler)")
    elif amp_enabled and device.type != "cuda" and is_main_process():
        logging.warning("AMP requested but CUDA not available; running in FP32")

    # Todo: Optionally resume
    # if cfg.resume_path:
    #     ckpt = torch.load(str(cfg.resume_path), map_location=device)
    #     model_to_load = agent.model.module if isinstance(agent.model, DDP) else agent.model
    #     model_to_load.load_state_dict(ckpt["model"])  # type: ignore

    train_iter = iter(train_loader)

    progress = tqdm(range(int(cfg.num_steps)), disable=not is_main_process(), desc="Train", dynamic_ncols=True)
    try:
        for step in progress:
            if stop_requested["val"]:
                break
            try:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
            except KeyboardInterrupt:
                stop_requested["val"] = True
                break

            if amp_enabled:
                with autocast(device_type=device.type, dtype=amp_dtype):
                    info = agent.update(batch)
            else:
                info = agent.update(batch)

            if is_main_process() and (step + 1) % int(cfg.log_interval) == 0:
                show = {k: (float(v) if isinstance(v, (int, float)) else float(v)) for k, v in info.items() if k in ("ddpm_loss", "mse") and v is not None}
                progress.set_postfix(show, refresh=True)
                logging.info("[step %d] %s", step + 1, info)
                if wandb_logger:
                    wandb_logger.log({"training": info}, step=step + 1)

            if (step + 1) % int(cfg.eval_interval) == 0 and is_main_process():
                eval_batches_config = cfg.eval_batches
                if eval_batches_config is None:
                    logging.info("[eval %d] Skipped - eval_batches set to null", step + 1)
                else:
                    metrics = []
                    val_iter = iter(val_loader)
                    max_eval_batches = int(eval_batches_config) if int(eval_batches_config) > 0 else 50
                    seen = 0
                    val_pb = tqdm(total=int(max_eval_batches), desc="Eval", leave=False, disable=not is_main_process(), dynamic_ncols=True)
                    for vb in val_iter:
                        metrics.append(agent.get_debug_metrics(vb))
                        seen += 1
                        if seen >= max_eval_batches:
                            break
                        val_pb.update(1)
                    val_pb.close()
                    avg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]} if metrics else {}
                    logging.info("[eval %d] %s", step + 1, avg)
                    if wandb_logger:
                        wandb_logger.log({"validation": avg}, step=step + 1)

            if is_main_process() and (step + 1) % int(cfg.save_interval) == 0:
                ckpt_path = save_checkpoint(save_root or str(cfg.save_dir), agent, step + 1)
                logging.info("Saved checkpoint to %s", ckpt_path)
    except KeyboardInterrupt:
        stop_requested["val"] = True
    finally:
        if ddp_enabled and ddp_active and dist.is_initialized():
            dist.destroy_process_group()
        if is_main_process():
            if wandb_logger and getattr(wandb_logger, "run", None):
                wandb_logger.run.finish()  # type: ignore[attr-defined]
        if stop_requested["val"]:
            sys.exit(0)


if __name__ == "__main__":
    main()
