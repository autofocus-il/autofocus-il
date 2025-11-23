import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from einops import rearrange
import datetime
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from models.linear_models import BCActor, Encoder, weight_init, VectorQuantizer

Task_to_Route = {
    "Mixed_": {
        "train": [
            (r, s)
            for r in [24759, 25857, 24211, 3100, 2416, 3472, 25863, 26408, 27494, 24258]
            for s in range(200, 220)
        ],  # route_id, seed
        "test": [
            (r, 400)
            for r in sorted(
                [24759, 25857, 24211, 3100, 2416, 3472, 25863, 26408, 27494, 24258]
            )
        ],
        "test_unseen": [
            (r, 400)
            for r in sorted(
                [18305, 1852, 24224, 3099, 3184, 3464, 27529, 26401, 2215, 25951]
            )
        ],
    },
    "ParkingCutIn_": {
        "train": [(24759, s) for s in range(200, 220)],  # route_id, seed
        "test": [(24759, 400)],
        "test_unseen": [(18305, 400)],
    },  # start seed for test
    "AccidentTwoWays_": {
        "train": [(25857, s) for s in range(200, 220)],  # route_id, seed
        "test": [(25857, 400)],
        "test_unseen": [(1852, 400)],
    },  # start seed for test
    "DynamicObjectCrossing_": {
        "train": [(24211, s) for s in range(200, 220)],  # route_id, seed
        "test": [(24211, 400)],  # start seed for test
        "test_unseen": [(24224, 400)],
    },  # start seed for test
    "CrossingBicycleFlow_": {
        "train": [(3100, s) for s in range(200, 220)],  # route_id, seed
        "test": [(3100, 400)],  # start seed for test
        "test_unseen": [(3099, 400)],
    },  # start seed for test
    "VanillaNonSignalizedTurnEncounterStopsign_": {
        "train": [(2416, s) for s in range(200, 220)],  # route_id, seed
        "test": [(2416, 400)],  # start seed for test
        "test_unseen": [(3184, 400)],
    },  # start seed for test
    "VehicleOpensDoorTwoWays_": {
        "train": [(3472, s) for s in range(200, 220)],  # route_id, seed
        "test": [(3472, 400)],  # start seed for test
        "test_unseen": [(3464, 400)],
    },  # start seed for test
    "PedestrianCrossing_": {
        "train": [(25863, s) for s in range(200, 220)],  # route_id, seed
        "test": [(25863, 400)],  # start seed for test
        "test_unseen": [(27529, 400)],
    },  # start seed for test
    "MergerIntoSlowTrafficV2_": {
        "train": [(26408, s) for s in range(200, 220)],  # route_id, seed
        "test": [(26408, 400)],  # start seed for test
        "test_unseen": [(26401, 400)],
    },  # start seed for test
    "BlockedIntersection_": {
        "train": [(27494, s) for s in range(200, 220)],  # route_id, seed
        "test": [(27494, 400)],  # start seed for test
        "test_unseen": [(2215, 400)],
    },  # start seed for test
    "HazardAtSideLaneTwoWays_": {
        "train": [(24258, s) for s in range(200, 220)],  # route_id, seed
        "test": [(24258, 400)],  # start seed for test
        "test_unseen": [(25951, 400)],
    },  # start seed for test
}


MAX_EPISODES = {k: len(v["train"]) for k, v in Task_to_Route.items()}


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def plot_gaze_and_obs(gaze, obs, save_path=None):
    # Create data for the plots
    y1 = gaze

    y2 = obs
    if obs.dtype == torch.uint8:
        y2 = obs.to(torch.float32) / 255

    y3 = (y1 * y2).to(torch.float32)

    if len(y3.shape) == 3:
        if y3.shape[0] == 3:
            y3 = rearrange(y3, "c h w -> h w c")
        elif y3.shape[0] == 1:
            y3 = y3[0]

    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # Plot 1
    ax1.imshow(y1, cmap="gray", vmax=1.0, vmin=0.0)
    ax1.set_title("pure gaze")

    # Plot 2
    ax2.imshow(y2, cmap="gray", vmax=1.0, vmin=0.0)
    ax2.set_title("pure obs")

    # Plot 3
    ax3.imshow(y3, cmap="gray", vmax=1.0, vmin=0.0)
    ax3.set_title("merged gaze and obs")

    ax1.set_xticks([])
    ax2.set_xticks([])
    ax3.set_xticks([])

    ax1.set_yticks([])
    ax2.set_yticks([])
    ax3.set_yticks([])
    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    # Show the plots
    plt.show()


def pad_and_convert_to_tensor(data, max_points=5):
    """
    向量化版本，比原版快3-5倍
    使用(-1, -1)标记无效填充位置，避免与真实(0,0)gaze点冲突
    """
    import numpy as np

    T = len(data)
    # 初始化为(-1, -1)而不是(0, 0)，标记无效位置
    result = np.full((T, max_points, 2), -1.0, dtype=np.float32)

    for i, frame in enumerate(data):
        # 处理单点情况
        if isinstance(frame[0], float):
            frame = [frame]

        # 快速处理：直接切片和赋值
        n_points = min(len(frame), max_points)
        if n_points > 0:
            # 转numpy并取前2列
            frame_array = np.array(frame[:n_points])
            if frame_array.ndim == 1:
                frame_array = frame_array.reshape(1, -1)
            result[i, :n_points, :] = frame_array[:, :2]

    return torch.from_numpy(result)


# ===================== DDP and Training Utilities =====================


def setup_ddp(storage_devices=None):
    """Initialize DDP environment with P2P support

    Args:
        storage_devices: List of storage device strings (e.g., ['cuda:0', 'cuda:1'])

    Returns:
        tuple: (is_ddp, rank, world_size, compute_device)
    """
    # Check if DDP environment variables are set
    if "RANK" not in os.environ:
        print("DDP environment not set, running in single GPU mode")
        return False, 0, 1, "cuda:0"

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Dynamically determine compute GPU pool
    # Total available GPUs = torch.cuda.device_count()
    # Storage GPUs = len(storage_devices)
    # Compute GPUs = Total - Storage
    total_gpus = torch.cuda.device_count()
    num_storage_gpus = len(storage_devices) if storage_devices else 0
    num_compute_gpus = total_gpus - num_storage_gpus

    if num_compute_gpus <= 0:
        raise ValueError(
            f"Not enough GPUs: total={total_gpus}, storage={num_storage_gpus}, compute={num_compute_gpus}"
        )

    if world_size > num_compute_gpus:
        raise ValueError(
            f"Too many processes: world_size={world_size} > compute_gpus={num_compute_gpus}"
        )

    # Extract storage device IDs to avoid conflicts
    storage_ids = set()
    if storage_devices:
        storage_ids = {int(dev.split(":")[1]) for dev in storage_devices}

    # Create compute pool avoiding storage devices: assign from available GPUs
    compute_pool = []
    for i in range(total_gpus):
        if i not in storage_ids:
            compute_pool.append(f"cuda:{i}")
        if len(compute_pool) >= num_compute_gpus:
            break

    if len(compute_pool) < world_size:
        raise ValueError(
            f"Not enough non-storage compute GPUs: available={len(compute_pool)}, needed={world_size}"
        )

    compute_device = compute_pool[rank % len(compute_pool)]

    if rank == 0:
        print(
            f"Dynamic GPU allocation: Total={total_gpus}, Compute={num_compute_gpus}, Storage={num_storage_gpus}"
        )
        print(f"Compute pool: {compute_pool}")

    # Set device for this process
    device_id = int(compute_device.split(":")[1])
    torch.cuda.set_device(device_id)

    # Enable P2P access if available - dynamically parse storage devices
    if storage_devices:
        # Extract logical device IDs from storage_devices list
        storage_ids = [int(dev.split(":")[1]) for dev in storage_devices]

        for storage_id in storage_ids:
            if storage_id < torch.cuda.device_count():  # Check if device exists
                try:
                    if torch.cuda.can_device_access_peer(device_id, storage_id):
                        torch.cuda.set_device(device_id)
                        # Note: PyTorch doesn't expose cudaDeviceEnablePeerAccess directly
                        # P2P will be enabled automatically when needed
                        print(f"P2P available: cuda:{device_id} <-> cuda:{storage_id}")
                except Exception as e:
                    pass  # Silently skip failed P2P checks

    if rank == 0:
        print(f"Rank {rank}: Using compute device {compute_device}")

    return True, rank, world_size, compute_device


def create_models(args, device):
    """Create encoder, actor and gaze-specific models

    Args:
        args: Training arguments containing model configuration
        device: Target device for models

    Returns:
        tuple: (encoder, actor, encoder_agil, gril_gaze_coord_predictor, quantizer)
    """
    # Calculate input channels based on gaze method
    coeff = 2 if args.gaze_method == "ViSaRL" else 1
    input_channels = coeff * args.stack * (1 if args.grayscale else 3)

    # Main encoder
    encoder = Encoder(
        input_channels,
        args.embedding_dim,
        args.num_hiddens,
        args.num_residual_layers,
        args.num_residual_hiddens,
    ).to(device)

    # Additional encoder for AGIL method
    encoder_agil = None
    if args.gaze_method == "AGIL":
        encoder_agil = Encoder(
            args.stack * (1 if args.grayscale else 3),
            args.embedding_dim,
            args.num_hiddens,
            args.num_residual_layers,
            args.num_residual_hiddens,
        ).to(device)

    # Actor head matching train_bc_fast.py
    encoder_output_dim = 20 * 38 * args.embedding_dim
    action_dim = getattr(args, "action_dim", 7)
    actor = BCActor(encoder_output_dim, args.z_dim, action_dim).to(device)

    # GRIL gaze coordinate predictor
    gril_gaze_coord_predictor = None
    if args.gaze_method == "GRIL":
        gril_gaze_coord_predictor = nn.Sequential(
            nn.Linear(args.z_dim, args.z_dim),
            nn.ReLU(),
            nn.Linear(args.z_dim, args.max_gaze_points * 2),
        )
        gril_gaze_coord_predictor.apply(weight_init)
        gril_gaze_coord_predictor.to(device)

    # VQ-VAE quantizer for Oreo method
    quantizer = None
    if args.dp_method == "Oreo":
        quantizer = VectorQuantizer(args.embedding_dim, args.num_embeddings, 0.25).to(
            device
        )

        # Load pre-trained VQ-VAE weights
        vqvae_path = f"trained_models/vqvae_models/{args.task}/seed_1_stack_{args.stack}_ep_{args.num_episodes}_grayscale_{args.grayscale}/model.torch"

        if os.path.exists(vqvae_path):
            for p in quantizer.parameters():
                p.requires_grad = False
            vqvae_dict = torch.load(vqvae_path, map_location="cpu", weights_only=True)
            encoder.load_state_dict(
                {k[9:]: v for k, v in vqvae_dict.items() if "_encoder" in k}
            )
            quantizer.load_state_dict(
                {k[11:]: v for k, v in vqvae_dict.items() if "_quantizer" in k}
            )
        else:
            print(f"Warning: VQ-VAE model not found at {vqvae_path}")

    return encoder, actor, encoder_agil, gril_gaze_coord_predictor, quantizer


def create_lr_scheduler(optimizer, args, train_loader_len):
    """Create cosine learning rate scheduler with linear warmup

    This implements a Diffusion Policy-style scheduler:
    - Linear warmup from 0 to peak LR over warmup_steps
    - Cosine decay from peak LR to eta_min for remaining steps

    Args:
        optimizer: PyTorch optimizer
        args: Training arguments
        train_loader_len: Length of training data loader

    Returns:
        tuple: (scheduler, batch_scheduler_update)
    """
    import math
    from torch.optim.lr_scheduler import LambdaLR

    # Calculate total training steps
    total_steps = (train_loader_len // args.gradient_accumulation_steps) * args.n_epochs
    warmup_steps = getattr(args, "num_warmup_steps", 0)

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup from 0 to 1.0
            return step / float(max(1, warmup_steps))
        else:
            # Cosine decay to eta_min
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_lr_ratio = args.eta_min / args.lr
            return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

    scheduler = LambdaLR(optimizer, lr_lambda)
    batch_scheduler_update = True  # Update every step for smooth transitions

    return scheduler, batch_scheduler_update


def create_save_info(args):
    """Create save directory and tag based on training configuration

    Args:
        args: Training arguments

    Returns:
        tuple: (save_tag, save_dir)
    """
    save_tag = f"s{args.seed}_n{args.num_episodes}_stack{args.stack}_gray{args.grayscale}_bs{args.bs}_lr{args.lr:.0e}"

    # Add cosine scheduler info to save tag
    save_tag += f"_cosine_warmup{args.num_warmup_steps}_eta{args.eta_min}"

    # Add pseudo tag to save name
    if args.pseudo:
        save_tag += "_pseudo"

    # Add optimization tags to save name
    if args.use_amp:
        save_tag += "_amp"
    if args.use_compile:
        save_tag += "_compile"
    if args.gradient_accumulation_steps > 1:
        save_tag += f"_acc{args.gradient_accumulation_steps}"

    # Add gaze method specific tags
    if args.gaze_method in ["Reg", "Teacher"]:
        save_tag += f"_gaze_{args.gaze_method}_beta_{args.gaze_beta}_lambda_{args.gaze_lambda}_dist_{args.prob_dist_type}"
    elif args.gaze_method in ["ViSaRL", "Mask", "AGIL"]:
        save_tag += f"_gaze_{args.gaze_method}"
    elif args.gaze_method in ["GRIL"]:
        save_tag += f"_gaze_{args.gaze_method}_lambda_{args.gaze_lambda}"
    elif args.gaze_method == "Contrastive":
        save_tag += f"_gaze_{args.gaze_method}_threshold_{args.gaze_contrastive_threshold}_lambda_{args.gaze_lambda}"

    # Add gaze mask parameters
    if args.gaze_method != "None":
        save_tag += f"_sig{args.gaze_mask_sigma}_co{args.gaze_mask_coeff}"

    # Add DP method
    if args.dp_method in ["Oreo", "IGMD", "GMD"]:
        save_tag += f"_dp_{args.dp_method}"

    if args.add_path:
        save_tag += "_" + args.add_path

    now_time = datetime.datetime.now()
    save_dir = "{}_{}".format(now_time.strftime("%Y_%m_%d_%H_%M_%S"), save_tag)

    return save_tag, save_dir


def get_training_args():
    """Parse training arguments for GABRIL DDP training

    Returns:
        argparse.Namespace: Parsed arguments
    """
    import argparse

    parser = argparse.ArgumentParser(description="P2P DDP Training for GABRIL")

    # Basic arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", type=str, default="Mixed_")
    parser.add_argument(
        "--datapath", type=str, default="/path/to/dataset/bench2drive220/"
    )
    parser.add_argument("--stack", type=int, default=2)
    parser.add_argument("--grayscale", type=bool, default=True)

    # Training arguments
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=50,
        help="Number of episodes to train on (minimum 50 for multi-GPU)",
    )
    parser.add_argument(
        "--force_ddp",
        action="store_true",
        help="Force DDP mode even with small datasets",
    )
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)

    # P2P architecture arguments
    parser.add_argument(
        "--storage_devices",
        nargs="+",
        default=["cuda:0", "cuda:1", "cuda:2"],
        help="Storage GPU devices for data preprocessing",
    )
    parser.add_argument(
        "--compute_device",
        type=str,
        default="cuda:3",
        help="Compute GPU device (will be overridden by rank in DDP mode)",
    )

    # Model arguments
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_hiddens", type=int, default=128)
    parser.add_argument("--num_residual_layers", type=int, default=2)
    parser.add_argument("--num_residual_hiddens", type=int, default=32)
    parser.add_argument("--z_dim", type=int, default=256)

    # Gaze arguments
    parser.add_argument(
        "--gaze_method",
        type=str,
        default="None",
        choices=[
            "None",
            "Teacher",
            "Reg",
            "Mask",
            "Contrastive",
            "ViSaRL",
            "AGIL",
            "GRIL",
        ],
        help="Gaze method; Use Reg or Teacher for GABRIL",
    )
    parser.add_argument(
        "--dp_method", type=str, default="None", choices=["None", "Oreo", "IGMD", "GMD"]
    )
    parser.add_argument(
        "--gaze_mask_sigma",
        type=float,
        default=30.0,
        help="Sigma of the Gaussian for the gaze mask",
    )
    parser.add_argument(
        "--gaze_mask_coeff",
        type=float,
        default=0.8,
        help="Base coefficient of the Gaussian for the gaze mask",
    )
    parser.add_argument(
        "--gaze_ratio",
        type=float,
        default=1.0,
        help="Ratio of episodes to use for gaze prediction",
    )
    parser.add_argument(
        "--gaze_beta", type=float, default=50.0, help="Softmax temperature for GABRIL"
    )
    parser.add_argument(
        "--gaze_lambda", type=float, default=10, help="Loss coefficient hyperparameter"
    )
    parser.add_argument(
        "--gaze_contrastive_threshold",
        type=float,
        default=10,
        help="Contrastive loss margin hyperparameter for the Contrastive method",
    )
    parser.add_argument(
        "--prob_dist_type", type=str, default="MSE", choices=["MSE", "TV", "KL", "JS"]
    )
    parser.add_argument("--max_gaze_points", default=5, type=int)
    parser.add_argument(
        "--gaze_predictor_path",
        type=str,
        default="",
        help="Path to pre-trained gaze predictor",
    )

    # VQ-VAE and Oreo arguments
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=512,
        help="Number of embeddings for VQ-VAE",
    )
    parser.add_argument(
        "--oreo_num_mask", type=int, default=4, help="Number of masks for Oreo method"
    )
    parser.add_argument(
        "--oreo_prob", type=float, default=0.5, help="Probability for Oreo masking"
    )

    # Optimization arguments
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--use_amp", type=bool, default=True)

    # Cosine scheduler with linear warmup (Diffusion Policy style)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help="Learning rate scheduler (fixed to cosine with warmup)",
    )
    parser.add_argument(
        "--eta_min",
        type=float,
        default=1e-6,
        help="Minimum learning rate for cosine decay",
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=500,
        help="Number of linear warmup steps before cosine decay",
    )

    parser.add_argument("--use_compile", type=bool, default=False)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--local-rank", type=int, default=0, help="Local rank for distributed training"
    )

    # Additional arguments that might be passed but not used
    parser.add_argument(
        "--pseudo",
        action="store_true",
        default=False,
        help="Using pseudo labels (compatibility argument)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of data loading workers"
    )
    parser.add_argument(
        "--add_path", type=str, default=None, help="Additional path tag for saving"
    )

    # Save and result save directory
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--result_save_dir", type=str, default="./results_p2p")

    # LR Finder arguments
    parser.add_argument(
        "--auto_lr_finder",
        action="store_true",
        default=True,
        help="Run LR range test before DDP wrapping and use suggested lr",
    )
    parser.add_argument(
        "--finder_end_lr",
        type=float,
        default=1.0,
        help="Upper bound for LR range test (exp mode)",
    )
    parser.add_argument(
        "--finder_max_iter",
        type=int,
        default=200,
        help="Max iterations for LR range test",
    )

    args = parser.parse_args()
    return args
