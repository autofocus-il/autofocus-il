Official repository for AutoFocus-IL: VLM-based Saliency Maps for Data-Efficient
Visual Imitation Learning without Extra Human Annotations

+ [Project Page](https://autofocus-il.github.io/)
+ [Paper](https://autofocus-il.github.io/assets/pdf/autofocus-il.pdf)

# 1. CARLA Experiments

Ensure you have the CARLA simulator **(0.9.15)** installed, see guidance [here](https://carla.readthedocs.io/en/latest/start_quickstart/#carla-installation).

## 1.1 Python Environment
Install dependencies for the saliency and training modules:

```bash
# Create environment (optional but recommended)
conda create -n autofocus-il python=3.10
conda activate autofocus-il

# Install requirements
pip install -r carla_saliency/requirements.txt
```

## 1.2 Configuration

Before running any scripts, you must configure paths and API keys.

### 1.2.1 Configure VLM API Provider

The pipeline uses a Vision-Language Model API for scene understanding. You need to configure:

1. **API Key** (environment variable)
2. **Provider settings** in config files

#### Example: Using SiliconFlow

[SiliconFlow](https://www.siliconflow.com/) provides fast inference for open-source VLMs like Qwen.

**Step 1: Set your API key**
```bash
export OpenAI_API_KEY="sk-..."
```

**Step 2: Update pipeline config** (e.g., `saliency_pipeline/configs/bdv2/pipeline.yaml` or `bench2drive/pipeline.yaml`)

```yaml
global_desc:
  model: "Qwen/Qwen2.5-VL-72B-Instruct"
  api_provider: "OpenAI"  # SiliconFlow uses OpenAI-compatible API
  # ... other settings

api:
  OpenAI:
    base_url: "https://api.siliconflow.cn/v1"
    api_key_env: "OpenAI_API_KEY"
    default_key: ""  # Leave empty to force env var usage
```

### 1.2.2 Update Paths

The configuration files currently use **placeholder paths** (e.g., `/path/to/dataset`, `/path/to/project_root`). You must update these to match your local environment.

**Key files to modify:**
*   `saliency_pipeline/configs/bench2drive/pipeline.yaml`: Set `dataset_dir` and `output_dir`.
*   `carla_saliency/configs/bench2drive_to_hdf5.yaml`: Set `dataset_root` and `output_hdf5`.
*   `carla_saliency/configs/train_bc.yaml`: Set `data.dataset_path`.

## 1.3 Data Generation (Saliency Pipeline)

This step processes raw driving data to generate saliency maps, bounding boxes, and VLM-filtered annotations.

**Entry Point:** `saliency_pipeline/run_pipeline.py`

### 1.3.1 Run Command
```bash
# Run the full pipeline (Global Desc -> VLM Filter -> BBox to Dataset)
python saliency_pipeline/run_pipeline.py --config saliency_pipeline/configs/bench2drive/pipeline.yaml
```

### 1.3.2 Configuration & Parameters

*   **Config File:** `saliency_pipeline/configs/bench2drive/pipeline.yaml`
    *   **`dataset.bench2drive.dataset_dir`**: Path to your raw Bench2Drive dataset.
    *   **`global_desc.model`**: Vision-Language Model to use.
    *   **`vlm_filter.text_prompt`**: The text prompt used for object detection/filtering.
    *   **`run.mode`**: Set to `all`, `single_route`, or `single_seed` to control scope.

## 1.4 Data Preprocessing (HDF5 Conversion)

Convert the processed dataset (images + saliency .pt files) into a Robomimic-compatible HDF5 format for training.

**Entry Point:** `carla_saliency/data_utils/bench2drive_to_hdf5.py`

### 1.4.1 Run Command

```bash
python -m carla_saliency.data_utils.bench2drive_to_hdf5 --config carla_saliency/configs/bench2drive_to_hdf5.yaml
```

### 1.4.2 Configuration & Parameters

*   **Config File:** `carla_saliency/configs/bench2drive_to_hdf5.yaml`
    *   **`dataset_root`**: Path to the dataset processed in Step 2 (containing `observations.pt`, `gaze.pt`, etc.).
    *   **`output_hdf5`**: Destination path for the resulting `.hdf5` file.
    *   **`include_gaze`**: Boolean flags to include/exclude specific saliency types.

## 1.5 Model Training (Behavior Cloning)

Train the imitation learning policy using the generated HDF5 dataset. Supports Distributed Data Parallel (DDP).

**Entry Point:** `carla_saliency/train/run_ddp_train_bc.bash`

### 1.5.1 Run Command

```bash
# Single Run
bash carla_saliency/train/run_ddp_train_bc.bash

# Multi-Run (Sweep over methods)
MULTI_RUN=1 METHOD_PAIRS="None:GMD,Reg:None" bash carla_saliency/train/run_ddp_train_bc.bash
```

### 1.5.2 Configuration & Parameters

*   **Config File:** `carla_saliency/configs/train_bc.yaml` (Hydra config)
    *   **`data.dataset_path`**: Point this to your generated `.hdf5` file.
    *   **`gaze.method`**: Saliency method (e.g., `ViSaRL`, `GMD`, `None`).
    *   **`gaze.ratio`**: Ratio of saliency data usage.
*   **Environment Variables (in bash script):**
    *   `CUDA_VISIBLE_DEVICES`: Select GPUs.
    *   `NPROC`: Number of GPUs/processes per node.

## 1.6 Evaluation

Evaluate the trained model in the CARLA simulator.

**Entry Point:** `carla_saliency/eval/seen_eval.sh` (or `unseen_eval.sh`)

### 1.6.1 Run Command

First, start your CARLA server in a separate terminal:
```bash
${CARLA_ROOT}/CarlaUE4.sh -quality-level=Epic -world-port=6000 -carla-rpc-port=3000 -RenderOffScreen
```

Then, run the evaluation script:
```bash
# Usage: ./seen_eval.sh [model_path] [tm_port] [port] [gpu_id] [confounded_flag]
bash carla_saliency/eval/seen_eval.sh model_dir_name 3000 6000 0 false

# eval in confounded mode
bash carla_saliency/eval/seen_eval.sh model_dir_name 3000 6000 0 true

# eval in unseen mode
bash carla_saliency/eval/unseen_eval.sh model_dir_name 3000 6000 0 false
```

### 1.6.2 Configuration & Parameters

*   **Script:** `carla_saliency/eval/seen_eval.sh`
    *   **`routes` array**: Modify the list of route IDs to evaluate specific scenarios.
    *   **`traffic_manager_port` / `port`**: Ensure these match your CARLA server settings.
    *   **`confounded`**: Set to `true` to enable confounded evaluation settings.

# 2. Robot Experiments (WidowX)

Use the `bridge_torch` toolchain to train and evaluate real-robot policies on BridgeData V2.

## 2.1 Data Preprocessing

**Convert BridgeData V2 to NumPy (required before training)**

```bash
python bridge_torch/data/bdv2_to_numpy.py \
  --input_path /abs/path/to/raw/bdv2 \
  --output_path /abs/path/to/processed/bdv2_numpy \
  --depth 2 --num_workers 8 --train_proportion 0.99 \
  --im_size 256 --saliency
```

Replace both path flags with your dataset locations before running.

## 2.2 Model Training

### 2.2.1 Multi-GPU Training with Hydra

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 bridge_torch/train_hydra.py \
  -m bridgedata=lift_carrot_mixed,pull_pot_100 \
  data_path=/abs/path/to/processed/bdv2_numpy \
  save_dir=/abs/path/to/runs/bridge_torch \
  algo=bc \
  algo.encoder=resnet101 \
  algo.model.use_proprio=true \
  algo.data.obs_horizon=2 \
  saliency.enabled=false \
  batch_size=2400 \
  num_steps=20000 \
  eval_interval=1000 \
  save_interval=1000 \
  log_interval=10
```

Update `data_path`, `save_dir`, and the `bridgedata` sweep to match the tasks and storage available on your machine.

### 2.2.2 Single-Task Training Helper Script

```bash
bash bridge_torch/run_bc.sh
```

Edit `DEVICES`, `TASK_LIST`, and the shared hyperparameters inside the script to reflect your GPUs and desired tasks before executing.

## 2.3 Robot Evaluation Setup (Server-Client)

To evaluate trained policies on the real WidowX robot, use the server-client architecture that isolates robot control from policy inference.

### 2.3.1 Prerequisites

1. **Robot Server Setup** (on robot control machine)  
   Follow `bridge_data_robot/README.md` to install dependencies and build the Docker environment:
   ```bash
   cd bridge_data_robot
   ./host_install.sh
   ./generate_usb_config.sh
   USB_CONNECTOR_CHART=$(pwd)/usb_connector_chart.yml docker compose up --build robonet
   ```

2. **Policy Client Setup** (can be same or different machine)  
   Install the `widowx_envs` package in your policy environment:
   ```bash
   cd bridge_data_robot/widowx_envs
   pip install -e .
   ```

### 2.3.2 Start Robot Server

In the robot control machine, launch the WidowX environment service:
```bash
cd bridge_data_robot
docker compose exec robonet bash -lic "widowx_env_service --server"
```

The server listens on `localhost:5556` by default. Use `--port` to change the port if needed.

### 2.3.3 Run Policy Evaluation

On the client machine (or same machine), run the evaluation script:
```bash
python bridge_torch/eval.py \
  --runs_root /abs/path/to/runs/bridge_torch \
  --goal_type bc \
  --im_size 256 \
  --video_save_path /abs/path/to/runs/videos \
  --ip localhost \
  --port 5556 \
  --num_timesteps 120 \
  --act_exec_horizon 1 \
  --deterministic \
  --show_image
```

**Key parameters:**
- `--ip`: IP address of the robot server (use actual IP if client is on a different machine)
- `--port`: Must match the server port (default: 5556)
- `--runs_root`: Directory containing trained model checkpoints
- `--video_save_path`: Where to save evaluation videos