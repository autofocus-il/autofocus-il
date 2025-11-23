#!/usr/bin/env bash
set -euo pipefail

# DDP launcher for Behavior Cloning (BC)
# Unified script for single/multi experiments
# Usage examples:
#   bash carla_saliency/train/run_ddp_train_bc.bash
#   NPROC=4 MASTER_PORT=29512 bash carla_saliency/train/run_ddp_train_bc.bash data.batch_size=128 optimizer.lr=3e-4
#   MULTI_RUN=1 bash carla_saliency/train/run_ddp_train_bc.bash  # Run built-in multi-run combinations
#   MULTI_RUN=1 METHOD_PAIRS="None:GMD,Reg:GMD,ViSaRL:None" bash carla_saliency/train/run_ddp_train_bc.bash  # Override combinations

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

# GPU selection and comm envs
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3,4,7}

# Torchrun params
NPROC=${NPROC:-3}
MASTER_PORT=${MASTER_PORT:-29501}
export MASTER_ADDR=127.0.0.1

export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
# Reduce C++ backtrace symbolization spam and lower C++ log level
export TORCH_SHOW_CPP_STACKTRACES=${TORCH_SHOW_CPP_STACKTRACES:-0}
export TORCH_DISABLE_ADDR2LINE=${TORCH_DISABLE_ADDR2LINE:-1}
export TORCH_CPP_LOG_LEVEL=${TORCH_CPP_LOG_LEVEL:-ERROR}
# Prefer loopback IPv4 to avoid IPv6 localhost issues
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
# Additional network settings for stability
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-1}
## Quiet down NCCL logging (INFO -> WARN)
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,ENV}
# Remove deprecated var if present to silence warnings
unset NCCL_ASYNC_ERROR_HANDLING || true
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
# Memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Silence Python warnings unless explicitly overridden
export PYTHONWARNINGS=${PYTHONWARNINGS:-ignore}



# If NPROC > number of visible GPUs, automatically scale down to avoid multiple processes on the same card
IFS=',' read -r -a __gpu_arr <<< "${CUDA_VISIBLE_DEVICES}"
__num_visible_gpus=${#__gpu_arr[@]}
if [[ ${NPROC} -gt ${__num_visible_gpus} ]]; then
  echo "[WARN] NPROC(${NPROC}) > visible GPUs(${__num_visible_gpus}); set NPROC=${__num_visible_gpus}"
  NPROC=${__num_visible_gpus}
fi

echo "Launching DDP BC training on GPUs: ${CUDA_VISIBLE_DEVICES} (nproc_per_node=${NPROC})"
echo "Master address: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Environment setup complete, starting training..."

# Define single run function
run_one() {
  local extra_overrides=("$@")
  torchrun \
    --nnodes=1 \
    --nproc_per_node="${NPROC}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    carla_saliency/train/train_bc.py \
    --config-name=train_bc \
    "${extra_overrides[@]}"
}

# Multi-experiment combinations (can be overridden by env var METHOD_PAIRS, comma separated; default built-in combinations)
__DEFAULT_METHOD_PAIRS=(
  # "None:GMD" 
  # "ViSaRL:None" 
  # "GRIL:None" 
  "None:None" 
  # "AGIL:None"
  # "Reg:GMD"
  # "Reg:None"
)

# Default gaze.ratio combinations (can be overridden by env var RATIOS, comma separated, e.g. "0.1,0.25,0.5,0.75")
__DEFAULT_GAZE_RATIOS=(
  1
)





if [[ "${MULTI_RUN:-1}" == "1" ]]; then
  # If user provides METHOD_PAIRS comma separated, parse as array
  METHOD_PAIRS_ARR=()
  if [[ -n "${METHOD_PAIRS:-}" ]]; then
    IFS=',' read -r -a METHOD_PAIRS_ARR <<< "${METHOD_PAIRS}"
  else
    METHOD_PAIRS_ARR=("${__DEFAULT_METHOD_PAIRS[@]}")
  fi

  # Parse RATIOS (gaze.ratio). Allow overriding via env var RATIOS, comma separated
  RATIOS_ARR=()
  if [[ -n "${RATIOS:-}" ]]; then
    IFS=',' read -r -a RATIOS_ARR <<< "${RATIOS}"
  else
    RATIOS_ARR=("${__DEFAULT_GAZE_RATIOS[@]}")
  fi

  echo "=== Multi-run DDP BC Training ==="
  total_pairs=${#METHOD_PAIRS_ARR[@]}
  total_ratios=${#RATIOS_ARR[@]}
  total_runs=$(( total_pairs * total_ratios ))
  echo "Will run ${total_pairs} method pairs x ${total_ratios} ratios = ${total_runs} runs"
  echo "Method pairs:"
  for pair in "${METHOD_PAIRS_ARR[@]}"; do
    IFS=':' read -r gaze_method dropout_method <<< "${pair}"
    echo "  - gaze.method=${gaze_method}, dropout.method=${dropout_method}"
  done
  echo "Ratios: ${RATIOS_ARR[*]}"
  echo ""

  __run_counter=0
  for pair in "${METHOD_PAIRS_ARR[@]}"; do
    IFS=':' read -r gaze_method dropout_method <<< "${pair}"
    for ratio in "${RATIOS_ARR[@]}"; do
      __run_counter=$((__run_counter + 1))
      echo "=== Run ${__run_counter}/${total_runs}: gaze.method=${gaze_method}, dropout.method=${dropout_method}, gaze.ratio=${ratio} ==="
      if ! run_one gaze.method="${gaze_method}" dropout.method="${dropout_method}" gaze.ratio="${ratio}" "$@"; then
        echo "ERROR: Run ${__run_counter} failed (gaze=${gaze_method}, dropout=${dropout_method}, ratio=${ratio})"
        echo "Continuing with next experiment..."
        sleep 5
        continue
      fi
      echo "=== Run ${__run_counter} completed successfully ==="
      echo ""
      sleep 5
    done
  done
  echo "=== All experiments completed! ==="
else
  # Single run: supports passing Hydra overrides externally (e.g. gaze.method/ dropout.method etc.)
  run_one "$@"
fi
