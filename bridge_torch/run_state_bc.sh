#!/bin/bash
# bc training without vision data
# Configuration
DEVICES=2

TASK_LIST=(
  "lift_carrot_mixed"
  "pull_pot_100"
  "pull_pot_mixed"
  "put_carrot_in_pot_100"
)

COMMON_ARGS="
  algo=state_bc
  algo.encoder=none
  algo.model.use_proprio=true
  algo.data.obs_horizon=2
  batch_size=4000
  num_steps=10000
  eval_interval=50
  save_interval=1000
  log_interval=10
"

# Run training for each task
for task in "${TASK_LIST[@]}"; do
  echo "Starting training for task: $task"
  CUDA_VISIBLE_DEVICES=$DEVICES python train_hydra.py \
    bridgedata=$task \
    $COMMON_ARGS
  
  echo "Completed training for task: $task"
  echo "----------------------------------------"
done

echo "All training tasks completed!"
