#!/bin/bash

# Configuration
DEVICES="2,3,4,5,7"
NPROC_PER_NODE=5
TASK_LIST=(
  # "lift_carrot_mixed"
  "pull_pot_mixed"
  "pull_pot_100"
  "put_carrot_in_pot_100"
)

# Common parameters
COMMON_ARGS="
  algo=bc
  algo.encoder=resnet34
  algo.model.use_proprio=true
  algo.data.obs_horizon=2
  saliency.enabled=false
  saliency.weight=5
  saliency.alpha=0.7
  batch_size=3200
  num_steps=10000
  eval_interval=2000
  save_interval=2000
  log_interval=10
"


# Run training for each task
for task in "${TASK_LIST[@]}"; do
  echo "Starting training for task: $task"
  CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node=$NPROC_PER_NODE train_hydra.py \
    bridgedata=$task \
    $COMMON_ARGS
  
  echo "Completed training for task: $task"
  echo "----------------------------------------"
done

echo "All training tasks completed!"
