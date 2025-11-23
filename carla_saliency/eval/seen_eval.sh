#!/bin/bash
# carla server: ${CARLA_ROOT}/CarlaUE4.sh -quality-level=Epic -world-port=6000 -carla-rpc-port=3000 -RenderOffScreen

# parameters: ./seen_eval.sh [model_path] [traffic-manager-port] [port] [cuda-device] [confounded_flag]
# example: ./seen_eval.sh pseudo_gmd 3000 6000 0 true
# route list:[2416,3100,3472,24211,24258,24759,25857,25863,26408,27494]
routes=(2416 3100 3472 24211 24258 24759 25857 25863 26408 27494)

run_dir=/path/to/project_root/runs/Mixed_
# todo: change dir
cd /path/to/project_root/carla_saliency/eval

# Check if model parameter is provided
if [ $# -eq 0 ]; then
    model=pseudo_gmd
    echo "Using default model: $model"
else
    model=$1
    echo "Using specified model: $model"
fi

# Check if traffic-manager-port parameter is provided
if [ $# -lt 2 ]; then
    traffic_manager_port=3000
    echo "Using default traffic-manager-port: $traffic_manager_port"
else
    traffic_manager_port=$2
    echo "Using specified traffic-manager-port: $traffic_manager_port"
fi

# Check if port parameter is provided
if [ $# -lt 3 ]; then
    port=6000
    echo "Using default port: $port"
else
    port=$3
    echo "Using specified port: $port"
fi

# Check if CUDA_VISIBLE_DEVICES parameter is provided
if [ $# -lt 4 ]; then
    cuda_device=0
    echo "Using default CUDA_VISIBLE_DEVICES: $cuda_device"
else
    cuda_device=$4
    echo "Using specified CUDA_VISIBLE_DEVICES: $cuda_device"
fi

# Check if confounded parameter is provided (5th arg, true/false)
if [ $# -lt 5 ]; then
    confounded=false
    echo "Using default confounded: $confounded"
else
    confounded=$5
    echo "Using specified confounded: $confounded"
fi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$cuda_device

# Function to generate random seeds
generate_random_seeds() {
    local seeds=()
    for i in {1..2}; do
        # Generate random number between 0-999
        seed=$((RANDOM % 1000))
        seeds+=($seed)
    done
    echo "${seeds[@]}"
}

for route_id in "${routes[@]}"; do
    echo "Running route: $route_id"
    
    # Generate 4 random seeds for current route
    random_seeds=($(generate_random_seeds))
    echo "Seeds used: ${random_seeds[@]}"
    
    for seed in "${random_seeds[@]}"; do
        echo "  Running seed: $seed"
        if [ "$confounded" = true ]; then
            python env_manager.py --agent=BC --params_path=$run_dir/$model --seed=$seed --routes-id=$route_id --video_path=auto --traffic-manager-port=$traffic_manager_port --port=$port --confounded
        else
            python env_manager.py --agent=BC --params_path=$run_dir/$model --seed=$seed --routes-id=$route_id --video_path=auto --traffic-manager-port=$traffic_manager_port --port=$port
        fi
        echo "  seed $seed finished"
    done
    
    echo "Route $route_id finished"
    echo "----------------------------------------"
done

echo "All routes finished!"
