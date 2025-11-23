# run torch version eval
python eval.py \
    --runs_root /path/to/torch_runs/bridgedata_torch \
    --goal_type bc \
    --im_size 256 \
    --video_save_path /path/to/torch_runs/videos \
    --ip localhost \
    --port 5556 \
    --num_timesteps 120 \
    --act_exec_horizon 1 \
    --deterministic \
    --show_image
 
# run numpy data convert 
python bridge_torch/data/bdv2_to_numpy.py \
  --input_path /path/to/dataset/bdv2 \
  --output_path /path/to/dataset/bdv2_numpy \
  --depth 2 --num_workers 8 --train_proportion 0.99 --im_size 256 --saliency

# run training with nohup
nohup bash -c 'CUDA_VISIBLE_DEVICES=2,4,5,7 torchrun --nproc_per_node=4 train_hydra.py \
  -m bridgedata=lift_carrot_mixed,pull_pot_100,pull_pot_mixed,put_carrot_in_pot_100 \
  algo=bc \
  algo.encoder=resnet101 \
  algo.model.use_proprio=true \
  algo.data.obs_horizon=2 \
  saliency.enabled=false \
  saliency.weight=0 \
  saliency.alpha=0 \
  batch_size=2400 \
  num_steps=20000 \
  eval_interval=1000 \
  save_interval=1000 \
  log_interval=10' > logs/multirun_$(date +%F_%T).log 2>&1 &

# run test training with test dataset
CUDA_VISIBLE_DEVICES=2 python train_hydra.py \
  --config-name test_run \
  data_path=/path/to/test_dataset/bdv2_numpy \
  save_dir=/path/to/torch_runs/bridgedata_torch_test \
  bridgedata=lift_carrot_100\
  algo=bc \
  algo.encoder=resnet101 \
  algo.data.obs_horizon=2 \
  algo.model.use_proprio=true \
  saliency.enabled=true \
  saliency.weight=5 \
  saliency.alpha=0.7 \
  batch_size=10 \
  num_steps=100 \
  eval_interval=10 \
  save_interval=50 \
  log_interval=10 \
  ddp.enabled=false

# multi run test
# run training with nohup (Hydra, with log redirection)
nohup bash -c 'CUDA_VISIBLE_DEVICES=2,4,5,7 torchrun --nproc_per_node=4 train_hydra.py \
  -m bridgedata=lift_carrot_mixed,pull_pot_100,pull_pot_mixed,put_carrot_in_pot_100 \
  data_path=/path/to/dataset/bdv2_numpy \
  save_dir=/path/to/torch_runs/bridgedata_torch_test \
  algo=bc \
  algo.encoder=resnet101 \
  algo.model.use_proprio=true \
  algo.data.obs_horizon=2 \
  saliency.enabled=false \
  saliency.weight=0 \
  saliency.alpha=0 \
  batch_size=20 \
  num_steps=10 \
  eval_interval=10 \
  save_interval=10 \
  log_interval=1' > logs/multirun_$(date +%F_%T).log 2>&1 &

# upload eval results to google drive
rclone copy /scr/litian/torch_runs/bridgedata_torch_0909_eval litian_LIRA:Data/ --progress
