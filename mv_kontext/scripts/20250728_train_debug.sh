export FLUX_SCHNELL=/group/40005/gzhiwang/pretrainedFLUX.1-schnell/flux1-schnell.safetensors 
export FLUX_DEV=/group/40005/gzhiwang/pretrained/FLUX.1-dev/flux1-dev.safetensors
export AE=/group/40005/gzhiwang/pretrained/FLUX.1-schnell/ae.safetensors
export FLUX_REDUX=/group/40005/public_models/share/image_generation/FLUX.1-Redux-dev/
export FLUX_KONTEXT=/group/40005/public_models/FLUX.1-Kontext-dev/

export XDG_CACHE=/group/40005/share/zhaoyangzhang/PretrainedCache
export TORCH_HOME=/group/40005/share/zhaoyangzhang/PretrainedCache
export HF_HOME=/group/40005/share/zhaoyangzhang/PretrainedCache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


export ENV_VENUS_PROXY=http://zzachzhang:rmdRjCXJAhvOXxhE@vproxy.woa.com:31289
export NO_PROXY=localhost,.woa.com,.oa.com,.tencent.com,.tencentcloudapi.com
export HTTP_PROXY=$ENV_VENUS_PROXY
export HTTPS_PROXY=$ENV_VENUS_PROXY
export no_proxy=$NO_PROXY
export http_proxy=$ENV_VENUS_PROXY
export https_proxy=$ENV_VENUS_PROXY


export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth1 # canavas
# export NCCL_SOCKET_IFNAME=bond1 # online coding
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO


HOST_NUM=1
INDEX=$1
CHIEF_IP=127.0.0.1
HOST_GPU_NUM=1

echo "compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: DEEPSPEED
fsdp_config: {}
machine_rank: $INDEX
main_process_ip: $CHIEF_IP
main_process_port: 27394
main_training_function: main
num_machines: $HOST_NUM
num_processes: $(($HOST_NUM*$HOST_GPU_NUM))
mixed_precision: bf16
use_cpu: false" > /tmp/default_config.yaml

cat /tmp/default_config.yaml

pwd


export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=0


# Get current date and time in format YYYYMMDD_HHMM
CURRENT_DATETIME=$(date +"%Y%m%d_%H%M")

accelerate launch  --config_file /tmp/default_config.yaml  mv_kontext/training_code/train_lora_flux.py \
    --pretrained_model_name_or_path=$FLUX_KONTEXT \
    --meta_path="data/navi/metainfo/navi_v1.5_metainfo_reorg.json" \
    --data_dir="data/navi/navi_v1.5" \
    --min_recon_num=1 \
    --max_recon_num=8 \
    --output_dir="mv_kontext/output/${CURRENT_DATETIME}_train_v1" \
    --mixed_precision="bf16" \
    --train_batch_size=2 \
    --dataloader_num_workers 4 \
    --rank=64 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing \
    --learning_rate=5e-5 \
    --report_to="wandb" \
    --wandb_dir="mv_kontext/output" \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=50000 \
    --offload \
    --seed="0" \
    --log_dataset_samples \
    --max_points=200000 \
    --conf_threshold 10 20 30 40 50 \
    --conf_threshold_value=1.0 \
    --apply_mask \
    --radius 0.01 \
    --points_per_pixel 100 \
    --bin_size 0 \
    --proportion_empty_prompts 0.1 \
    --checkpointing_steps 1 \
    --validation_steps 5 \
    --test_meta_path="data/navi/navi_test_data/test_metainfo_20250714_223602.json" \
    --test_data_dir="data/navi/navi_test_data" \
    --checkpoints_total_limit 1

if [ $? != 0 ]; then
   echo "Fail! Exit with 1"
   cd /group/40034/yaoweili/code/
   python multi_occupy.py
   exit 1
else
   echo "Success! Exit with 0"
   cd /group/40034/yaoweili/code/
   python multi_occupy.py
   exit 0
fi