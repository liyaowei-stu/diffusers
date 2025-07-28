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


# 禁用InfiniBand
export NCCL_IB_DISABLE=1  # 改为1，禁用InfiniBand
# 禁用P2P通信，使用TCP
export NCCL_P2P_DISABLE=1


# 启用InfiniBand
# export NCCL_IB_DISABLE=0  # 启用InfiniBand网络
# export NCCL_IB_GID_INDEX=3  # 设置IB全局标识符索引
# export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3  # 指定使用的IB主机通道适配器
# export NCCL_IB_TC=184  # 设置IB传输类别
# export NCCL_IB_TIMEOUT=23  # IB操作超时时间
# export NCCL_IB_RETRY_CNT=7  # IB重试次数
# export NCCL_IB_PCI_RELAXED_ORDERING=1  # 启用PCI放松排序，可能提高性能

export NCCL_SOCKET_IFNAME=eth1  # 指定使用的网络接口
export NCCL_NET_GDR_LEVEL=2  # GPU Direct RDMA级别设置


# 增加额外的NCCL配置以提高稳定性
export NCCL_TIMEOUT=1800  # 30分钟总超时
export NCCL_RETRY_CNT=20  # 全局重试次数
export NCCL_CHECK_POINTERS=1  # 检查指针有效性
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # 启用异步错误处理

export NCCL_ALGO=Ring  # 使用Ring算法进行集体通信
export OMP_NUM_THREADS=60  # OpenMP线程数
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # 按PCI总线ID顺序枚举GPU设备

export NCCL_DEBUG=INFO


HOST_NUM=2
INDEX=$1
CHIEF_IP=29.151.32.74
HOST_GPU_NUM=8
MAIN_PROCESS_PORT=25444


echo "compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_multinode_launcher: standard
  gradient_accumulation_steps: 1
  gradient_clipping: 1.0
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  zero_stage: 0
distributed_type: DEEPSPEED
fsdp_config: {}
machine_rank: $INDEX
main_process_ip: $CHIEF_IP
main_process_port: $MAIN_PROCESS_PORT
main_training_function: main
num_machines: $HOST_NUM
num_processes: $(($HOST_NUM*$HOST_GPU_NUM))
mixed_precision: bf16
use_cpu: false" > /tmp/default_config.yaml

cat /tmp/default_config.yaml

pwd



# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0


# Get current date and time in format YYYYMMDD_HHMM
CURRENT_DATETIME=$(date +"%Y%m%d_%H%M")

if [ "$INDEX" -eq 0 ]; then
  echo "主节点(rank 0)准备启动，等待从节点连接..."
  echo "请确保在5秒内在从节点上启动训练脚本"
  sleep 5
  echo "开始训练..."
else
  echo "从节点(rank $INDEX)准备连接到主节点 $CHIEF_IP:$MAIN_PROCESS_PORT"
fi

accelerate launch  --config_file /tmp/default_config.yaml  mv_kontext/training_code/train_lora_flux.py \
    --pretrained_model_name_or_path=$FLUX_KONTEXT \
    --meta_path="data/navi/metainfo/navi_v1.5_metainfo_reorg.json" \
    --data_dir="data/navi/navi_v1.5" \
    --min_recon_num=1 \
    --max_recon_num=8 \
    --output_dir="mv_kontext/output/20250716_1201_train_v1" \
    --mixed_precision="bf16" \
    --train_batch_size=8 \
    --dataloader_num_workers 4 \
    --rank=64 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing \
    --learning_rate=1e-4 \
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
    --checkpointing_steps 25 \
    --validation_steps 25 \
    --test_meta_path="data/navi/navi_test_data/test_metainfo_20250714_223602.json" \
    --test_data_dir="data/navi/navi_test_data" \
    --checkpoints_total_limit 15 \
    --resume_from_checkpoint "latest"

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