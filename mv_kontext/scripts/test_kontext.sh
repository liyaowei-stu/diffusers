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


export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=3
export NCCL_TOPO_FILE=/tmp/topo.txt

export CUDA_VISIBLE_DEVICES=0

python mv_kontext/test_kontext.py