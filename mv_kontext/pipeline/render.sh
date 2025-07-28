
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

# python ic_custom_data_prepare/batch_vggt_preds_to_rendering_1pass.py \
#     --meta_path data/navi/metainfo/test_navi_v1.5.json \
#     --data_dir data/navi/navi_test/white_bg/conf_40.0 \
#     --output_dir data/navi/navi_test/white_bg \
#     --max_points 200000 \
#     --conf_threshold 70.0 \
#     --conf_threshold_value 1.0 \
#     --radius 0.01 \
#     --points_per_pixel 100 \
#     --bin_size 0 \
#     --apply_mask \
#     --batch_size 2



# Loop through different confidence thresholds
for conf_threshold in 10.0; do
    echo "Processing with conf_threshold: $conf_threshold"
    
    output_dir="data/navi/navi_test_threshold_test/conf_${conf_threshold}_8_recon_num"
    
    python mv_kontext/pipeline/render.py \
        --meta_path data/navi/metainfo/navi_v1.5_metainfo_reorg.json \
        --data_dir data/navi/navi_v1.5 \
        --output_dir $output_dir \
        --max_points 200000 \
        --conf_threshold $conf_threshold \
        --conf_threshold_value 1.0 \
        --radius 0.01 \
        --points_per_pixel 100 \
        --bin_size 0 \
        --apply_mask \
        --batch_size 2
done


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