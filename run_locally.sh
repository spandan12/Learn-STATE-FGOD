CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES
ds_name='CUB'

for sd in 1; do 
    timestamp=$(date +---20%y-%m-%d-%T---)
    python3 train.py \
        --config-file configs/prompt/cub_al.yaml \
        SEED ${sd} \
        OUTPUT_DIR "logs_live_c/"${timestamp}"res_mod_"${fstr}"_FSTR_"${ds_name}"_kl_"${kl_val}"/seed"${sd}
done