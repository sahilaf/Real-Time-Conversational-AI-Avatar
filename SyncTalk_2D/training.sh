#! /bin/bash
# input: bash training_328.sh file_name cuda_id
file_name=$1
cuda_id=$2
asr="ave"
file_path="./dataset/$file_name/$file_name.mp4"
data_dir="./dataset/$file_name"

CUDA_VISIBLE_DEVICES=$cuda_id python data_utils/process.py $file_path
CUDA_VISIBLE_DEVICES=$cuda_id python syncnet.py --save_dir ./syncnet_ckpt/$file_name --dataset_dir $data_dir --asr $asr
syncnet_checkpoint_dir=$(ls -v ./syncnet_ckpt/$file_name/*.pth | tail -n 1)
CUDA_VISIBLE_DEVICES=$cuda_id python train.py --dataset_dir $data_dir --save_dir ./checkpoint/$file_name --asr $asr --use_syncnet --syncnet_checkpoint $syncnet_checkpoint_dir
