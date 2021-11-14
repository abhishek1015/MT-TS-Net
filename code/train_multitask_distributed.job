#!/bin/bash

#SBATCH --job-name survival-modeling
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mail-user=abhishek1015@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=40
#SBATCH --ntasks-per-core=1

batch_size=$1
num_patch=$2
patch_size=$3
recon_size=$4
warmup_k=$5
modelarch=$6
learning_rate=$7
dropout=$8
kl_coeff=$9
latent_dim=${10}
stat_norm_scheme=${11}
reference_patch=${12}

source ~/.bash_profile
module load cuDNN/7.6.5/CUDA-10.2
module load jq
conda activate /data/Jiang_Lab/Data/MT-TS-Net-condaenv/mt-ts-net
conda info --envs
python --version

cd /data/Jiang_Lab/Data/MT-TS-Net/code/

echo $SLURM_JOB_ID
model_json="model_"$SLURM_JOB_ID".json"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL
export MASTER_ADDR=localhost
export MASTER_PORT=23456
export NCCL_SOCKET_IFNAME=^docker0,lo
export OMP_NUM_THREADS=4

# set modelarch
jq 'del(.modelarch)' model.json > tmp.$$.json && mv tmp.$$.json $model_json
jq '. + { "modelarch": '$modelarch'}' $model_json > tmp.$$.json && mv tmp.$$.json $model_json

python3 -m torch.distributed.launch --nproc_per_node=4 train_multitask_distributed.py \
  --dataset_csv "/data/Jiang_Lab/Data/MT-TS-Net/code/data/tcga-brca/casewise_linked_data.csv" \
  --seg_dir "/data/Jiang_Lab/Data/tcga-brca-segmentations/" \
  --output_dir "/data/Jiang_Lab/Data/MT-TS-Net-output/" \
  --checkpoint_dir "/data/Jiang_Lab/Data/MT-TS-Net-checkpoint/" \
  --save_model_freq 10 \
  --batch_size $batch_size \
  --patch_size $patch_size \
  --num_patch $num_patch \
  --num_workers 25 \
  --recon_size $recon_size \
  --dropout_rate $dropout \
  --stat_norm_scheme $stat_norm_scheme \
  --color_norm \
  --warmup_k $warmup_k \
  --model_args_file $model_json \
  --use_img_network \
  --multitask \
  --num_ge 93 \
  --kl_coeff $kl_coeff \
  --fetchge \
  --learning_rate $learning_rate \
  --optimizer "Adam" \
  --latent_dim $latent_dim \
  --distributed \
  --reference_patch $reference_patch \
  --norecon_loss
  #--continue_train \
  #--pretrained_dir "/data/dubeyak/checkpoint_dir/23869143/"
