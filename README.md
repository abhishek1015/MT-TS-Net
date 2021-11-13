# MT-TS-Net




## Launch slurm job for training network

```
cd code
./launch_multitask_training_job.sh
```

or 

```
#!/bin/bash

patch_size=224
modelarch="\"resnet18\""
latent_dim=512
description="\"brca_multitask_experiment\""
stat_norm_scheme="pretrained"
reference_patch="reference_patch_224.pkl"
learning_rate=0.00001
num_patch=32
batch_size=8
recon_size=64
dropout=0.8
kl_coeff=0

sbatch --gres=gpu:p100:4 --time=10:00:00 train_multitask_distributed.job $batch_size $num_patch $patch_size $recon_size $warmup_k $modelarch $learning_rate $dropout $kl_coeff $latent_dim $stat_norm_scheme $reference_patch
```
