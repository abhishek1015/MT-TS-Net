# MT-TS-Net

### Create conda environment & installing packages

```
conda create --name mt-ts-net python=3.8
conda activate mt-ts-net
./install-packages.sh
```

### Launch slurm job for training network

```
#!/bin/bash

patch_size=224
modelarch="\"resnet18\""
latent_dim=512
description="\"brca_multitask_experiment\""
stat_norm_scheme="pretrained"
reference_patch="/data/Jiang_Lab/Data/MT-TS-Net/code/reference_patch_224.pkl"
learning_rate=0.00005
num_patch=32
batch_size=12
recon_size=64
dropout=0.5
kl_coeff=0

sbatch --gres=gpu:p100:4 --time=10:00:00 train_multitask_distributed.job $batch_size $num_patch $patch_size $recon_size $warmup_k $modelarch $learning_rate $dropout $kl_coeff $latent_dim $stat_norm_scheme $reference_patch
```

### Save MT-TS-Net representation and associated annotations of random patches

```
python script_save_HE_representation.py <slurm-job-id> <checkpoint-number> <modelarch>
python script_save_HE_representation_with_brca_sseg.py <slurm-job-id> <checkpoint-number> <modelarch>
```


### Visualize the task losses

Open notebook on browser 
```
http://localhost:9999/notebooks/notebooks/post-training-analysis.ipynb
```

### Visualize density maps

```
http://localhost:9999/notebooks/notebooks/HE-encoder-visualization.ipynb
```


### External dependencies
```
1. https://github.com/mahmoodlab/CLAM
2. https://github.com/DataX-JieHao/Cox-PASNet
3. https://github.com/EIDOSlab/torchstain
```
