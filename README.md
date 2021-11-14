# MT-TS-Net

### Create conda environment & installing packages

```
conda create --name mt-ts-net python=3.8
conda activate mt-ts-net
./install-packages.sh
```

### Biowulf configuration for slurm job

edit ~/.bashrc
```
alias pip='/data/Jiang_Lab/Data/MT-TS-Net-condaenv/mt-ts-net/bin/pip'
alias python='/data/Jiang_Lab/Data/MT-TS-Net-condaenv/mt-ts-net/bin/python'
conda activate /data/Jiang_Lab/Data/MT-TS-Net-condaenv/mt-ts-net
cd /data/Jiang_Lab/Data/MT-TS-Net/code/
```

edit ~/.bash_profile
```
PATH=/data/Jiang_Lab/Data/MT-TS-Net-condaenv/mt-ts-net/bin/:$PATH:$HOME/.local/bin:$HOME/bin
export PATH
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

### Start jupyter server
```
jupyter notebook --port 9999 --no-browser
```

### Visualize the task losses

Open notebook on browser 
```
http://localhost:9999/notebooks/notebooks/post-training-analysis.ipynb
```

### External dependencies
```
1. https://github.com/mahmoodlab/CLAM
2. https://github.com/DataX-JieHao/Cox-PASNet
3. https://github.com/EIDOSlab/torchstain
```
