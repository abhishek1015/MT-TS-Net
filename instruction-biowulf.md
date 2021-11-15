## Instruction to use Biowulf 


### Connect to biowulf at a port 9999

```
ssh -L 9999:localhost:9999 dubeyak@biowulf.nih.gov
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

### Submit slurm job
```
./launch_multitask_training_job.sh
```

### Check available gpu nodes
```
freen | grep -E 'Partition|----|gpu'
```

### Start a interactive session with 1 GPU and start ssh port forwarding
```
sinteractive --gres=gpu:p100:1 --cpus-per-task=40
ssh -L 9999:localhost:9999 dubeyak@<machine-name> -N -v -v
```


### Start jupyter server
```
ssh <machine-name>
jupyter notebook --port 9999 --no-browser
```

### Open notebooks on browser
```
http://localhost:9999/notebooks/notebooks/post-training-analysis.ipynb
http://localhost:9999/notebooks/notebooks/HE-encoder-visualization.ipynb
``` 




