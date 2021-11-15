import torch
import random
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from semanticseg_data import SemanticSegData 
import torch.utils.data as data_utils
import torchvision.models as models
from components import ResNetDecoder, DecoderBlock
from custom_resnet_module  import customResnetModel
from basic_vae_module import *
from vae_inference import *
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler, Subset, TensorDataset, WeightedRandomSampler
from collections import OrderedDict
import re
import sys

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)
np.random.seed(0)

slurm_job_id = sys.argv[1]

if len(sys.argv) > 2:
    _checkpoint_number = sys.argv[2]
    modelarch = sys.argv[3]

print(slurm_job_id)

os.environ["SLURM_JOB_ID"] = slurm_job_id
project_str = 'brca'

use_amp=False
has_ctl=True
has_decoder=False

if modelarch=='vaeresnet':
    stat_norm_scheme='random'
elif modelarch=='resnet18':
    stat_norm_scheme='pretrained'

num_samples=151
ge_count=93
num_patch=400
batch_size=1
if modelarch=='vaeresnet':
    patch_size=64
elif modelarch=='resnet18':
    patch_size=224
num_workers=12
split_seed=10000
fea_size=2048
latent_dim=512
duration_quantiles = [0.1, 0.5, 0.9]
pretrained_dir='/data/Jiang_Lab/Data/MT-TS-Net-checkpoint/'+os.environ["SLURM_JOB_ID"]+'/'
print(pretrained_dir)
full_data = True
color_norm=True
num_img_channel=3
mpp=0.5
if modelarch=='vaeresnet':
    patch_size=64
elif modelarch=='resnet18':
    patch_size=224
color_norm=True
if modelarch=='vaeresnet':
    stat_norm_scheme="random"
elif modelarch=='resnet18':
    stat_norm_scheme="pretrained"
data = pd.read_csv('/data/Jiang_Lab/Data/tcga-brca-semantic-seg/semantic_seg_data.csv', delimiter=',')
if modelarch=='vaeresnet':
    reference_filename = '/data/Jiang_Lab/Data/MT-TS-Net/code/reference_patch_64.pkl'
elif modelarch=='resnet18':
    reference_filename = "/data/Jiang_Lab/Data/MT-TS-Net/code/reference_patch_224.pkl"
dataset = SemanticSegData(data, mpp, patch_size, num_patch, num_workers, color_norm, stat_norm_scheme, reference_filename)
ds = torch.utils.data.DataLoader(dataset,batch_size=1, num_workers=num_workers, shuffle=False)

if len(sys.argv)>2:
    pretrained_model_name = _checkpoint_number + '_net_' + modelarch + '.pth'
else:
    pretrained_model_name = str(max([int(x.split('_')[0]) for x in os.listdir(pretrained_dir) if x.split('_')[0]!='kaplan' and x != os.environ["SLURM_JOB_ID"]+'_full.npy'])) + '_net_' + modelarch + '.pth'

pretrained_model_name = pretrained_dir + pretrained_model_name;
print(pretrained_model_name)

if modelarch=='vaeresnet':
  model = customVAE(latent_dim=latent_dim, enc_out_dim=fea_size, enc_type='resnet50',first_conv=False,maxpool1=False, input_channels=3)
  #model = customVAE.load_from_checkpoint(checkpoint_path='/data/Jiang_Lab/Data/VAE-tcga-brca-model/normal_color.ckpt')
  model.fc = torch.nn.Linear(in_features=latent_dim, out_features=latent_dim, bias=True)
  model.fc_mu = nn.Linear(in_features=fea_size, out_features=latent_dim)
  model.fc_var = nn.Linear(in_features=fea_size, out_features=latent_dim)
elif modelarch == 'resnet18':
  model = customResnetModel()
model.logvar_prog = torch.nn.Parameter(torch.zeros(1))
model.logvar_morph = torch.nn.Parameter(torch.zeros(1))
model.logvar_stage = torch.nn.Parameter(torch.zeros(1))
model.logvar_tcga_subtype = torch.nn.Parameter(torch.zeros(1))
model.logvar_immune_subtype = torch.nn.Parameter(torch.zeros(1))
model.logvar_mdsc = torch.nn.Parameter(torch.zeros(1))
model.logvar_caf = torch.nn.Parameter(torch.zeros(1))
model.logvar_m2 = torch.nn.Parameter(torch.zeros(1))
model.logvar_dysfunction = torch.nn.Parameter(torch.zeros(1))
model.logvar_exclusion = torch.nn.Parameter(torch.zeros(1))
if has_ctl:
  model.logvar_ctl = torch.nn.Parameter(torch.zeros(1))
#if has_decoder:
model.logvar_patch_recon = torch.nn.Parameter(torch.zeros(1))
model.logvar_ge = torch.nn.Parameter(torch.zeros(ge_count))
model.dropout = torch.nn.Dropout(0.8)
fc_config = {'prog': 1, 'stage': 1, 'morph': 3, 'tcgasubtype': 5, 'immunesubtype': 6, 'mdsc': 1, 'caf': 1, 'm2': 1, 'exclusion': 1, 'dysfunction': 1, 'ctl': 1, 'ge': ge_count}
num_out = {'COXPH': 1,'multitask': fc_config}
loss='multitask'
model.fc_prog = torch.nn.Linear(in_features=latent_dim, out_features=num_out[loss]['prog'], bias=True);
model.fc_stage = torch.nn.Linear(in_features=latent_dim, out_features=num_out[loss]['stage'], bias=True);
model.fc_morph = torch.nn.Linear(in_features=latent_dim, out_features=num_out[loss]['morph'], bias=True);
model.fc_tcgasubtype = torch.nn.Linear(in_features=latent_dim, out_features=num_out[loss]['tcgasubtype'], bias=True);
model.fc_immunesubtype = torch.nn.Linear(in_features=latent_dim, out_features=num_out[loss]['immunesubtype'], bias=True);
model.fc_MDSC = torch.nn.Linear(in_features=latent_dim, out_features=num_out[loss]['mdsc'], bias=True);
model.fc_CAF = torch.nn.Linear(in_features=latent_dim, out_features=num_out[loss]['caf'], bias=True);
model.fc_M2 = torch.nn.Linear(in_features=latent_dim, out_features=num_out[loss]['m2'], bias=True);
model.fc_Exclusion = torch.nn.Linear(in_features=latent_dim, out_features=num_out[loss]['exclusion'], bias=True);
model.fc_Dysfunction = torch.nn.Linear(in_features=latent_dim, out_features=num_out[loss]['dysfunction'], bias=True);
if has_ctl:
  model.fc_ctl = torch.nn.Linear(in_features=latent_dim, out_features=num_out[loss]['ctl'], bias=True);
model.fc_ge = torch.nn.Linear(in_features=latent_dim, out_features=num_out[loss]['ge'], bias=True);
L=latent_dim
D=latent_dim
K=1
model.attention_V = torch.nn.Sequential(torch.nn.Linear(L, D), torch.nn.Tanh())
model.attention_U = torch.nn.Sequential(torch.nn.Linear(L, D), torch.nn.Sigmoid())
model.attention_weights = torch.nn.Linear(D, K)
_checkpoint = torch.load(pretrained_model_name)

#if modelarch == 'vaeresnet':
_model = _checkpoint["model"]
new_state_dict = OrderedDict()
pattern = re.compile('module.')
for k,v in _model.items():
    if re.search("module", k):
        new_state_dict[re.sub(pattern, '', k)] = v
    else:
        new_state_dict = _model
#else:
#    new_state_dict=_checkpoint
model.load_state_dict(new_state_dict);

#if  modelarch == 'vaeresnet':
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
scaler.load_state_dict(_checkpoint["scaler"])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=model.to(device)

#feas = np.zeros([num_samples*num_patch, fea_size])
feas = np.zeros([num_samples*num_patch, latent_dim])
semantic_segs = np.zeros([num_samples*num_patch, 1])

ds_iter = iter(ds)
model.eval()
#with torch.cuda.amp.autocast(enabled=use_amp), torch.no_grad():
with torch.no_grad():
    for count in range(num_samples):
        data_dict = next(ds_iter)
        image = data_dict['image']
        semantic_seg = data_dict['semantic_seg']
        num_images = image.shape[0]
        #pdb.set_trace()
        image = torch.reshape(image, 
            (num_images*num_patch, num_img_channel, patch_size, patch_size))
        image = image.to(device)
        if modelarch == 'vaeresnet':
            img_fea = model.encoder(image)
            mu = model.fc_mu(img_fea)
            log_var = model.fc_var(img_fea)
            std = torch.exp(log_var / 2)
            q = torch.distributions.Normal(mu, std)
            z = q.rsample()
            img_fea = mu
        elif modelarch == 'resnet18':
            img_fea = model._run_step(image)
        img_fea = torch.reshape(img_fea, (num_images*num_patch, -1))
        for p in range(num_patch):
            i = count*num_patch+p
            feas[i,:] = img_fea.cpu().numpy()[p,:]
            semantic_segs[i,:] = semantic_seg.cpu().numpy()[0,p]
print(feas.shape)
print(semantic_segs.shape)

file_suffix='.npy'

store_to_file = pretrained_dir + '/' + os.environ["SLURM_JOB_ID"] + "_sseg" + file_suffix

with open(store_to_file, 'wb') as f:
    np.save(f, feas)
    np.save(f, semantic_segs)
