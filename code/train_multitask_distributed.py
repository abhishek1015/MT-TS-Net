from __future__ import print_function
import os
import pdb
import json
import random
import torch
import pdb, traceback, sys
import logging
import argparse
import numpy as np
from tqdm import tqdm
from random import randint
from util.meters import CSVMeter
from sklearn.metrics import f1_score
from pathlib import Path
from torchvision.utils import make_grid, save_image
from torchvision import datasets, transforms, utils
from torchsummary import summary
import torchvision.models as models
from svsdata import SVSData
from util.visualizer import Visualizer
from collections import OrderedDict
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler, Subset, TensorDataset, WeightedRandomSampler
from util.loss import CoxPHLoss
from util import util
from lifelines.utils import concordance_index
from model import ConvNet
from model import MnistResNet
import torch.utils.data as data_utils
import adabound
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
import matplotlib
import matplotlib.pyplot as plt
from vae_inference import *
from sklearn.model_selection import train_test_split
import pandas as pd
#from components_grayscale import ResNetDecoder, DecoderBlock, DecoderBottlenec
#from basic_vae_module_grayscale import *
from basic_vae_module import *
from components import ResNetDecoder, DecoderBlock, DecoderBottleneck
from custom_resnet_module  import customResnetModel
from collections import OrderedDict
import re
#from torchcontrib.optim import SWA
#from pl_bolts.models.autoencoders import VAE


logging.basicConfig(filename='train' + os.environ['SLURM_JOB_ID'] + '.log', level=logging.INFO)
logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class Trainer:
    """Class responsible for training a model and outputing test metrics."""
    def __init__(
            self,
            args
            ):

        for k in args.keys():
            setattr(self, k, args[k])

        model_args = self.load_model_args(self.model_args_file)

        for k in model_args.keys():
            setattr(self, k, model_args[k])
        
        self.set_loss()
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.load_model()
        
        self.model.to(self.device)

        if self.distributed==True and torch.cuda.device_count() > 1:
            #https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
            logger.info("Let's use " + str(torch.cuda.device_count()) + " GPUs!")
            self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                     device_ids=[self.local_rank],
                     output_device=self.local_rank) #, find_unused_parameters = True)

        self.set_optim()

        if self.local_rank==0:
            os.makedirs(self.output_dir, exist_ok=True)
            self.epoch_meter = CSVMeter(os.path.join(self.output_dir, 'epoch_metrics_'+ os.environ['SLURM_JOB_ID'] + '.csv'), buffering=1)
            self.iter_meter = CSVMeter(os.path.join(self.output_dir, 'iter_metrics_'+ os.environ['SLURM_JOB_ID'] + '.csv'))
        
        self.load_datasets()

        logger.info(summary(self.model))
        logger.info('Optimizer: ' + self.optimizer)
        logger.info('svs file list: ' + self.dataset_csv)

    def load_datasets(self):
        csvdata = pd.read_csv(self.dataset_csv, delimiter=',')
        if len(self.duration_quantiles)>0:
            self._q = [0] * len(self.duration_quantiles)
            for idx_qs in range(len(self.duration_quantiles)):
                self._q[idx_qs] = csvdata[csvdata['event']>0]['duration'].quantile(q=self.duration_quantiles[idx_qs])
        targets = csvdata['event']
        train_idx, remaining_idx= train_test_split(np.arange(len(targets)), test_size=0.3, shuffle=True, stratify=targets, random_state=self.split_seed)
        trainset = csvdata.iloc[train_idx].copy().reset_index() 
        trainset = trainset.drop(['index'], axis=1)
        self.svs_dataset_train = SVSData(trainset, self.seg_dir, self.mpp, self.patch_size, self.num_patch, self.num_workers, self.color_norm, self.stat_norm_scheme, self._q, self.use_img_network, self.fetchge, False, self.reference_patch)
        
        remainingset = csvdata.iloc[remaining_idx].copy().reset_index()
        remainingset = remainingset.drop(['index'], axis=1)
        targets = remainingset['event']
        val_idx, test_idx= train_test_split(np.arange(len(targets)), test_size=0.5, shuffle=True, stratify=targets, random_state=self.split_seed)
        valset = remainingset.iloc[val_idx].copy().reset_index()
        valset = valset.drop(['index'], axis=1)
        testset = remainingset.iloc[test_idx].copy().reset_index()
        testset = testset.drop(['index'], axis=1)
        
        nval = int(len(valset))
        ntest = int(len(testset))
  
        self.svs_dataset_val = SVSData(valset, self.seg_dir, self.mpp, self.patch_size, self.test_num_patch, round(self.num_workers/2), self.color_norm, self.stat_norm_scheme, self._q, self.use_img_network, self.fetchge, True, self.reference_patch)
        self.svs_dataset_test = SVSData(testset, self.seg_dir, self.mpp, self.patch_size, self.test_num_patch, round(self.num_workers/2), self.color_norm, self.stat_norm_scheme, self._q, self.use_img_network,self.fetchge, True, self.reference_patch)
        sample_weights=(trainset['event']==1)*(1.0/(trainset['event']==1).sum()) +  (trainset['event']==0)*(3.0/(trainset['event']==0).sum()) 
        self.trainds = DataLoader(
           self.svs_dataset_train,
           batch_size=self.batch_size,
           shuffle=not self.distributed,
           #shuffle=False,
           num_workers=self.num_workers,
           pin_memory=True,
           sampler=DistributedSampler(trainset, shuffle=False, num_replicas=self.world_size, rank=self.local_rank, drop_last=False) if self.distributed else WeightedRandomSampler(sample_weights.values,len(self.svs_dataset_train),replacement=True),
        ) 
        self.valds = DataLoader( 
           self.svs_dataset_val,
           batch_size=self.batch_size,
           shuffle=not self.distributed,
           num_workers=self.num_workers,
           pin_memory=True,
           #sampler=DistributedSampler(valset) if self.distributed else None,
           )
        self.testds = DataLoader( 
           self.svs_dataset_test,
           batch_size=self.batch_size,
           shuffle=not self.distributed,
           #shuffle=False,
           num_workers=self.num_workers,
           pin_memory=True,
           #sampler=DistributedSampler(testset) if self.distributed else None,
           )
        logger.info('Training samples: %d' % len(self.trainds)*self.batch_size)
        logger.info('Validation samples: %d' % len(self.valds)*self.batch_size)
        logger.info('Test samples : %d' % len(self.testds)*self.batch_size)


    def load_feature_network(self):
       if self.modelarch=='resnet18':
           self.model = customResnetModel()
       elif self.modelarch=='resnet34':
           self.model = models.resnet34(pretrained=True)
           self.model.fc = torch.nn.Linear(in_features=self.num_img_fea[self.modelarch], out_features=self.num_img_fea[self.modelarch], bias=True)
           #self.model.decoder = ResNetDecoder(DecoderBlock, [3, 4, 6, 3], self.num_img_fea[self.modelarch], self.recon_size, False, False) 
           #self.model.fc_mu = nn.Linear(in_features=self.num_img_fea[self.modelarch], out_features=int(self.num_img_fea[self.modelarch]))
           #self.model.fc_var = nn.Linear(in_features=self.num_img_fea[self.modelarch], out_features=int(self.num_img_fea[self.modelarch]))
       elif self.modelarch=='inception_v3':
           self.model = models.inception_v3(pretrained=True)
           self.model.fc = torch.nn.Linear(in_features=self.num_img_fea[self.modelarch], out_features=self.num_img_fea[self.modelarch], bias=True)
           self.model.decoder = ResNetDecoder(DecoderBottleneck, [3, 4, 6, 3], self.num_img_fea[self.modelarch], self.recon_size, False, False) 
           self.model.fc_mu = nn.Linear(in_features=self.num_img_fea[self.modelarch], out_features=int(self.num_img_fea[self.modelarch]))
           self.model.fc_var = nn.Linear(in_features=self.num_img_fea[self.modelarch], out_features=int(self.num_img_fea[self.modelarch]))
       elif self.modelarch == 'vaeresnet':
           self.model = customVAE(latent_dim=self.latent_dim, enc_out_dim=2048, enc_type='resnet50',first_conv=False,maxpool1=False, input_channels=self.num_img_channel)
           if self.latent_dim == 64:
               self.model = customVAE.load_from_checkpoint(checkpoint_path='/data/Jiang_Lab/Data/VAE-tcga-brca-model/small_color.ckpt')
           elif self.latent_dim == 2048:
               self.model = customVAE.load_from_checkpoint(checkpoint_path='/data/Jiang_Lab/Data/VAE-tcga-brca-model/normal_color.ckpt')
           self.model.fc = torch.nn.Linear(in_features=self.num_img_fea[self.modelarch], out_features=self.num_img_fea[self.modelarch], bias=True)
       else:
           raise ValueError('Unrecognised model')

    def load_multitask_weights(self):
        self.model.logvar_prog = torch.nn.Parameter(torch.zeros(1))
        self.model.logvar_morph = torch.nn.Parameter(torch.zeros(1))
        self.model.logvar_stage = torch.nn.Parameter(torch.zeros(1))
        self.model.logvar_tcga_subtype = torch.nn.Parameter(torch.zeros(1))
        self.model.logvar_immune_subtype = torch.nn.Parameter(torch.zeros(1))
        self.model.logvar_mdsc = torch.nn.Parameter(torch.zeros(1))
        self.model.logvar_caf = torch.nn.Parameter(torch.zeros(1))
        self.model.logvar_m2 = torch.nn.Parameter(torch.zeros(1))
        self.model.logvar_dysfunction = torch.nn.Parameter(torch.zeros(1))
        self.model.logvar_exclusion = torch.nn.Parameter(torch.zeros(1))
        self.model.logvar_ctl = torch.nn.Parameter(torch.zeros(1))
        self.model.logvar_ge = torch.nn.Parameter(torch.zeros(self.num_out['multitask']['ge'])) 
        self.model.logvar_patch_recon = torch.nn.Parameter(torch.zeros(1))
 
    def load_otherfea_map(self):
        if 'self.model' not in globals():
            self.model = torch.nn.Identity(1)  
        self.num_nonimg_fea = 4 + self.num_out['multitask']['morph'] + self.num_out['multitask']['tcgasubtype'] + self.num_out['multitask']['immunesubtype'] + 5 + self.num_out['multitask']['ge']
        ofea_fc_modules = [] 
        ofea_fc_modules.append(torch.nn.Sequential(
            torch.nn.Linear(in_features=self.num_nonimg_fea, out_features=self.nonimg_fc_width, bias=True),
            torch.nn.ReLU()))
        self.model.ofea_network = torch.nn.Sequential(*ofea_fc_modules)

    def load_final_layer(self):
        self.model.dropout = torch.nn.Dropout(self.dropout_rate)
        if self.use_img_network and self.use_nonimg_network:
            self.model.fc_ll = torch.nn.Linear(in_features=self.num_img_fea[self.modelarch]+self.nonimg_fc_width, out_features=self.num_out[self.loss], bias=True);
        elif self.use_nonimg_network:
            self.model.fc_ll = torch.nn.Linear(in_features=self.nonimg_fc_width, out_features=self.num_out[self.loss], bias=True);
        elif self.use_img_network:
            if self.multitask:               
                self.model.fc_prog = torch.nn.Linear(in_features=self.num_img_fea[self.modelarch], out_features=self.num_out[self.loss]['prog'], bias=True);
                self.model.fc_stage = torch.nn.Linear(in_features=self.num_img_fea[self.modelarch], out_features=self.num_out[self.loss]['stage'], bias=True);
                self.model.fc_morph = torch.nn.Linear(in_features=self.num_img_fea[self.modelarch], out_features=self.num_out[self.loss]['morph'], bias=True);
                self.model.fc_tcgasubtype = torch.nn.Linear(in_features=self.num_img_fea[self.modelarch], out_features=self.num_out[self.loss]['tcgasubtype'], bias=True);
                self.model.fc_immunesubtype = torch.nn.Linear(in_features=self.num_img_fea[self.modelarch], out_features=self.num_out[self.loss]['immunesubtype'], bias=True);
                self.model.fc_MDSC = torch.nn.Linear(in_features=self.num_img_fea[self.modelarch], out_features=self.num_out[self.loss]['mdsc'], bias=True);
                self.model.fc_CAF = torch.nn.Linear(in_features=self.num_img_fea[self.modelarch], out_features=self.num_out[self.loss]['caf'], bias=True);
                self.model.fc_M2 = torch.nn.Linear(in_features=self.num_img_fea[self.modelarch], out_features=self.num_out[self.loss]['m2'], bias=True);
                self.model.fc_Exclusion = torch.nn.Linear(in_features=self.num_img_fea[self.modelarch], out_features=self.num_out[self.loss]['exclusion'], bias=True);
                self.model.fc_Dysfunction = torch.nn.Linear(in_features=self.num_img_fea[self.modelarch], out_features=self.num_out[self.loss]['dysfunction'], bias=True);
                self.model.fc_ctl = torch.nn.Linear(in_features=self.num_img_fea[self.modelarch], out_features=self.num_out[self.loss]['ctl'], bias=True);
                self.model.fc_ge = torch.nn.Linear(in_features=self.num_img_fea[self.modelarch], out_features=self.num_out[self.loss]['ge'], bias=True);
            else:
                self.model.fc_ll = torch.nn.Linear(in_features=self.num_img_fea[self.modelarch], out_features=self.num_out[self.loss], bias=True);
        return

    def load_gated_attention_layer(self, L, D, K):
        self.model.attention_V = torch.nn.Sequential(torch.nn.Linear(L, D), torch.nn.Tanh())
        self.model.attention_U = torch.nn.Sequential(torch.nn.Linear(L, D), torch.nn.Sigmoid())
        self.model.attention_weights = torch.nn.Linear(D, K)
        return
    
    def load_plain_attention_layer(self, L, D, K):
        self.model.attention_V = torch.nn.Sequential(torch.nn.Linear(L, D), torch.nn.Tanh())
        self.model.attention_weights = torch.nn.Linear(D, K)
        return

    def load_model(self):
       self.num_img_fea = { 
                    'resnet18': 512,
                    'resnet34': 512,
                    'inception_v3': 2048, 
                    'vaeresnet': self.latent_dim,
                  } 

       if self.seg_dir.split('/')[4].split('-')[1] == 'brca':
           fc_config = {'prog': 1, 'stage': 1, 'morph': 3, 'tcgasubtype': 5, 'immunesubtype': 6, 'mdsc': 1, 'caf': 1, 'm2': 1, 'exclusion': 1, 'dysfunction': 1, 'ctl': 1, 'ge': self.num_ge}
       elif self.seg_dir.split('/')[4].split('-')[1] == 'luad':
           fc_config = {'prog': 1, 'stage': 1, 'morph': 12, 'tcgasubtype': 7, 'immunesubtype': 6, 'mdsc': 1, 'caf': 1, 'm2': 1, 'exclusion': 1, 'dysfunction': 1, 'ctl': 1, 'ge': self.num_ge}
       elif self.seg_dir.split('/')[4].split('-')[1] == 'gbm':
           fc_config = {'prog': 1, 'stage': 1, 'morph': 1, 'tcgasubtype': 1, 'immunesubtype': 6, 'mdsc': 1, 'caf': 1, 'm2': 1, 'exclusion': 1, 'dysfunction': 1, 'ctl': 1, 'ge': self.num_ge}
       
       self.num_out = { 
                   'COXPH': 1,
                   'multitask': fc_config
                 }

       if self.use_img_network:
           self.load_feature_network()
       
           if self.MIL_pool == 'gated_attention':
               L = self.num_img_fea[self.modelarch]
               D = self.num_img_fea[self.modelarch]
               K = 1
               self.load_gated_attention_layer(L, D, K)
       
           if self.MIL_pool == 'plain_attention':
               L = self.num_img_fea[self.modelarch]
               D = self.num_img_fea[self.modelarch]
               K = 1
               self.load_plain_attention_layer(L, D, K)

       if self.use_nonimg_network:
           self.load_otherfea_map()

       self.load_final_layer()

       if self.multitask:
           self.load_multitask_weights()

       self.modelname = self.modelarch
       logger.info("Successfully loaded model: " + self.modelarch)
 
       if self.continue_train==True:
           pretrained_model_name = str(max([int(x.split('_')[0]) for x in os.listdir(self.pretrained_dir) if x.split('_')[0]!='kaplan' and x.endswith('.pth')])) + '_net_' + self.modelname + '.pth'
           pretrained_model_name = self.pretrained_dir + pretrained_model_name;
           if self.distributed:
               _checkpoint = torch.load(pretrained_model_name)
               _model = _checkpoint["model"]
               new_state_dict = OrderedDict()
               pattern = re.compile('module.')
               for k,v in _model.items():
                   if re.search("module", k):
                       new_state_dict[re.sub(pattern, '', k)] = v
                   else:
                       new_state_dict = _model
               self.model.load_state_dict(new_state_dict);
               self.scaler.load_state_dict(_checkpoint["scaler"])
           else:
               _checkpoint = torch.load(pretrained_model_name)
               self.model.load_state_dict(_checkpoint["model"]);
               #self.scaler.load_state_dict(_checkpoint["scaler"])
           logger.info("Loaded pretrained model from " + pretrained_model_name)
       
    def load_model_args(self, argsfile):
        """Load model args from json."""
        model_args = json.load(open(argsfile, 'r'))
        print(model_args)
        return model_args

    def set_optim(self):  
        if self.optimizer == "Adam": 
            self.optim = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                eps=1e-7,  # match tf2.1 adam
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "SGD":
            self.optim = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate, 
                momentum=0.9
            )
        elif self.optimizer == "AdaBound":
            self.optim = adabound.AdaBound(
                self.model.parameters(), 
                lr=self.learning_rate, 
                final_lr=self.final_lr
            )

        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=0.95)
        #self.optimizer = torchcontrib.optim.SWA(self.optimizer, swa_start=10, swa_freq=5, swa_lr=0.05)

    def censoredCrossEntropyLoss(self, logits, labels, events):
        batch_size = logits.size()[0]            # batch_size
        ## observed data loglikelihood
        loglikelihood = F.log_softmax(logits, dim=1)   # compute the log of softmax values
        observed_loglikelihood = loglikelihood[events.nonzero(as_tuple=False).flatten()]
        observed_loglikelihood = observed_loglikelihood[range(observed_loglikelihood.shape[0]), labels[events.nonzero(as_tuple=False).flatten()]]
        observed_loglikelihood = observed_loglikelihood.mean()
        ## censored data loglikelihood
        likelihood = F.softmax(logits, dim=1)
        censored_likelihood = likelihood[(1-events).nonzero(as_tuple=False).flatten()]
        likelihood_cumsum = likelihood.fliplr().cumsum(1).fliplr()
        censored_likelihood_cumsum = likelihood_cumsum[(1-events).nonzero(as_tuple=False).flatten()]
        censored_likelihood = censored_likelihood_cumsum[range(censored_likelihood_cumsum.shape[0]), labels[(1-events).nonzero(as_tuple=False).flatten()]] - censored_likelihood[range(censored_likelihood.shape[0]), labels[(1-events).nonzero(as_tuple=False).flatten()]]  + 1e-8
        censored_loglikelihood = torch.log(censored_likelihood)
        censored_loglikelihood = censored_loglikelihood.mean()
        if censored_loglikelihood.isnan().item() and observed_loglikelihood.isnan().item():
            return torch.zeros(1).to(self.device)
        elif censored_loglikelihood.isnan().item():
            return -1*observed_loglikelihood
        elif observed_loglikelihood.isnan().item():
            return -1*censored_loglikelihood 
        else:
           agg_loglikelihood = -1*(censored_loglikelihood+observed_loglikelihood)/2.0
           return agg_loglikelihood

    def set_loss(self):
        if self.multitask:
            self.loss='multitask'
            self.criterion_ph = CoxPHLoss()
            self.criterion_mse = torch.nn.MSELoss()
            self.criterion_ce = torch.nn.CrossEntropyLoss()
        else: 
            self.loss='COXPH'
            self.criterion = CoxPHLoss()
        logger.info("Successfully initialize loss: " + self.loss)

    def train(self):
        """Train model"""
        self._epbar = range(self.num_epochs)
        if self.progress:
            self._epbar = tqdm(self._epbar, desc='epoch')


        for self._epoch in self._epbar:
            """set model to training mode"""
            self.model.train()

            """training"""
            eploss, epmetrics = self.epoch()
            
            if self.use_scheduler:
                self.scheduler.step()

            if self.local_rank == 0:

                """set model to evaluation mode"""
                self.model.eval()

                with torch.no_grad(): 
                    valmetrics = self.validate()

                if self.progress:
                    self._epbar.set_postfix(loss=eploss, **valmetrics)
                else:
                    print(f"secs {self.epoch_meter.elapsed_seconds()} seed {self.split_seed} fold {self.fold} epoch {self._epoch}")

                self.epoch_meter.update(**epmetrics, **valmetrics)
                self.epoch_meter.flush()

            if self.local_rank==0 and self._epoch % self.save_model_freq == 0:
                self.save_checkpoint(self._epoch)


    def epoch(self):
        """Train model on training dataset"""
        self._itbar = self.trainds
        if self.progress:
            self._itbar = tqdm(self._itbar, desc='iter')
        eploss = 0
        epmetrics={}
        ep_multitask_losses={}
        for self._iter, data_dict in enumerate(self._itbar):
            
            setattr(self, 'image', data_dict['image'])
            
            for k in self.linked_features:
                setattr(self, k, data_dict[k])

            itloss, it_multitask_losses = self.iteration()
            
            if self.progress:
                self._itbar.set_postfix(loss=itloss)
            
            if self.local_rank==0:
                self.iter_meter.update(loss=itloss)
                self.iter_meter.flush()
            
            eploss += itloss / len(self.trainds)
            
            if len(ep_multitask_losses.keys())==0:
                for k in it_multitask_losses.keys():
                    ep_multitask_losses[k]=it_multitask_losses[k] / len(self.trainds)
            else:
                for k in it_multitask_losses.keys():
                    ep_multitask_losses[k] += it_multitask_losses[k] / len(self.trainds)
        # track loss
        epmetrics['train_loss'] = eploss
        for k in it_multitask_losses.keys():
            epmetrics[ 'train_' + k] = ep_multitask_losses[k]
        return eploss, epmetrics

    def feature_aggregation(self, logits):
        # logit: N x P x L
        L = logits.shape[2]
        P = logits.shape[1]
        N = logits.shape[0]
        if self.MIL_pool == 'mean':
            agg_fea = torch.mean(logits, dim=1)
        elif self.MIL_pool == 'max':
            agg_fea,_ = torch.max(logits, dim=1)
        elif self.MIL_pool == 'logsumexp':  
            agg_fea = torch.logsumexp(logits, dim=1)
        elif self.MIL_pool == 'gated_attention':
            A_V = self.model.module.attention_V(logits) #A_V: N x P x D
            A_U = self.model.module.attention_U(logits) #A_U: N x P x D
            A = self.model.module.attention_weights(A_V * A_U) # A: N x P x 1
            A = F.softmax(A, dim=1)
            #A = torch.reshape(A, (N*P, 1))
            #A = F.softmax(A, dim=0)  # softmax over P -- A: N x P x 1
            #A = torch.reshape(A, (N, P, 1))
            A = A.repeat(1,1,L)
            agg_fea = torch.sum(A*logits, dim=1)
        elif self.MIL_pool == 'plain_attention':
            A_V = self.model.module.attention_V(logits) #A_V: N x P x D
            A = self.model.module.attention_weights(A_V) # A: N x P x 1
            A = F.softmax(A, dim=1)  # softmax over P -- A: N x P x 1
            A = A.repeat(1,1,L)
            agg_fea = torch.sum(A*logits, dim=1)
        return agg_fea

    def non_img_fea(self, num_samples):
        stage = self.ajcc_pathologic_stage.view(num_samples, 1)
        t = self.ajcc_pathologic_t.view(num_samples, 1)
        n = self.ajcc_pathologic_n.view(num_samples, 1)
        age = self.age_at_index.view(num_samples, 1)
        one_hot_morph = torch.zeros(num_samples, self.num_out['multitask']['morph'])
        one_hot_morph[range(num_samples), self.morphology]=1
        one_hot_tcga_subtype = torch.zeros(num_samples, self.num_out['multitask']['tcgasubtype'])
        one_hot_tcga_subtype[range(num_samples), self.tcga_subtype]=1
        one_hot_immune_subtype = torch.zeros(num_samples, self.num_out['multitask']['immunesubtype'])
        one_hot_immune_subtype[range(num_samples), self.immune_subtype]=1
        MDSC = self.MDSC.view(num_samples, 1) 
        CAF = self.CAF.view(num_samples, 1)
        M2 = self.M2.view(num_samples, 1)
        Exclusion = self.Exclusion.view(num_samples, 1)
        Dysfunction = self.Dysfunction.view(num_samples, 1) 
        ctl = self.ctl.view(num_samples, 1) 
        ge = self.ge.view(num_samples, self.num_out['multitask']['ge']) 
        stage =stage.to(self.device)
        t = t.to(self.device)
        n = n.to(self.device)
        age=age.to(self.device)
        one_hot_morph = one_hot_morph.to(self.device)
        one_hot_tcga_subtype = one_hot_tcga_subtype.to(self.device)
        one_hot_immune_subtype = one_hot_immune_subtype.to(self.device)
        MDSC = MDSC.to(self.device)
        CAF = CAF.to(self.device)
        M2 = M2.to(self.device)
        Exclusion = Exclusion.to(self.device)
        Dysfunction = Dysfunction.to(self.device)
        ctl = ctl.to(self.device)
        if self.fetchge:
            ge = ge.to(self.device)
        fea = torch.cat(( stage.float(), t.float(), n.float(), age.float(), one_hot_morph.float(), one_hot_tcga_subtype.float(), one_hot_immune_subtype.float(), MDSC.float(), CAF.float(), M2.float(), Exclusion.float(), Dysfunction.float(), ge.float()), dim=1)
        return fea

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return p, q, z

    def prediction(self, num_samples, split):

        l={}
        us_l={}
        num_patch = self.test_num_patch if (split=='test' or split=='val') else self.num_patch 
        
        with torch.cuda.amp.autocast(enabled=self.use_amp):    
            if self.use_img_network:
                if self.modelarch == 'vaeresnet':
                    img_fea = self.model.module.encoder(self.image)
                    mu = self.model.module.fc_mu(img_fea)
                    log_var = self.model.module.fc_var(img_fea)
                    p, q, z = self.sample(mu, log_var)
                    img_fea = mu
                    if not self.norecon_loss: 
                        recon_img = self.model.module.decoder(z)
                else:
                    #print(self.model)
                    img_fea = self.model.module._run_step(self.image)         
                    if split=='train' and self.modelarch=='inception_v3':
                        img_fea=img_fea[0]
                if self._epoch > self.warmup_k:
                    img_fea = torch.reshape(img_fea, (num_samples, num_patch, -1))
                    agg_img_fea = self.feature_aggregation(img_fea)
                    concat_fea = agg_img_fea

            if self.use_nonimg_network:
                non_img_fea = self.non_img_fea(num_samples)    
                concat_fea = non_img_fea

            if self.use_img_network and self.use_nonimg_network: 
                concat_fea =  torch.cat((non_img_fea, agg_img_fea), dim=1)
        

            if self._epoch > self.warmup_k:
            
                concat_fea = self.model.module.dropout(concat_fea)
            
                #prognosis
                avgpred_prog = self.model.module.fc_prog(concat_fea)
                us_l['us_prognosis_loss'] = self.criterion_ph(avgpred_prog, self.duration, self.event)
                l['prognosis_loss'] = 2.0*torch.exp(-self.model.module.logvar_prog)*us_l['us_prognosis_loss'] + self.model.module.logvar_prog
            
                #staging
                avgpred_stage = self.model.module.fc_stage(concat_fea)
                us_l['us_staging_loss'] = self.criterion_mse(avgpred_stage, self.ajcc_pathologic_stage.float()) 
                l['staging_loss'] = torch.exp(-self.model.module.logvar_stage)*us_l['us_staging_loss'] + self.model.module.logvar_stage

                #morphology
                avgpred_morph = self.model.module.fc_morph(concat_fea)
                us_l['us_morphology_loss'] = self.criterion_ce(avgpred_morph, self.morphology) 
                l['morphology_loss'] = 2.0*torch.exp(-self.model.module.logvar_morph)*us_l['us_morphology_loss'] + self.model.module.logvar_morph

                #molecular subtype
                avgpred_tcgasubtype = self.model.module.fc_tcgasubtype(concat_fea)
                us_l['us_molecular_subtyping_loss'] = self.criterion_ce(avgpred_tcgasubtype, self.tcga_subtype)
                l['molecular_subtyping_loss'] =  2.0*torch.exp(-self.model.module.logvar_tcga_subtype)*us_l['us_molecular_subtyping_loss'] + self.model.module.logvar_tcga_subtype

                #immunestubtype
                avgpred_immunesubtype = self.model.module.fc_immunesubtype(concat_fea)
                us_l['us_immune_subtyping_loss'] = self.criterion_ce(avgpred_immunesubtype, self.immune_subtype)
                l['immune_subtyping_loss'] = 2.0*torch.exp(-self.model.module.logvar_immune_subtype)*us_l['us_immune_subtyping_loss'] + self.model.module.logvar_immune_subtype

                #tide scores
                avgpred_MDSC = self.model.module.fc_MDSC(concat_fea)
                us_l['us_MDSC_loss'] = self.criterion_mse(avgpred_MDSC, self.MDSC.float())
                l['MDSC_loss'] = torch.exp(-self.model.module.logvar_mdsc)*us_l['us_MDSC_loss'] + self.model.module.logvar_mdsc
                avgpred_CAF = self.model.module.fc_CAF(concat_fea)
                us_l['us_CAF_loss'] = self.criterion_mse(avgpred_CAF, self.CAF.float())
                l['CAF_loss'] = torch.exp(-self.model.module.logvar_caf)*us_l['us_CAF_loss'] + self.model.module.logvar_caf
                avgpred_M2 = self.model.module.fc_M2(concat_fea)
                us_l['us_M2_loss'] = self.criterion_mse(avgpred_M2, self.M2.float())
                l['M2_loss'] = torch.exp(-self.model.module.logvar_m2)*us_l['us_M2_loss'] + self.model.module.logvar_m2
                avgpred_Exclusion = self.model.module.fc_Exclusion(concat_fea)
                us_l['us_Exclusion_loss'] = self.criterion_mse(avgpred_Exclusion, self.Exclusion.float())
                l['Exclusion_loss'] = torch.exp(-self.model.module.logvar_exclusion)*us_l['us_Exclusion_loss'] + self.model.module.logvar_exclusion
                avgpred_Dysfunction = self.model.module.fc_Dysfunction(concat_fea)
                us_l['us_Dysfunction_loss'] = self.criterion_mse(avgpred_Dysfunction, self.Dysfunction.float())
                l['Dysfunction_loss'] =  torch.exp(-self.model.module.logvar_dysfunction)*us_l['us_Dysfunction_loss'] + self.model.module.logvar_dysfunction
                avgpred_ctl = self.model.module.fc_ctl(concat_fea)
                us_l['us_ctl_loss'] = self.criterion_mse(avgpred_ctl, self.ctl.float())
                l['ctl_loss'] =  torch.exp(-self.model.module.logvar_ctl)*us_l['us_ctl_loss'] + self.model.module.logvar_ctl

                # gene expression
                avgpred_ge = self.model.module.fc_ge(concat_fea)
                us_l['us_ge_loss'] = torch.mean((self.ge.float()-avgpred_ge) ** 2, 0)
                agg_ge_loss = torch.exp(-self.model.module.logvar_ge)*us_l['us_ge_loss']+self.model.module.logvar_ge
                us_l['us_ge_loss'] = us_l['us_ge_loss'].sum().unsqueeze(0)
                l['ge_loss'] = agg_ge_loss.sum().unsqueeze(0)

            #recon loss
            if not self.norecon_loss:
                img_ds =  F.interpolate(self.image, size=self.recon_size)
                recon_loss = F.mse_loss(recon_img,  img_ds, reduction='mean')
                us_l['us_recon_loss'] = recon_loss
                recon_loss = torch.exp(-self.model.module.logvar_patch_recon)*us_l['us_recon_loss']+self.model.module.logvar_patch_recon
                us_l['us_recon_loss'] = us_l['us_recon_loss'].unsqueeze(0)
                l['recon_loss'] = recon_loss.unsqueeze(0)
                
                # https://towardsdatascience.com/beginner-guide-to-variational-autoencoders-vae-with-pytorch-lightning-part-2-6b79ad697c79
                #kl = (-0.5 *(1 + log_var - mu.pow(2) - log_var.exp())).sum(dim=1).mean(dim=0)
                log_qz = q.log_prob(z)
                log_pz = p.log_prob(z)
                kl = log_qz - log_pz
                kl = kl.mean()
                us_l['us_kl_loss'] = kl.unsqueeze(0)
                l['kl_loss'] =  self.kl_coeff*kl.unsqueeze(0)

            if self._epoch > self.warmup_k:
                avgpred = avgpred_prog
            else:
                avgpred = torch.zeros(self.batch_size)
 
            loss = torch.zeros_like(l['staging_loss'])
            if self._epoch > self.warmup_k:
                for k in l.keys():
                    loss = loss + l[k]
            else:
                loss = us_l['us_recon_loss'] + l['kl_loss']

        return avgpred, loss, l, us_l

    def copy_to_gpu(self):
        self.image = self.image.to(self.device)
        self.duration = self.duration.to(self.device)
        self.duration_q = self.duration_q.to(self.device)
        self.event = self.event.to(self.device)
        self.ajcc_pathologic_stage = self.ajcc_pathologic_stage.to(self.device)
        self.ajcc_pathologic_t = self.ajcc_pathologic_t.to(self.device)
        self.ajcc_pathologic_n = self.ajcc_pathologic_n.to(self.device)
        self.age_at_index = self.age_at_index.to(self.device)
        self.morphology = self.morphology.to(self.device)
        self.tcga_subtype = self.tcga_subtype.to(self.device)
        self.immune_subtype = self.immune_subtype.to(self.device)
        self.MDSC = self.MDSC.to(self.device)
        self.CAF = self.CAF.to(self.device)
        self.M2 = self.M2.to(self.device)
        self.Exclusion = self.Exclusion.to(self.device)
        self.Dysfunction = self.Dysfunction.to(self.device)
        self.ctl = self.ctl.to(self.device)
        if self.fetchge:
            self.ge = self.ge.to(self.device)

    def iteration(self):
        """Train model on one iteration of mini-batch"""
        self.optim.zero_grad()
        num_samples=self.image.shape[0]

        if self.use_img_network:
            self.image = torch.reshape(self.image, (num_samples*self.num_patch, self.num_img_channel, self.patch_size, self.patch_size))
        
        self.copy_to_gpu()
        self.model.to(self.device)

        avgpred, loss, multitask_losses, us_multitask_losses = self.prediction(num_samples, 'train')

        itloss = loss.detach().cpu().item()
        it_multitask_losses = {}
        for k in multitask_losses.keys():
            it_multitask_losses[k]=multitask_losses[k].detach().cpu().item()
        for k in us_multitask_losses.keys():
            it_multitask_losses[k]=us_multitask_losses[k].detach().cpu().item()

        self.scaler.scale(loss).backward()
        #loss.backward()

        self.scaler.unscale_(self.optim)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.scaler.step(self.optim)
        self.scaler.update()
        #self.optim.step()
        return itloss, it_multitask_losses

    def plot_kaplan_meirer_curve(self, p, g, e, split):
      p_25=np.percentile(p, 25)
      p_75=np.percentile(p, 75)
      idx_low=p<p_25
      idx_high=p>p_75
      idx_medium=(~idx_low) & (~idx_high)
      idx_low = idx_low.flatten()
      idx_high = idx_high.flatten()
      idx_medium = idx_medium.flatten()
      plt.figure()
      if len(g[idx_high])>0:
        kmf_h = KaplanMeierFitter(label="High-risk group", alpha=1)
        kmf_h.fit(g[idx_high]/30, e[idx_high])
        kmf_h.plot()
      if  len(g[idx_medium])>0:
        kmf_m = KaplanMeierFitter(label="Medium-risk group", alpha=1)
        kmf_m.fit(g[idx_medium]/30, e[idx_medium])
        kmf_m.plot()
      if len(g[idx_low])>0:
        kmf_l = KaplanMeierFitter(label="Low-risk group", alpha=1)
        kmf_l.fit(g[idx_low]/30, e[idx_low])
        kmf_l.plot()
      plt.tight_layout()
      plt.xlabel('Months')
      plt.ylabel('Survival probability')
      checkpoint_dir = self.checkpoint_dir+ '/' + os.environ['SLURM_JOB_ID'] + '/';
      os.makedirs(checkpoint_dir, exist_ok=True)
      plt.savefig(checkpoint_dir + 'kaplan_meirer_curve_' + split + '_' + str(self._epoch) + '.png')
 
    def compute_logrank_pvalue(self, p, g, e):
      p_25=np.percentile(p, 25)
      p_75=np.percentile(p, 75)
      idx_low=p<p_25
      idx_high=p>p_75
      idx_medium=(~idx_low) & (~idx_high)
      idx_low = idx_low.flatten()
      idx_high = idx_high.flatten()
      idx_medium = idx_medium.flatten()
      test_result = logrank_test(g[idx_high], g[idx_low], event_observed_A=e[idx_high], event_observed_B=e[idx_low])
      return test_result.p_value

    def validate(self):
        """Compute loss and f1-score on validation and test datasets"""
        metrics = {}
        for split, loader in [('val', self.valds), ('test', self.testds)]:
            valbar = loader
            if self.progress:
                valbar = tqdm(loader, desc=split)
            valloss = 0
            val_multitask_losses={}
            Ypreds, Yactual, E = [], [], []
            for data_dict in valbar:
                
                self.image = data_dict['image']
                
                for k in self.linked_features:
                    setattr(self, k, data_dict[k])

                num_samples=self.image.shape[0]

                if self.use_img_network:
                    self.image = torch.reshape(self.image, (num_samples*self.test_num_patch, self.num_img_channel, self.patch_size, self.patch_size))

                self.copy_to_gpu()
                self.model.to(self.device)

                avgpred, loss, multitask_losses, us_multitask_losses = self.prediction(num_samples, split)

                itloss = loss.detach().cpu().item()

                it_multitask_losses = {}
                for k in multitask_losses.keys():
                    it_multitask_losses[k]=multitask_losses[k].detach().cpu().item()
                for k in us_multitask_losses.keys():
                    it_multitask_losses[k]=us_multitask_losses[k].detach().cpu().item()

                valloss +=itloss
                if len(val_multitask_losses.keys())==0:
                    for k in it_multitask_losses.keys():
                        val_multitask_losses[k]=it_multitask_losses[k]
                else:
                    for k in it_multitask_losses.keys():
                        val_multitask_losses[k]+=it_multitask_losses[k]

                Yactual.append(self.duration.cpu().numpy())
                E.append(self.event.cpu().numpy())
                Ypreds.append(avgpred.mul(-1).cpu().numpy())
            metrics[split + '_loss'] = valloss/len(valbar)
            for k in val_multitask_losses.keys():
                metrics[split + '_' + k] = val_multitask_losses[k]/len(valbar)

            if self._epoch > self.warmup_k: 
                try:
                    metrics[split + '_c_index'] = concordance_index(np.concatenate(Yactual), np.concatenate(Ypreds), np.concatenate(E))
                except:
                    metrics[split + '_c_index'] = float("NAN")
                try:
                    metrics[split + '_logrank_pvalue'] = self.compute_logrank_pvalue(np.concatenate(Ypreds)*-1, np.concatenate(Yactual), np.concatenate(E));
                except:
                    metrics[split + '_logrank_pvalue'] = float("NAN")
                self.plot_kaplan_meirer_curve(np.concatenate(Ypreds)*-1, np.concatenate(Yactual), np.concatenate(E), split)
        return metrics

    def save_checkpoint(self, epoch):
        """Save checkpoint."""
        checkpoint_dir = self.checkpoint_dir+ '/' + os.environ['SLURM_JOB_ID'] + '/';
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint = { "model": self.model.state_dict(), 
                       "optimizer": self.optim.state_dict(), 
                       "scaler": self.scaler.state_dict()}
        torch.save(checkpoint, os.path.join(checkpoint_dir, '%s_net_%s.pth' % (str(epoch), self.modelname)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--split_seed',
        type=int,
        default=10000,
        help='Which random seed to use for splitting data',
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=0,
        help='Which fold to run',
    )
    parser.add_argument(
        '--model_args_file',
        type=str,
        default='model.json',
        help='Location of json file describing model',
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=500,
        help='How many epochs to run',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='How many examples to process between gradient steps',
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        default=224,
        help='patch_size for sampling',
        )
    parser.add_argument(
        '--recon_size',
        type=int,
        default=64,
        help='decoder reconstruction size',
    )
    parser.add_argument(
        '--norecon_loss',
        action='store_true',
        help='No reconstruction loss',
    ) 
    parser.add_argument(
        '--num_ge',
        type=int,
        default=93,
        help='number of gene expression profile',
    )
    parser.add_argument(
        '--num_patch',
        type=int,
        default=20,
        help='number of patch to sample',
        )
    parser.add_argument(
        '--test_num_patch',
        type=int,
        default=128,
        help='number of patch to sample',
        )
    parser.add_argument(
        '--nonimg_fc_width',
        type=int,
        default=30000,
        help='number of fully connect layers',
        )
    parser.add_argument(
        '--init_fc_method',
        type=str,
        default='pytorch_default',
        help='choose weight initialization for fully connected layers',
        )
    parser.add_argument(
        '--dropout_rate',
        type=float,
        default=0.5,
        help='dropout rate',
        )
    parser.add_argument(
        '--num_bag',
        type=int,
        default=2000,
        help='number of bag',
    )
    parser.add_argument(
        '--num_img_channel',
        type=int,
        default=3,
        help='number of channels in image',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=10,
        help='number of workers for reading patch',
        )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-2,
        help='Learning rate for optimizer',
    )
    parser.add_argument(
        '--final_lr',
        type=float,
        default=1e-3,
        help='final learning-rate used by adabound',
    )
    parser.add_argument(
        '--kl_coeff',
        type=float,
        default=0.000001,
        help='coefficient for kl term of the loss',
    )
    parser.add_argument(
        '--latent_dim',
        type=int,
        default=2048,
        help='latent dimension',
    )
    parser.add_argument(
        '--warmup_k',
        type=int,
        default=-1,
        help='warm up epoch',
    )
    parser.add_argument(
        '--warmup_lr',
        type=float,
        default=0.0001,
        help='learning rate in warm-up phase',
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0,
        help='weight_decay for optimizer',
    )
    parser.add_argument(
        '--hide_progress',
        action='store_true',
        help='Do not print progress bar (saves space in log files)',
    )
    parser.add_argument(
        '--gpu',
        '-g',
        type=int,
        default=0,
        help='Index of GPU to use',
    )
    parser.add_argument(
        '--dataset_csv',
        '-d',
        type=str,
        default='data/small_linked_tcga-brca_data.csv',
        help='Dataset',
    )
    parser.add_argument(
        '--seg_dir',
        '-sd',
        type=str,
        default='/data/Jiang_Lab/Data/tcga-brca-segmentations/',
        help='Segmentation directory',
    )
    parser.add_argument(
        '--gene_file',
        type=str,
        default='data/tcga-brca/gene_expression.csv',
        help='Gene expression file',
    )
    parser.add_argument(
        '--output_dir',
        '-o',
        type=str,
        default='/pylon5/med200003p/dubeyak/output_dir/',
        required=True,
        help='Directory where we will write all output',
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='/pylon5/med200003p/dubeyak/checkpoint_dir/',
        help='Directory where we will store checkpoint models.',
    )
    parser.add_argument(
        '--pretrained_dir',
        type=str,
        default=None,
        help='Previous output_dir of a CNN from which to load initial state.',
    )
    parser.add_argument(
        '--continue_train',
        action='store_true',
        help='Resume training from pretrained model',
    )
    parser.add_argument(
        '--reference_patch',
        type=str,
        default=None,
        help='Provide reference patch to load',
    )
    parser.add_argument(
        '--optimizer', 
        default='Adam',
        choices=['Adam', 'SGD', 'AdaBound'],
        help='Choose optimizer',
    )
    parser.add_argument(
        '--num_class',
        type=int,
        default=4,
        help = 'number of class',
    )
    parser.add_argument(
        '--use_img_network',
        action='store_true',
        help='Use image feature network',
    )
    parser.add_argument(
        '--multitask',
        action='store_true',
        help='Use multitask learning',
    )
    parser.add_argument(
        '--use_nonimg_network',
        action='store_true',
        help='Use non-image feature network', 
    )
    parser.add_argument(
        '--description',
        default='none',
        type=str.lower,
        help='experiment description',
    )
    parser.add_argument(
        '--save_model_freq', 
        type=int, 
        default=5, 
        help='save model after save_model_frequency epochs'
    )

    parser.add_argument(
        '--distributed',
        action='store_true',
        help='Use distributed data parallel'
    )

    parser.add_argument(
        '--use_scheduler',
        action='store_true',
        help='Use scheduler',
    )

    parser.add_argument(
        '--color_norm',
        action='store_true',
        help='Color Normalization'
    )

    parser.add_argument(
        '--fetchge',
        action='store_true',
        help='getch gene expression'
    )

    parser.add_argument(
        '--stat_norm_scheme',
        type=str,
        default="pretrained",
        help='statistical normalization scheme'
    )

    parser.add_argument(
        '--local_rank', 
        type=int, 
        default=0, 
        help='local_rank'
    )

    parser.add_argument(
        '--use_amp',
        action='store_true',
        help='use automatic mix precision',
    )

    parser.add_argument(
       '--mpp',
        type=int, 
        default=0.5, 
        help='micron per pixel'
    )

    args = parser.parse_args()

    logger.info(args)

    # Reproducibility
    # cf. https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(0)
    np.random.seed(0)
 
    try:
        args.device=f'cuda:{args.gpu}'
        args.name='survival_analysis'

        if args.distributed:
            args.world_size = torch.cuda.device_count()
            dist.init_process_group('nccl', rank=args.local_rank, world_size=args.world_size)
        else:
            args.local_rank = 0
            args.world_size = 1
        args.device = torch.device('cuda', args.local_rank)
        args.progress=not args.hide_progress 

        t = Trainer(vars(args))
        
        t.train()

    finally:
        if args.distributed==True:
            dist.destroy_process_group()
