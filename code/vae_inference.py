from __future__ import print_function
from torch.nn import functional as F
from basic_vae_module import VAE
import pytorch_lightning as pl
import PIL
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from utils import *
from datetime import datetime
from torchvision.utils import save_image
import torchvision.utils as vutils
import torchvision
from torchvision import datasets, transforms
import torch

class customVAE(VAE):
    def __init__(self,latent_dim,enc_out_dim,enc_type,first_conv,maxpool1, *args, **kwargs):
        super(customVAE, self).__init__(latent_dim=int(latent_dim),enc_out_dim=int(enc_out_dim),enc_type=enc_type,first_conv=first_conv,maxpool1=maxpool1,input_height=64,input_channels=3,*args)
        self.example_input_array = torch.rand(1, 3, 64, 64)
        self.test_outs = []
        self.time = datetime.now()        

    def training_epoch_end(self,output):
        now = datetime.now()
        delta = now - self.time
        self.time = now
        tensorboard_logs = {'time_secs_epoch': delta.seconds}
        self.log_dict(tensorboard_logs) 

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        if self.global_rank == 0:
            if batch_idx == 0:
                self.test_outs = batch
        return loss

    def validation_epoch_end(self,output):
        if self.global_rank == 0:
            x, y = self.test_outs
            z, x_hat, p, q = self._run_step(x) 
            vutils.save_image(x_hat,
                f"{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/"
                f"recons_{self.logger.name}_{self.current_epoch}.png",
                normalize=True,
                nrow=8)
            if self.current_epoch == 0:
                vutils.save_image(x,
                    f"{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/"
                    f"orig_{self.logger.name}_{self.current_epoch}.png",
                    normalize=True,
                    nrow=8)

    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction='mean')
        output_comparison = torch.cat((x,x_hat),2)    
        vutils.save_image(output_comparison,
                    f"{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/"
                    f"inference_{self.logger.name}_batch_idx{batch_idx}.png",
                    normalize=True,
                    nrow=8)
        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def test_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss
