from __future__ import print_function
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms, utils
from multiprocessing import Pool
import pyvips
import random
import numpy as np
import pdb
from util import util
import torchstain
import cv2
import pickle
import sys
import time
sys.path.append('/data/Jiang_Lab/Data/MT-TS-Net/code/ext/CLAM/')
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core import wsi_utils
from wsi_core.util_classes import isInContourV1, isInContourV2, isInContourV3_Easy, isInContourV3_Hard, Contour_Checking_fn
from sklearn.model_selection import train_test_split
import pandas as pd
import gzip
import matplotlib.image as mpimg

class SemanticSegData(Dataset):
    def __init__(self, csvdata, mpp, patch_size, num_patches, num_workers, color_norm, stat_norm_scheme, reference_filename):
        super(SemanticSegData, self).__init__()
        self.base_dir='/data/Jiang_Lab/Data/tcga-brca-semantic-seg/';
        self.mask_dir= self.base_dir + 'masks/'
        self.image_dir=self.base_dir + 'images/'
        self.imageFilenames = csvdata
        self.mpp = mpp
        self.patchSize = patch_size
        self.numPatches = num_patches
        self.color_norm = color_norm
        self.stat_norm_scheme = stat_norm_scheme
        self.num_workers=num_workers
        self.reference_filename = reference_filename
        self.fetchpatch=True
        pool = Pool(processes=self.num_workers)
        if self.color_norm:
            self.normalizer = None
            self.reference_patch = torch.load(self.reference_filename)
            self.normalizer = torchstain.MacenkoNormalizer(backend='torch')
            self.normalizer.fit(self.reference_patch*255)
     
    def _normalize_image(self, to_transform):
        norm,_,_ = self.normalizer.normalize(I=to_transform*255, stains=False)
        norm = norm.permute(2, 0, 1)/255.0
        return norm

    def _load_file(self,index):
        filename = self.imageFilenames['filename'][index]
        image_filename = self.image_dir + filename
        mask_filename = self.mask_dir + filename
        img = mpimg.imread(image_filename)
        mask = mpimg.imread(mask_filename)
        data = {'image': img,
                'img_filename': filename,
               'mask': mask}
        return data

    def _filter_whitespace(self,tensor_3d):
        tensor_3d=tensor_3d.squeeze(0)
        r = np.mean(np.array(tensor_3d[0]))
        g = np.mean(np.array(tensor_3d[1]))
        b = np.mean(np.array(tensor_3d[2]))
        channel_avg = np.mean(np.array([r,g,b]))
        if channel_avg < 0.82 and channel_avg >0.05:
            return True
        else:
            return False

    def _img_to_tensor(self, _img,rand_i,rand_j, downsample):
        _patch = _img[rand_i:rand_i+self.patchSize*downsample, rand_j:rand_j+self.patchSize*downsample, :]
        return torch.tensor(_patch).permute([2, 0, 1])

    def _filter_based_on_mask(self, _mask, rand_i, rand_j, downsample):
        in_segment = False
        _segmantic_seg=None
        _diff = abs(_mask[rand_i:rand_i+self.patchSize*downsample, rand_j:rand_j+self.patchSize*downsample].mean() - _mask[rand_i, rand_j])
        #print(_diff)
        if _diff < 0.00001:
            in_segment=True
            _segmantic_seg = torch.tensor(_mask[rand_i, rand_j])
        return in_segment, _segmantic_seg
    
    def _patching(self, img, mask, num_chanel):
        img_tensor_stack = torch.zeros(self.numPatches, num_chanel, self.patchSize, self.patchSize)
        semantic_seg = torch.zeros(self.numPatches)
        _img = img
        _mask = mask
        _mpp = 0.25 
        downsample = round(self.mpp/_mpp) 
        count=0  
        x_len = _img.shape[0]
        y_len = _img.shape[1]
        while count < self.numPatches: #[4, x , y] -> many [4, 512, 512]
            rand_i = random.randint(0, x_len-self.patchSize*downsample)
            rand_j = random.randint(0, y_len-self.patchSize*downsample)
            in_segment, _semantic_seg = self._filter_based_on_mask(_mask, rand_i, rand_j, downsample)
            if in_segment:
                temp = self._img_to_tensor(_img,rand_i,rand_j, downsample)
                temp = temp.unsqueeze(0)
                temp = F.interpolate(temp, scale_factor=1/downsample, recompute_scale_factor=True)
                if not self._filter_whitespace(temp):
                    continue
                if self.normalizer!=None:
                    temp = temp.squeeze(0)
                    try:
                        temp = self._normalize_image(temp)
                    except RuntimeError:
                        continue
                    if self.stat_norm_scheme == 'pretrained':
                        preprocess = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0),
                            transforms.GaussianBlur(3, sigma=(0.5, 0.5)),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                    elif self.stat_norm_scheme == 'random':
                            preprocess = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0),
                            transforms.GaussianBlur(3, sigma=(0.5, 0.5)),
                            transforms.Normalize(mean=[0.5937,0.5937,0.5937], std=[0.0810,0.0810,0.0810])])
                    temp = preprocess(temp)
                    temp.unsqueeze(0)
                img_tensor_stack[count] = temp
                semantic_seg[count]=_semantic_seg 
                count+=1
        img = {'image': img_tensor_stack,
               'semantic_seg': semantic_seg}
        return img

    def __getitem__(self, index):
        wsi_obj = self._load_file(index)
        img  = self._patching(wsi_obj['image'], wsi_obj['mask'], 3)
        return img 
    
    def __len__(self):
        return len(self.imageFilenames)

if __name__ == "__main__":
  #xmlfile = "data/tcga-brca/casewise_linked_data.csv"
  _seed=1234
  torch.manual_seed(_seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  random.seed(_seed)
  np.random.seed(_seed)
  mpp=0.5
  patch_size=64
  num_patches=10
  num_workers=8
  color_norm=True
  stat_norm_scheme="random"
  data = pd.read_csv('/data/Jiang_Lab/Data/tcga-brca-semantic-seg/semantic_seg_data.csv', delimiter=',')
  reference_filename = '/data/dubeyak/Cancer-research/code/reference_patches/reference_patch_23606750.pkl'
  input_data = SemanticSegData(data, mpp, patch_size, num_patches, num_workers, color_norm, stat_norm_scheme, reference_filename)
  train_loader = torch.utils.data.DataLoader(input_data,batch_size=1, shuffle=True)
  data_iter = iter(train_loader)
  for i in range(len(data_iter)):
    print(i)
    batch = next(data_iter)
    _img=batch['image']
    _semantic_seg=batch['semantic_seg']
