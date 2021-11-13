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
sys.path.append('/data/dubeyak/Cancer-research/code/ext/CLAM/')
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core import wsi_utils
from wsi_core.util_classes import isInContourV1, isInContourV2, isInContourV3_Easy, isInContourV3_Hard, Contour_Checking_fn
from sklearn.model_selection import train_test_split
import pandas as pd
import gzip

class SVSData(Dataset):
    def __init__(self, csvdata, seg_dir, mpp, patch_size, num_patches, num_workers, color_norm, stat_norm_scheme, _q, fetchpatch, fetchge, testmode, reference_filename):
        super(SVSData, self).__init__()
        self.base_dir='/data/Jiang_Lab/Data/';
        self.seg_dir=seg_dir
        self.imageFilenames = csvdata
        self._q=_q
        self.fetchpatch = fetchpatch
        self.fetchge = fetchge
        self.mpp = mpp
        self.patchSize = patch_size
        self.numPatches = num_patches
        self.color_norm = color_norm
        self.stat_norm_scheme = stat_norm_scheme
        self.num_workers=num_workers
        self.sampled_coords = []
        self.reference_filename = reference_filename
        pool = Pool(processes=self.num_workers)
        self.ensembl_gene_id_to_gene_mapping_ctl = pd.read_json('/data/dubeyak/Cancer-research/code/data/tcga-brca/ensembl_gene_id_to_gene_mapping_CTLonly.json', typ='series')
        self.ensembl_gene_id_to_gene_mapping = pd.read_json('/data/dubeyak/Cancer-research/code/data/tcga-brca/ensembl_gene_id_to_gene_mapping_pwonly.json', typ='series')
        #self.ensembl_gene_id_to_gene_mapping = pd.read_json('/data/dubeyak/Cancer-research/code/data/tcga-brca/ensembl_gene_id_to_gene_mapping.json', typ='series')
        self.testmode=testmode
        if self.fetchpatch and self.color_norm:
            self.normalizer = None
            if self.reference_filename != None:
                self.reference_patch = torch.load(self.reference_filename)
            elif self.testmode:
                self.reference_patch = torch.load('reference_patch_' +  os.environ['SLURM_JOB_ID'] + '.pkl')
            else:
                index = random.randint(0, len(self.imageFilenames))
                files = self.imageFilenames['histopathology_files'][index]
                files = files.split(',')
                files = [x.replace('[', '').replace(']', '').replace('\'', '').strip() for x in files]
                fidx = random.randint(0,len(files)-1)
                reference_filename = self.base_dir+files[fidx]
                reference_img = [pyvips.Image.new_from_file(reference_filename)]
                segmentation_dict = self._load_segmentation([files[fidx]])
                reference_patches = self._patching(reference_img, segmentation_dict, 3)
                self.reference_patch = reference_patches[0]
                torch.save(self.reference_patch, 'reference_patch_' +  os.environ['SLURM_JOB_ID'] + '.pkl')
            self.normalizer = torchstain.MacenkoNormalizer(backend='torch')
            self.normalizer.fit(self.reference_patch*255)
     
    def _normalize_image(self, to_transform):
        norm,_,_ = self.normalizer.normalize(I=to_transform*255, stains=False)
        norm = norm.permute(2, 0, 1)/255.0
        return norm

    def _load_segmentation(self, _fids):
        segmentation_dict=[]
        for _fid in _fids:
            _fid = _fid.split('/')[3].replace('.svs', '.pkl')
            pickle_file = self.seg_dir + _fid
            a_file = open(pickle_file, "rb")
            segmentation_dict.append(pickle.load(a_file))
            a_file.close()
        return segmentation_dict

    def get_gene_expression(self, f_name, mapping, _sum):
        expressions=np.zeros([len(mapping.index.values)])
        if f_name != 'NA':
            with gzip.open(self.base_dir + f_name,'rt') as f:
                ln=0
                for line in f:
                    ensembl_gene_id = line.split()[0].split('.')[0]
                    if ensembl_gene_id in mapping:
                        exp = float(line.split()[1])
                        expressions[ln]=exp
                        ln=ln+1
        ge_value = torch.from_numpy(expressions)
        ge_value = torch.log2(ge_value + 1.0)
        if _sum:
            ge_value = torch.sum(ge_value)
        return ge_value

    def get_duration_label(self, duration):
        for idx_ql in range(len(self._q)):
            if duration < self._q[idx_ql]:
                return idx_ql
        return len(self._q)
 
    def get_lymph_node_extent(self, ajcc_pathologic_n):
        if ajcc_pathologic_n in ['N0', 'N0 (i+)', 'N0 (i-)', 'N0 (mol+)']:
            n = 0
        elif ajcc_pathologic_n in ['N1', 'N1a', 'N1b', 'N1c', 'N1mi']:
            n = 1
        elif ajcc_pathologic_n in ['N2', 'N2a', 'N2b', 'N2c']:
            n = 2
        elif ajcc_pathologic_n in ['N3', 'N3a', 'N3b', 'N3c']:
            n = 3
        elif ajcc_pathologic_n in ['NX']:
            n = 4
        else:
            n = -1
            #raise  ValueError('Unknown ajcc_pathologic_n: '+ ajcc_pathologic_n)
        return n

    def get_tumor_label(self, ajcc_pathologic_t):
        if ajcc_pathologic_t in ['T1', 'T1a', 'T1b', 'T1c', 'T1d']:
            t = 0
        elif ajcc_pathologic_t in ['T2', 'T2a', 'T2b', 'T2c', 'T2d']:
            t = 1
        elif ajcc_pathologic_t in ['T3', 'T3a', 'T3b', 'T3c', 'T3d']:
            t = 2
        elif ajcc_pathologic_t in ['T4', 'T4a', 'T4b', 'T4c', 'T4d']:
            t = 3
        else:
            t = -1
            #raise  ValueError('unknown ajcc_pathologic_t: ' + ajcc_pathologic_t)
        return t

    def get_stage_label(self, ajcc_pathologic_stage):
        if ajcc_pathologic_stage in ['Stage I', 'Stage IA', 'Stage IB']:
            stage = 0
        elif ajcc_pathologic_stage in ['Stage II', 'Stage IIA', 'Stage IIB']:
            stage = 1
        elif ajcc_pathologic_stage in ['Stage III', 'Stage IIIA', 'Stage IIIB', 'Stage IIIC']:
            stage = 2
        elif ajcc_pathologic_stage in ['Stage IV']:
            stage = 3
        else:
            stage = -1
            #raise ValueError('Unknown stage: ' + ajcc_pathologic_stage)
        return stage  
 
    def get_morphology_label(self, morphology):
        # handling brca dataset
        if  self.seg_dir.split('/')[4].split('-')[1] == 'brca':
            if morphology == '8500/3':
                morphology_label = 0
            elif morphology == '8520/3':
                morphology_label = 1
            elif morphology == '8522/3':
                morphology_label = 2
        elif self.seg_dir.split('/')[4].split('-')[1] == 'luad':
           morph_dict={'8140/3': 0, '8255/3':1 , '8252/3':2, '8480/3':3, '8253/3':4, '8260/3':5, '8550/3':6, '8265/3':7, '8250/3':8, '8230/3':9, '8310/3':10, '8490/3':11}
           if morphology in morph_dict.keys():
               morphology_label=morph_dict[morphology]
           else:
               morphology_label=12
        else:
           morphology_label=0
        return morphology_label

    def get_immune_subtype_label(self, immune_subtype):
        immune_dict={'C1': 0, 'C2': 1, 'C3': 2, 'C4': 3, 'C5': 4, 'C6': 5}
        immune_subtype_label = immune_dict[immune_subtype]
        return immune_subtype_label

    def get_tcga_subtype_label(self, tcga_subtype):
        if  self.seg_dir.split('/')[4].split('-')[1] == 'brca':
            subtype_dict = {'BRCA.LumA': 0, 'BRCA.LumB': 1, 'BRCA.Basal': 2, 'BRCA.Her2': 3,  'BRCA.Normal': 4}
            subtype = subtype_dict[tcga_subtype]
        elif self.seg_dir.split('/')[4].split('-')[1] == 'luad':
            subtype_dict = {'LUAD.1': 0, 'LUAD.2': 1, 'LUAD.3': 2, 'LUAD.4': 3, 'LUAD.5': 4, 'LUAD.6': 5}
            if tcga_subtype in subtype_dict.keys():
                subtype = subtype_dict[tcga_subtype]
            else:
                subtype = 6
        elif self.seg_dir.split('/')[4].split('-')[1] == 'gbm':
           subtype = 0
        return subtype

    def _load_file(self,index):
        if self.fetchpatch:
            filenames = self.imageFilenames['histopathology_files'][index]
            filenames = filenames.split(',')
            filenames = [x.replace('[', '').replace(']', '').replace('\'', '').strip() for x in filenames]
            image=[]
            if self.testmode:
                img_filename = filenames
                for filename in filenames:
                    image.append(pyvips.Image.new_from_file(self.base_dir+filename))   
            else:
                fidx = random.randint(0,len(filenames)-1)
                img_filename = [filenames[fidx]]
                image.append(pyvips.Image.new_from_file(self.base_dir+filenames[fidx]))
        else:
            image = None
            img_filename=['NA']
            fidx=0
            filenames=['NA']
        duration = self.imageFilenames['duration'][index]
        if len(self._q)>0:
            duration_q = self.get_duration_label(duration)
        else:
            duration_q = duration
        event = self.imageFilenames['event'][index]
        ajcc_pathologic_stage = self.imageFilenames['ajcc_pathologic_stage'][index]
        ajcc_pathologic_stage = self.get_stage_label(ajcc_pathologic_stage)
        ajcc_pathologic_t = self.imageFilenames['ajcc_pathologic_t'][index]
        ajcc_pathologic_t = self.get_tumor_label(ajcc_pathologic_t)
        ajcc_pathologic_n = self.imageFilenames['ajcc_pathologic_n'][index]
        ajcc_pathologic_n = self.get_lymph_node_extent(ajcc_pathologic_n)
        age_at_index=self.imageFilenames['age_at_index'][index]
        morphology = self.imageFilenames['morphology'][index]
        morphology = self.get_morphology_label(morphology)
        tcga_subtype = self.imageFilenames['tcga_subtype'][index]
        tcga_subtype = self.get_tcga_subtype_label(tcga_subtype)
        immune_subtype = self.imageFilenames['immune_subtype'][index]
        immune_subtype = self.get_immune_subtype_label(immune_subtype)
        MDSC = self.imageFilenames['MDSC'][index]
        CAF = self.imageFilenames['CAF'][index]
        M2 = self.imageFilenames['M2'][index]
        Exclusion = self.imageFilenames['Exclusion'][index]
        Dysfunction = self.imageFilenames['Dysfunction'][index]
        if 'ge_file' in self.imageFilenames.keys():
            ge_file = self.imageFilenames['ge_file'][index]
        else:
            ge_file = 'NA'
        data = {'image': image,
                'img_filename': img_filename, 
                'duration': duration,
                'duration_q': duration_q,
                'event': event, 
                'ajcc_pathologic_stage': ajcc_pathologic_stage, 
                'ajcc_pathologic_n': ajcc_pathologic_n, 
                'ajcc_pathologic_t': ajcc_pathologic_t, 
                'morphology': morphology, 
                'age_at_index': int(age_at_index)/10.0, 
                'tcga_subtype': tcga_subtype,
                'immune_subtype': immune_subtype,
                'MDSC': MDSC,
                'CAF': CAF,
                'M2': M2,
                'Exclusion': Exclusion,
                'Dysfunction': Dysfunction,
                'ge_file': ge_file}
        return data

    def _filter_based_on_segmentation(self, segmentation_dict, rand_i, rand_j):
        for idx, cont in enumerate(segmentation_dict['contours_tissue']):
            cont_check_fn = isInContourV3_Hard(contour=segmentation_dict['contours_tissue'][idx], patch_size=self.patchSize, center_shift=1.0)
            coord=(rand_i, rand_j)
            temp = WholeSlideImage.process_coord_candidate(coord, segmentation_dict['holes_tissue'][idx], self.patchSize, cont_check_fn)
            if temp!=None:
                return True
        else:
            return False
        
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
    def _patching(self, img, segmentation_dict, num_chanel):
        div =len(img)
        numPatches_per_image = [self.numPatches // div + (1 if x < self.numPatches % div else 0)  for x in range (div)]
        #print(sum(numPatches_per_image))
        #print(numPatches_per_image)
        count = 0
        coords = []
        img_tensor_stack = torch.zeros(self.numPatches, num_chanel, self.patchSize, self.patchSize)
        for idx in range(len(img)):
            _img = img[idx]
            _segmentation_dict = segmentation_dict[idx]  
            locations = _segmentation_dict['patch_loc']
            if 'aperio.MPP' in _img.get_fields():
                _mpp = float(_img.get('aperio.MPP'))
                #_appmag = int(float(img.get('aperio.AppMag')))
                #_objective_power = int(float(img.get('openslide.objective-power')))
                #_mpp_x = float(img.get('openslide.mpp-x'))
                #_mpp_y = float(img.get('openslide.mpp-y'))
                #print(img.get('openslide.vendor'))
                downsample = round(self.mpp/_mpp)   
            else:
               downsample=2
            count_per_img=0 
            while count_per_img < numPatches_per_image[idx]: #[4, x , y] -> many [4, 512, 512]
                rand_idx = random.randint(0, len(locations)-1)
                rand_i = locations[rand_idx][0]
                rand_j = locations[rand_idx][1]
                rand_i = rand_i + random.randint(0, self.patchSize*downsample)
                rand_i = min(rand_i, _img.width-self.patchSize*downsample)
                rand_j = rand_j + random.randint(0, self.patchSize*downsample)
                rand_j = min(rand_j, _img.height-self.patchSize*downsample)
                in_segment = self._filter_based_on_segmentation(_segmentation_dict, rand_i, rand_j)
                if in_segment:
                    temp = self._img_to_tensor(_img,rand_i,rand_j, downsample)
                    temp = temp.unsqueeze(0)
                    temp = F.interpolate(temp, scale_factor=1/downsample, recompute_scale_factor=True)
                    if not self._filter_whitespace(temp):
                        #print('pass')
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
                        elif self.stat_norm_scheme == 'grayscale':
                            preprocess = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor(),
                                transforms.RandomHorizontalFlip(),
                                transforms.Normalize((0.5,), (0.5,))])
                        temp = preprocess(temp)
                        temp.unsqueeze(0)
                    img_tensor_stack[count] = temp 
                    count+=1
                    count_per_img+=1
                    #print(count)
        return img_tensor_stack

    def _img_to_tensor(self,img,x,y, downsample):
        t = img.crop(x,y,self.patchSize*downsample,self.patchSize*downsample)
        t_np = util.vips2numpy(t)
        tt_np = transforms.ToTensor()(t_np)
        out_t = tt_np[:3,:,:]
        return out_t
    def __getitem__(self, index):
        wsi_obj = self._load_file(index)
        if self.fetchge == False:
            ge = torch.empty(1)
        else:
            ge = self.get_gene_expression(wsi_obj['ge_file'], self.ensembl_gene_id_to_gene_mapping, False)
        ctl = self.get_gene_expression(wsi_obj['ge_file'], self.ensembl_gene_id_to_gene_mapping_ctl, True)
        if self.fetchpatch:
            segmentation_dict = self._load_segmentation(wsi_obj['img_filename'])
            if self.stat_norm_scheme == 'grayscale':
                img_p = self._patching(wsi_obj['image'], segmentation_dict, 1)
            else:
                img_p = self._patching(wsi_obj['image'], segmentation_dict, 3)
        else:
            img_p=torch.empty(1)
        img = {'image': img_p,
                'ge': ge,
                'ctl': ctl, 
                'duration': wsi_obj['duration'], 
                'duration_q': wsi_obj['duration_q'],
                'event': wsi_obj['event'], 
                'ajcc_pathologic_stage': wsi_obj['ajcc_pathologic_stage'], 
                'ajcc_pathologic_t': wsi_obj['ajcc_pathologic_t'], 
                'ajcc_pathologic_n': wsi_obj['ajcc_pathologic_n'],
                'morphology': wsi_obj['morphology'], 
                'age_at_index': wsi_obj['age_at_index'], 
                'tcga_subtype': wsi_obj['tcga_subtype'],
                'immune_subtype': wsi_obj['immune_subtype'],
                'MDSC': wsi_obj['MDSC'],
                'CAF': wsi_obj['CAF'],
                'M2': wsi_obj['M2'],
                'Exclusion': wsi_obj['Exclusion'],
                'Dysfunction': wsi_obj['Dysfunction']}
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
  xmlfile = "data/tcga-brca/casewise_linked_data.csv"
  dataset = pd.read_csv(xmlfile, delimiter=',')
  reference_filename = '/data/dubeyak/Cancer-research/code/reference_patches/reference_patch_23606750.pkl'
  input_data = SVSData(dataset, '/data/Jiang_Lab/Data/tcga-brca-segmentations/', 0.5, 224,128,16, True, 'pretrained', [0.25,0.5,0.75], True, True, False, reference_filename)
  #def __init__(self, csvdata, seg_dir, mpp, patch_size, num_patches, num_workers, color_norm, stat_norm_scheme, _q, fetchpatch, fetchge, testmode):
  train_loader = torch.utils.data.DataLoader(input_data,batch_size=1, shuffle=True)
  data_iter = iter(train_loader)
  _MDSC = torch.zeros(size=(len(data_iter),1))
  _CAF = torch.zeros(size=(len(data_iter),1))
  _M2 = torch.zeros(size=(len(data_iter),1))
  _Exclusion = torch.zeros(size=(len(data_iter),1))
  _Dysfunction = torch.zeros(size=(len(data_iter),1))
  _ctl = torch.zeros(size=(len(data_iter),1))
  _ge = torch.zeros(size=(len(data_iter),93))
  for i in range(len(data_iter)):
    print(i)
    batch = next(data_iter)
    #pdb.set_trace()
    _MDSC[i]=batch['MDSC'].item()
    _CAF[i]=batch['CAF'].item()
    _M2[i]=batch['M2'].item()
    _Exclusion[i]=batch['Exclusion'].item()
    _Dysfunction[i]=batch['Dysfunction'].item()
    _ctl[i]=batch['ctl'].item()
    _ge[i]=batch['ge'][0]
  print('MDSC: mean: ' + str(_MDSC.mean()) +  'std: ' + str(_MDSC.std()))
  print('CAF: mean: ' + str(_CAF.mean()) +  'std: ' + str(_CAF.std()))
  print('M2: mean: ' + str(_M2.mean()) +  'std: ' + str(_M2.std()))
  print('Exclusion: mean: ' + str(_Exclusion.mean()) +  'std: ' + str(_Exclusion.std()))
  print('Dysfunction: mean: ' + str(_Dysfunction.mean()) +  'std: ' + str(_Dysfunction.std()))
  print('ctl: mean: ' + str(_ctl.mean()) +  'std: ' + str(_ctl.std()))
  print('ge: mean: ' + str(_ge.mean()) +  'std: ' + str(_ge.std()))
  pdb.set_trace()
