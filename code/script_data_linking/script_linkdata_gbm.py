import pandas as pd
import csv
import pdb, traceback, sys
import numpy as np
import math

project_id ={'3': 'tcga-gbm'}
base_data_dir = '/data/dubeyak/Cancer-research/code/data/'
seg_dir = '/data/Jiang_Lab/Data/'
project = '3'
patch_count_threshold = int(input('patch count threshold: '))
svsfile = pd.read_csv(base_data_dir + project_id[project] + '/svsfiles.xml')
clinicalfile = pd.read_csv(base_data_dir + project_id[project] + '/clinical.tsv', delimiter='\t')
immunofile = pd.read_csv(base_data_dir +'../script_data_download/Immunosubtype.txt', delimiter='\t')
tidefile = pd.read_csv(base_data_dir + 'tide-results/Tumor_Dysf_Excl_scores/TCGA.GBM.RNASeq.norm_subtract.OS_base', delimiter='\t')
number_available_sampled_patches = pd.read_csv(seg_dir + project_id[project] + '-segmentations/number_sampled_patches.csv', delimiter=',')
#immunofile = immunofile.fillna('NA')
info_columns = ['filename', 'duration', 'event', 'ajcc_pathologic_stage', 'ajcc_pathologic_n', 'ajcc_pathologic_m', 'ajcc_pathologic_t', 'morphology', 'primary_diagnosis', 'age_at_index', 'tcga_subtype', 'MDSC', 'CAF', 'M2', 'Exclusion', 'Dysfunction']

with open(base_data_dir + project_id[project] + '/linked_data.csv', mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(info_columns)
    for case in range(len(svsfile)):
        filename=svsfile.iloc[case][0]
        case_submitter_id = filename.split('/')[3]
        case_submitter_id = case_submitter_id[0:12]
        days_to_death = clinicalfile[clinicalfile['case_submitter_id']==case_submitter_id]['days_to_death']
        days_to_last_follow_up = clinicalfile[clinicalfile['case_submitter_id']==case_submitter_id]['days_to_last_follow_up']
        vital_status = clinicalfile[clinicalfile['case_submitter_id']==case_submitter_id]['vital_status']
        ajcc_pathologic_stage = clinicalfile[clinicalfile['case_submitter_id']==case_submitter_id]['ajcc_pathologic_stage']
        ajcc_pathologic_n = clinicalfile[clinicalfile['case_submitter_id']==case_submitter_id]['ajcc_pathologic_n']
        ajcc_pathologic_m = clinicalfile[clinicalfile['case_submitter_id']==case_submitter_id]['ajcc_pathologic_m']
        ajcc_pathologic_t = clinicalfile[clinicalfile['case_submitter_id']==case_submitter_id]['ajcc_pathologic_t']
        morphology = clinicalfile[clinicalfile['case_submitter_id']==case_submitter_id]['morphology']
        primary_diagnosis = clinicalfile[clinicalfile['case_submitter_id']==case_submitter_id]['primary_diagnosis']
        age_at_index = clinicalfile[clinicalfile['case_submitter_id']==case_submitter_id]['age_at_index']
        tcga_subtype = immunofile[immunofile['TCGA Participant Barcode'] ==case_submitter_id]['TCGA Subtype']

        if case_submitter_id in tidefile.index:
            MDSC = tidefile.loc[case_submitter_id]['MDSC']
            CAF = tidefile.loc[case_submitter_id]['CAF']
            M2 = tidefile.loc[case_submitter_id]['M2']
            Exclusion = tidefile.loc[case_submitter_id]['Exclusion']
            Dysfunction = tidefile.loc[case_submitter_id]['Dysfunction']
        else:
            MDSC=0
            CAF=0
            M2=0
            Exclusion=0
            Dysfunction=0

        # https://www.biostars.org/p/96209/ 
        #pdb.set_trace()
        if len(vital_status)>0:
            if vital_status.values[0] == 'Alive':
                duration = days_to_last_follow_up.values[0]
                event = str(0)
            elif vital_status.values[0] == 'Dead':
                duration = days_to_death.values[0]
                event = str(1)
            else:
                print('Error msg: Nor alive or dead!')
                continue
            
            _patch_count = number_available_sampled_patches[number_available_sampled_patches['case_id'] == filename.split('/')[3].replace('.svs', '')]['number_sampled_patches']
            
            if (duration == "\'--"
                or len(tcga_subtype) == 0 
                or pd.isnull(tcga_subtype.values[0])
                or _patch_count.values[0] < patch_count_threshold
               ):
                  continue


            csv_writer.writerow([filename, 
                duration, 
                event, 
                0, 
                0, 
                0, 
                0,
                morphology.values[0],
                primary_diagnosis.values[0],
                age_at_index.values[0],
                tcga_subtype.values[0],
                MDSC,
                CAF,
                M2,
                Exclusion,
                Dysfunction])
