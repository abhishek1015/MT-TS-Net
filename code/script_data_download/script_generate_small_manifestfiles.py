import pandas as pd
import pdb
import os
import math

project_id = {'1': 'tcga-brca', '2': 'tcga-luad'}

manifest_file=input('input manifest file: ')
n_manifests = int(input('number of manifests: '))
project = input('input project id. enter 1 for tcga-brca and 2 for tcga-luad: ' )

dataloc = '/data/Jiang_Lab/Data/' + project_id[project]

df = pd.read_csv(manifest_file, sep='\t');
files = df['id'];
downloaded = {}
subfolders = [ f.path for f in os.scandir(dataloc) if f.is_dir() ]
for x in subfolders:
  x = os.path.relpath(x, dataloc);
  downloaded[x]=1

# remains to be downloaded
remaining=[];
count = 0
for f in files:
  if f not in downloaded.keys():
    remaining.append(f);

count = len(remaining)
n_item_per_manifest = math.ceil(count/n_manifests);

part_idx = 0;
l_count = 0;
for i in range(len(df)):
  if l_count > n_item_per_manifest:
    part_manifest_name = 'part_manifests_' + project_id[project] + '/' + manifest_file + '_part' + str(part_idx);
    part_frame.to_csv(part_manifest_name, sep='\t', header=True);
    l_count=0
    part_idx= part_idx+1
  if l_count==0:
    part_frame = pd.DataFrame(columns=df.columns);
  idstr = df.iloc[i]['id']
  if idstr not in downloaded:
  	part_frame.loc[l_count] = df.iloc[i];
  	l_count = l_count+1
part_manifest_name = 'part_manifests_' + project_id[project] + '/' + manifest_file + '_part' + str(part_idx);
part_frame.to_csv(part_manifest_name, sep='\t', header=True);
