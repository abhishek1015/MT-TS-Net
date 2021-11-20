import pandas as pd
import requests
import json
import pdb
import gzip
import mygene

project_str = 'tcga-brca'
data_basedir = '/data/dubeyak/Cancer-research/code/data/'

mg = mygene.MyGeneInfo()
file_endpt = 'https://api.gdc.cancer.gov/files/'
output_map_file = data_basedir +  project_str + '/ensembl_gene_id_to_gene_mapping.json'

gene_expression_sample_file = '/data/Jiang_Lab/Data/tcga-brca/67a2c791-b15d-45b3-b41b-c48fa1c43b79/9c67b11d-1bac-446f-8bd7-ca34b7bc2ad6.FPKM.txt.gz'
symbol_dict={}
row_dict={}
genes=[]
expressions=[] 

f_uuid = gene_expression_sample_file.split('/')[5]
f_name = gene_expression_sample_file
with gzip.open(f_name,'rt') as f:
    for line in f:
        gene = line.split()[0]
        genes.append(gene.split('.')[0])
symbols = mg.getgenes(genes, fields='symbol') 


pdb.set_trace()
ge = pd.read_csv(data_basedir +  project_str + '/gene_expression.csv', delimiter=',')
threshold = 0.2
#threshold = 0
gene_above_threshold = ((ge!=0).astype(int).sum(axis=0)>=len(ge)*threshold).astype(int)

for g in range(len(symbols)):
    if 'symbol' in symbols[g].keys():
        if symbols[g]['symbol'] in gene_above_threshold[gene_above_threshold==1].keys():
            symbol_dict[symbols[g]['query']] = symbols[g]['symbol']

with open(output_map_file, 'w') as fp:
    json.dump(symbol_dict, fp)
