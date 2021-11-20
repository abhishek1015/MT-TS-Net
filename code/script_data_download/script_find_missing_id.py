import pandas as pd
import pdb

downloaded_svs = pd.read_csv('svsfiles.xml')
for x in range(len(downloaded_svs)):
  tmp = downloaded_svs.iloc[x]['filename'];
  downloaded_svs.loc[x]=tmp.split('/')[-1]
all_items_in_manifestfile = pd.read_csv('gdc_manifest.2020-12-02-tcga-brca.txt', sep='\t');

count=0
frame = pd.DataFrame(columns=all_items_in_manifestfile.columns)
for x in range(len(all_items_in_manifestfile)):
  f = all_items_in_manifestfile.iloc[x]['filename'] 
  if f.endswith('svs') and f not in downloaded_svs['filename'].values:
    #pdb.set_trace()
    frame.loc[count]=all_items_in_manifestfile.iloc[x];
    count=count+1;
manifest_name = 'missing_svs_manifests.txt';
frame.to_csv(manifest_name, sep='\t', header=True);  
