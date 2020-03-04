# Code to find and write the location of gifs using the ID in csv
### The code is written out in blocks to ensure low memory usage

import numpy as np
import pandas as pd
import subprocess as sp
import time
from pathlib import Path

#####################
### Define locations

ip_fname='/global/project/projectdirs/dasrepo/vpa/supernova_cnn/data/raw_data/autoscan_features.3.csv'
df1=pd.read_csv(ip_fname,sep=',',comment='#')

data_dir='/global/project/projectdirs/dasrepo/vpa/supernova_cnn/data/raw_data/'
fname='data_file_names.csv'

### Collect IDs and labels
id_list=df1.ID.values
label_list=df1.OBJECT_TYPE.values
del(df1)
# num_samples=len(id_list)
# blocks_size=100

num_samples,block_size=100,50
assert block_size<num_samples, "Block size %s is not smaller than total number %s"%(block_size,num_samples)


#####################
### Write column names to file 
cols=['ID','label','file_path']
df2=pd.DataFrame(columns=cols)
df2.to_csv(fname,mode='w',index=False)

t1=time.time()
#####################
### Iterate through IDs and get file names in a list and write them to file in blocks

for count in range(num_samples):
    
    ta=time.time()
    blk_idx=count/block_size
    
    ### Create dataframe again for each block
#     id=df1.iloc[count].ID
#     label=df1.iloc[count].OBJECT_TYPE
    id=id_list[count]
    label=label_list[count]
    
    #print(id,label)
    ## find the location of .gif files using ID. This is a time consuming operation
    a1=[str(i) for i in Path(data_dir).rglob('*{0}.gif'.format(id))]
    #print(a1)
    if count%block_size==0: # Start of a new block
        df2=pd.DataFrame(columns=cols)
    
    ### Append to DataFrame
    dict1=dict(zip(cols,[id,label,a1]))
    df2=df2.append(dict1,ignore_index=True)
    
    ### Append to output file after each block
    if (count+1)%block_size==0: # Last element of the block
        df2.to_csv(fname,mode='a',header=False,index=False)
    tb=time.time()
    
    print("Time taken for sample %s: %s"%(count,tb-ta))
    
t2=time.time()
print("Total time taken",t2-t1)
######################
### Read files from csv ### 
df3=pd.read_csv(fname,sep=',',comment='#')
print(df3.shape)