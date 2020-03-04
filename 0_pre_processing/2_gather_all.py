# Code to gather ID,label and file paths for every image in the dataset

### Feb 11,2020
### author: Venkitesh. vpa@lbl.gov

## The code reads in 2 dataframes: One contains all info including ID and Labels; the other contains the ID and file paths. This code matches the IDs to get the Labels


import numpy as np
import pandas as pd
import time

#####################
### Define locations

data_dir='/global/project/projectdirs/dasrepo/vpa/supernova_cnn/data/raw_data/'
f1='/global/project/projectdirs/dasrepo/vpa/supernova_cnn/data/gathered_data/autoscan_features.3.csv'
f2='/global/project/projectdirs/dasrepo/vpa/supernova_cnn/data/gathered_data/gathered_filepaths.csv'

t1=time.time()

### Read first dataframe and get labels
df1=pd.read_csv(f1,sep=',',comment='#')
print(df1.shape)

### Grab ID and label and clear memory
df3=df1[['ID','OBJECT_TYPE']]
del(df1)
df3.head(5)

### Read second dataframe with filepaths
df2=pd.read_csv(f2,sep=',',comment='#')
print(df2.shape)

#####################
#### Iterate over first dataframe, grab labels and add to second dataframe
t2=time.time()

### Method using for loop
# df2['Label']=""
# for index,row in df2.iterrows():
#     id=row['ID']
#     a1=df3[df3.ID==id]
#     a2=a1['OBJECT_TYPE'].values
#     assert len(a2)==1, "Return array instead of a number"%(a2)
#     label=a2[0]
# #     print(id,label)
#     df2['Label']=label

### Alternative way using Pandas
df2=pd.merge(df2,df3,on='ID',how='outer')
df2=df2.rename({'OBJECT_TYPE':'Label'},axis=1) ### Rename column name to 'Label'

t3=time.time()
print("Time of operation",t3-t2)
print("Total time",t3-t1)


#####################
### Write to file
fname='/global/project/projectdirs/dasrepo/vpa/supernova_cnn/data/gathered_data/summary_label_files.csv'
df2.to_csv(fname,mode='w',index=False)


