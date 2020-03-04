# Code to extract data from .gif files into .npy arrays

####  Feb 27, 2020: This is an old code that made single channel .npy files. A new code was developed to make it a 3 channel array for 3 file types.
### Feb 14, 2020 
##### Venkitesh Ayyar (vpa@lbl.gov)

import sys
import os
import numpy as np
import time
import pandas as pd

import matplotlib.pyplot as plt


#######################################
#######################################

def f_get_df(mode):
    '''
    Function to get Dataframe and shuffle entires
    3 modes: 
    - full: Get a big dataframe, shuffling all entries
    - split: Split dataframe into :srch,temp,diff and shuffle each and return list of 3 dataframes
    
    '''
    
    data_dir='/global/project/projectdirs/dasrepo/vpa/supernova_cnn/data/gathered_data/'
    fname1=data_dir+'summary_label_files.csv'
    df=pd.read_csv(fname1,sep=',',comment='#')
    
    print(df.shape)
    num_samples=df.shape[0]
    num_sig,num_bkgnd=df[df.Label==1].shape[0],df[df.Label==0].shape[0]
    print("Proportion of Signal-Background: {0}-{1}\nProportion of Signal: {2}".format(num_sig,num_bkgnd,num_sig*1.0/(num_sig+num_bkgnd)))
        
    if mode=='full':
        df=df.sample(frac=1.0,random_state=37)
#         df=df.head(99)
        return df
    
#     df=df.head(99)
    
    ### Mode=split
    if mode=='split':
        df_list=[pd.DataFrame([]) for i in range(3)]
        str_lst=['srch','temp','diff']
        for count,strg in enumerate(str_lst):
            dfr=df[df.filename.str.startswith(strg)]
            df_list[count]=dfr.sample(frac=1.0,random_state=37)
    
        return df_list
    
    
def f_get_data(df):
    '''
    Function to get data from .gif files into index, images, labels.
    Uses matplotlib.pyplot.imread 
    '''
    
    file_list=df['file path'].values
    ### Get images into numpy array
    img=np.array([plt.imread(fle) for fle in file_list])
    images = np.expand_dims(img, -1)
    
    ### Extract labels
    labels = df['Label'].values
    ### Store the index value of the dataframe
    idx=df.index.values 
    
    
    return idx,images,labels

def f_save_files(idx,img,label,name_prefix):
    '''
    Save the index, image and label in 3 .npy files
    '''
    save_location='/global/project/projectdirs/dasrepo/vpa/supernova_cnn/data/gathered_data/temp_data/'
    f1,f2,f3=[name_prefix+i for i in ['_idx','_x','_y']]
    
    for fname,data in zip([f1,f2,f3],[idx,img,label]):
        np.save(save_location+fname,data)


########################
##### Main Code ########
if __name__=='__main__':
    
    ### Two modes: full-> entire data is shuffled and stored. split-> data is split into 
    #temp,srch, diff and each is shuffled and stored in a file
    mode='full'
    mode='split'
    
    if mode=='full':
        
        t1=time.time()
        df=f_get_df(mode)
        t2=time.time()
        print("Setup time",t2-t1)
        idx,img,label=f_get_data(df)
        t3=time.time()
        print("Extraction time",t3-t2)
        f_save_files(idx,img,label,name_prefix='full_')
        t4=time.time()
     
    if mode=='split':

        t1=time.time()
        ### Get a list of Dataframes for srch,temp,diff files
        df_list=f_get_df(mode)
        t2=time.time()
        print("Setup time",t2-t1)    

        str_lst=['srch','temp','diff']
        for count,(df,prefix) in enumerate(zip(df_list,str_lst)):
            print(count,prefix)
            ### Get numpy arrays for each type
            idx,img,label=f_get_data(df)
            t3=time.time()
            print("Extraction time for %s : %s " %(count,t3-t2))
            ### Save data to .npy files 
            f_save_files(idx,img,label,prefix)
            t4=time.time()

            print("File save time for %s : %s "%(count,t4-t3))
