# Code to extract data from .gif files into .npy arrays
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
    
def f_get_df():
    '''
    Function to get Dataframe and shuffle entires
    3 modes: 
    - full: Get a big dataframe, shuffling all entries
    - split: Split dataframe into :srch,temp,diff and shuffle each and return list of 3 dataframes
    '''
    
    data_dir='/global/project/projectdirs/dasrepo/vpa/supernova_cnn/data/gathered_data/'
    fname1=data_dir+'summary_label_files.csv'
    df=pd.read_csv(fname1,sep=',',comment='#')
    
    ### Print summary of data
    print(df.shape)
    num_samples=df.shape[0]
    num_sig,num_bkgnd=df[df.Label==1].shape[0],df[df.Label==0].shape[0]
    print("Proportion of Signal-Background: {0}-{1}\nProportion of Signal: {2}".format(num_sig,num_bkgnd,num_sig*1.0/(num_sig+num_bkgnd)))

    df=df.head(300000)  ## When using parts, make sure you take multiples of 3 so that all 3 files for each ID are taken
#     df=df.sample(frac=1.0,random_state=37)  ### The shuffling part is done in the np array of IDs rather than here. 
    
    return df


def f_get_data(df):
    '''
    Function to get data from .gif files into index, images, labels.
    Uses matplotlib.pyplot.imread 
    '''
    
    combined_imgs_lst=[]
    label_lst=[]
    ### Get list of IDs. Each ID has a srch,temp,diff file
    idx_arr=np.unique(df.ID.values)
    
    ## Shuffle IDs
    np.random.seed(37)
    np.random.shuffle(idx_arr) ## When using parts, make sure you take multiples of 3 so that all 3 files for each ID are taken

    ### Iterate over IDs, stacking 3 numpy arrays (temp,srch,diff) for each
    for idx in idx_arr:
        try: 
            ### Extract the 3 images and create stacked numpy array
            file_list=[df[(df.ID==idx) & (df.filename.str.startswith(strg))]['file path'].values[0] for strg in ['temp','srch','diff']]
            
            img=np.dstack([plt.imread(fle) for fle in file_list]) ## Create stacked numpy array of 3 images
        #     img=np.expand_dims(img,axis=0)
            combined_imgs_lst.append(img)             ## Append image to list

            ### Extract the first label
            label=[df[(df.ID==idx) & (df.filename.str.startswith(strg))]['Label'].values[0] for strg in ['temp','srch','diff']]
            ## Check that all 3 images have same label
            assert all(x==label[0] for x in label), "Labels for temp,srch,diff are not identical %"%(label)
            label_lst.append(label[0])
            
        except Exception as e:
            print(e,'for index',idx)
            pass
    
    ### Stack the combined image list
    images=np.stack(combined_imgs_lst,axis=0)
    print(images.shape)
    
    ### Extract labels
    labels = np.array(label_lst)
    
    ### Store the ID of the dataframe
    idx=idx_arr[:]
    
    return idx,images,labels 


def f_save_files(idx,img,label,name_prefix):
    '''
    Save the ID, image and label in 3 .npy files
    '''
    save_location='/global/project/projectdirs/dasrepo/vpa/supernova_cnn/data/gathered_data/'
    f1,f2,f3=[name_prefix+i for i in ['idx','x','y']]
    
    for fname,data in zip([f1,f2,f3],[idx,img,label]):
        np.save(save_location+fname,data)

        
if __name__=='__main__':

    t1=time.time()
    df=f_get_df()
    t2=time.time()
    #df=df.head(300)
    print("Setup time",t2-t1)
    idx,img,label=f_get_data(df)
    t3=time.time()
    print("Extraction time",t3-t2)
    f_save_files(idx,img,label,name_prefix='full_')
    t4=time.time()
    
    print("File save time %s"%(t4-t3))
