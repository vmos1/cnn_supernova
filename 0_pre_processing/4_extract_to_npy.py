# Code to extract data from .gif files into .npy arrays
### It stores the images in the form (samples,img_x,img_y,channels) where channels=0,1,2 corresponding to the 
### files temp,srch,diff
### Feb 14, 2020 
##### Venkitesh Ayyar (vpa@lbl.gov)

import sys
import os
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import argparse

## modules for parallelization of python for loop
from multiprocessing import Pool
from functools import partial

#######################################
#######################################
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Code to extract data from .gif files to .npy files', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arg = parser.add_argument
    add_arg('--batch_size','-b',type=int,default=100, help='Number of samples in each temp file.' )
    add_arg('--cores','-c',type=int,default=20,help='Number of parallel jobs you want to start')
    
    return parser.parse_args()


def f_get_df():
    '''
    Function to get Dataframe and shuffle entries
    3 modes: 
    - full: Get a big dataframe, shuffling all entries
    - split: Split dataframe into :temp,srch,diff and shuffle each and return list of 3 dataframes
    '''
    
    data_dir='/global/project/projectdirs/dasrepo/vpa/supernova_cnn/data/gathered_data/'
    fname1=data_dir+'summary_label_files.csv'
    df=pd.read_csv(fname1,sep=',',comment='#')
    
    ### Print summary of data
    print(df.shape)
    num_samples=df.shape[0]
    num_sig,num_bkgnd=df[df.Label==1].shape[0],df[df.Label==0].shape[0]
    print("Proportion of Signal-Background: {0}-{1}\nProportion of Signal: {2}".format(num_sig,num_bkgnd,num_sig*1.0/(num_sig+num_bkgnd)))

    return df


def f_get_data(df,idx_arr):
    '''
    Function to get data from .gif files into index, images, labels.
    Uses Dataframe and an array of indices to extract data.
    Uses matplotlib.pyplot.imread 
    '''
    
    combined_imgs_lst=[]
    label_lst=[]
    
    ### Shorten datafame to the IDs in idx array. This speeds up the search for values.
    df=df[df.ID.isin(idx_arr)]

    ### Iterate over IDs, stacking 3 numpy arrays (temp,srch,diff) for each
    for idx in idx_arr:
        try: 
            ### Extract the 3 images and create stacked numpy array
            file_list=[df[(df.ID==idx) & (df.filename.str.startswith(strg))]['file path'].values[0] for strg in ['temp','srch','diff']]
            
            img=np.dstack([plt.imread(fle) for fle in file_list]) ## Create stacked numpy array of 3 images
            combined_imgs_lst.append(img)             ## Append image to list

            ### Extract the first label
            label=[df[(df.ID==idx) & (df.filename.str.startswith(strg))]['Label'].values[0] for strg in ['temp','srch','diff']]
            ## Check that all 3 images have same label
            assert all(x==label[0] for x in label), "Labels for temp,srch,diff are not identical %"%(label)
            label_lst.append(label[0])
            
        except Exception as e:
            print("Found exception",e,'for index',idx)
            raise SystemError
#             pass
    
    ### Stack the combined image list
    images=np.stack(combined_imgs_lst,axis=0)
#     print(images.shape)
    
    ### Extract labels
    labels = np.array(label_lst)
    
    ### Store the ID of the dataframe
    idx=idx_arr[:]
    
    return idx,images,labels


def f_save_files(idx,img,label,name_prefix,save_location):
    '''
    Save the ID, image and label in 3 .npy files
    '''
    f1,f2,f3=[name_prefix+i for i in ['idx','x','y']]
    
    for fname,data in zip([f1,f2,f3],[idx,img,label]):
        np.save(save_location+fname,data)


def f_concat_temp_files(num_batches,save_location):
    '''
    Function to concatenate temp files to creat the full file.
    Steps: get data from temp files, stack numpy arrays and delete temp files
    '''
    
    for count in np.arange(num_batches):
        prefix='temp_data_%s'%(count)
        f1,f2,f3=[prefix+ii+'.npy' for ii in ['_x','_y','_idx']]
        
        xs,ys,idxs=np.load(save_location+f1),np.load(save_location+f2),np.load(save_location+f3)
    
        ### Join arrays to create large array
#         print(xs.shape,count+1,"out of ",num_batches)

        if count==0:
            x=xs;y=ys;idx=idxs
        else:
            x = np.vstack((x,xs))
            y = np.concatenate((y,ys))
            idx= np.concatenate((idx,idxs))
            
        for fname in [f1,f2,f3]: os.remove(save_location+fname) # Delete temp file
    print("Deleted temp files")
        
    return x,y,idx

def f_write_temp_files(count,batch_size,save_location):
    '''
    Function to write temporary files
    Arguments: count: index of idx array and batch_size : size of batch
    Takes in indices count*batch_size -> (count+1)*batch_size
    Can be used to run in parallel
    '''
    t3=time.time()
    idx,img,label=f_get_data(df,idx_arr[count*batch_size:(count+1)*batch_size])
    prefix='temp_data_{0}_'.format(count)
    f_save_files(idx,img,label,prefix,save_location)
    t4=time.time()
    print("Extraction time for count ",count,":",t4-t3)
    
    
#######################################
#######################################

if __name__=='__main__':
    
    save_location='/global/project/projectdirs/dasrepo/vpa/supernova_cnn/data/gathered_data/temp_data/'
    
    args=parse_args()
    #print(args)
    batch_size,procs=args.batch_size,args.cores
    print('batch size {0}, and processes {1}'.format(batch_size,procs))
    
    t1=time.time()
    ### Get Dataframe with indices and file names
    df=f_get_df()
    ### Uncomment the line below if you want to run a slice of the dataset
    ###df=df.head(300000) ### make sure you take multiples of 3 so that all 3 files for each ID are taken
    
    ### Get list of IDs. Each ID has a temp,srch,diff file
    idx_arr=np.unique(df.ID.values)
    #### Shuffle IDs
    np.random.seed(37)
    np.random.shuffle(idx_arr) 
    np.save(save_location+'initial_idx_arr.npy',idx_arr)  ### Save the ID file for final comparison
    
    t2=time.time()
    print("Setup time (reading the DataFrame) ",t2-t1)
    
    data_size=idx_arr.shape[0]
    batch_size=min(batch_size,data_size) ### Fix for large batch size
    num_batches=int(np.ceil(data_size/batch_size))
    print(data_size,batch_size,num_batches)
    print("Number of temp files: ",num_batches)
    
    ### Save batches of samples to temp files 
    ##### This part is parallelized
    with Pool(processes=procs) as p:
        ## Fixing the last 2 arguments of the function. The map takes only functions with one argument
        f_temp_func=partial(f_write_temp_files,batch_size=batch_size,save_location=save_location)
        ### Map the function for each batch. This is the parallelization step
        p.map(f_temp_func, np.arange(num_batches))
    
    t5=time.time()
    
    ### Concatenate temp files
    t6=time.time()
    img,label,idx=f_concat_temp_files(num_batches,save_location)
    t7=time.time()
    print("Time for concatenation of file:",t7-t6)
    
    ### Save concatenated files
    f_save_files(idx,img,label,'full_',save_location)
    t8=time.time()
    
    print("total time",t8-t1)
    
    ######################################################
    ######################################################
    ### Check if the concatenated index arrays are identical
    ## this is a minor concatenation check. The full check is performed in a jupyter notebook.
    a1=np.load(save_location+'initial_idx_arr.npy')
    a2=np.load(save_location+'full_idx.npy')

    assert np.array_equal(a1,a2),"The index arrays after concatenation are not identical"
    print("ID arrays identical? ",np.array_equal(a1,a2))
