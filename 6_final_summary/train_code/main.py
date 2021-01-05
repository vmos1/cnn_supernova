# The main code for training a CNN on the DES Supernova images.
### Created by Venkitesh Ayyar (vpa@lbl.gov)
### March 7, 2020 

import numpy as np
#import pandas as pd
import time
import argparse
import sys
import subprocess as sp
import pickle
import yaml

## M-L modules
import tensorflow.keras
from tensorflow.keras import layers, models, optimizers, callbacks  # or tensorflow.keras as keras
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc, roc_auc_score
from tensorflow.keras.models import load_model

## modules from other files
from models import *
from utils import dataset, cnn_model, f_get_data, f_load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train and test CNN for Supernova data")
    add_arg = parser.add_argument
    
    add_arg('--config','-c', type=str, default='config.yaml',help='The .yaml file that stores the configuration.')
    add_arg('--train','-tr',  action='store_true' ,dest='train' ,help='Has the model been trained?')
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--gpu', type=str, choices=['None','maeve','cori'], default='None', help='Whether using gpu, if so, maeve or cori.')    
    add_arg('--model_list', '-mdlst', nargs='+', type=int, dest='mod_lst',help=' Enter the list of model numbers to test ', required=True)

    return parser.parse_args()


if __name__=='__main__':
    
    args=parse_args()
    print(args)
    ## Note: --train means models needs to be trained. hence train_status=False
    model_lst=args.mod_lst
    pre_norm=False
    if pre_norm : print("Prenormalization",pre_norm)
    
    ##### Stuff for GPU #####
    if args.gpu!='None': 
        script_loc={'maeve':'/home/vpa/standard_scripts/','cori':'/global/u1/v/vpa/standard_scripts/'}
        ## special imports for setting keras and tensorflow variables.
        sys.path.insert(0,script_loc[args.gpu])
        from keras_tf_parallel_variables import configure_session
        ### Set tensorflow and keras variables
        configure_session(intra_threads=32, inter_threads=2, blocktime=1, affinity='granularity=fine,compact,1,0')
    
    t1=time.time()
    
    ### Read configuration ###
    config_file=args.config
    config_dict=f_load_config(config_file)
    print(config_dict)
    
    batch_size=config_dict['training']['batch_size']
    num_epochs=config_dict['training']['n_epochs']
    model_save_dir=config_dict['output_dir']
    
    ### Extract data ###
    data_dir=config_dict['data']['data_dir']
    prefix=config_dict['data']['prefix']
    #### Read data from files 
    data_dict=f_get_data(prefix,data_dir,pre_norm=pre_norm)
    
    size_data=data_dict['labels'].shape[0]
    print("Size of entire dataset is : ",size_data)
    #### Define the indices for training, validation and test data
    train_size,val_size,test_size=int(0.7*size_data),int(0.1*size_data),int(0.2*size_data)
    
    ### Get random indices for test,train,val
#     np.random.seed(225) # Set random seed
    np.random.seed(737581) # Set random seed
    test_idx=np.random.choice(np.arange(size_data),test_size,replace=False)
    # get remaining indices without test indices
    rem_idx1=np.array(list(set(np.arange(size_data))-set(test_idx)))
    val_idx=np.random.choice(rem_idx1,val_size,replace=False)
    rem_idx2=np.array(list(set(rem_idx1)-set(val_idx)))
    train_idx=np.random.choice(rem_idx2,train_size,replace=False)
    
    print("Shapes of indices",train_idx.shape,test_idx.shape,val_idx.shape)
    
    #### Storing arrays into train,validation, test objects and deleting the full data dictionary
    train_data=dataset('training',data_dict,train_idx)
    val_data=dataset('validation',data_dict,val_idx)
    test_data=dataset('test',data_dict,test_idx)
    del data_dict

    print("\nData shapes: Train {0}, Validation {1}, Test {2}\n".format(train_data.x.shape,val_data.x.shape,test_data.x.shape))
    
    t2=time.time()
    print("Time taken to read and process input files",t2-t1)
     
    #### ML part ####
    
    for i in model_lst:
        model_name=str(i)
        
        ### Define Object for cnn_model
        Model=cnn_model(model_name,model_save_dir)
        
        ### Define the keras ML model and store in the object
        model=f_define_model(config_dict,name=model_name)
        Model.f_build_model(model)
        
        if args.train: # If model hasn't been trained, train and save files
            ### Train model ###
            Model.f_train_model(train_data,val_data,num_epochs=num_epochs,batch_size=batch_size)
            
            ### Save model and history ###
            Model.f_save_model_history()
        
        else: # If using pre-trained model, check if files exist and load them.
            print("Using trained model")
            ### Read stored model and history
            Model.f_load_model_history()

        #################################
        ### Test model ###
        Model.f_test_model(test_data)
        
        ### Save prediction array and labels of test,train and val data
        Model.f_save_predictions(test_data,train_data,val_data)
        