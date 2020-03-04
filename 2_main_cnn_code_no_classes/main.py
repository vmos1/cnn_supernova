# The main code for training a CNN on the ATLAS SUSY 2D images
# Created by Venkitesh Ayyar. August 19, 2019

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
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
from modules import *

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train and test CNN for ATLAS SUSY data", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arg = parser.add_argument
    
    add_arg('--config','-c', type=str, default='config.yaml',help='The .yaml file that stores the configuration.')
    add_arg('--train','-tr',  action='store_true' ,dest='train' ,help='Has the model been trained?')
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--gpu', type=str, choices=['None','maeve','cori'],default='None', help='Whether using gpu, if so, maeve or cori.')    
    add_arg('--model_list', '-mdlst', nargs='+', type=int, dest='mod_lst',help=' Enter the list of model numbers to test ', required=True)

    return parser.parse_args()


if __name__=='__main__':

    args=parse_args()
    print(args)
    ## Note: --train means models needs to be trained. hence train_status=False
    model_lst=args.mod_lst
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
    
    ### Extract data ###
    
    data_dir=config_dict['data']['data_dir']
    #prefix=config_dict['data']['prefix']
    #### Training data
    data_dict=f_get_data(data_dir)
    
    size_data=data_dict['labels'].shape[0]
    ### Define the indices for training, validation and test data
    train_idx=int(0.4*size_data)
    val_idx=train_idx+int(0.05*size_data)
    test_idx=val_idx+int(0.05*size_data)

    
    train_x,train_y=data_dict['images'][:train_idx],data_dict['labels'][:train_idx]
    val_x,val_y=data_dict['images'][train_idx:val_idx],data_dict['labels'][train_idx:val_idx]

    print("Train data shape",train_x.shape,train_y.shape)
    print("Validation data shape",val_x.shape,val_y.shape)

    t2=time.time()
    print("Time taken to read files",t2-t1)
    
    model_save_dir=config_dict['output_dir']
    
    
    
    for i in model_lst:
        model_name=str(i)

        ### Compile model ###
        # Declare names of files for storing model, model weights, history
        fname_model,fname_model_wts,fname_history='model_{0}.h5'.format(model_name),'model_wts_{0}.h5'.format(model_name),'history_{0}.pickle'.format(model_name)

        ### Define model 
        model=f_define_model(config_dict,name=model_name)
        
        if args.train: # If model hasn't been trained, train and save files
            
            ### Train model ###
            history=f_train_model(model,train_x,train_y,model_weights=model_save_dir+fname_model_wts,num_epochs=num_epochs,batch_size=batch_size,val_x=val_x,val_y=val_y)
 
            ### Save model and history ###
            fname_model,fname_history='model_{0}.h5'.format(model_name),'history_{0}.pickle'.format(model_name)

            model.save(model_save_dir+fname_model)
            with open(model_save_dir+fname_history, 'wb') as f:
                pickle.dump(history, f)
         
        else: # If using pre-trained model, check if files exist and load them.
            print("Using trained model")
            ### Read model and history

            ## Check if files exist
            assert os.path.exists(model_save_dir+fname_model),"Model not saved: %s"%(model_save_dir+fname_model)
            assert os.path.exists(model_save_dir+fname_history),"History not saved"

            model=load_model(model_save_dir+fname_model)
            with open(model_save_dir+fname_history,'rb') as f:
                history= pickle.load(f)

        #################################
        ### Test model ###

        ## Read Test_data
        test_x,test_y=data_dict['images'][val_idx:test_idx],data_dict['labels'][val_idx:test_idx]
        print("Test data shape",test_x.shape,test_y.shape)
        
        y_pred=f_test_model(model,test_x,test_y)   ### Prediction using model

        ## Save the predictions on test data for the labels, for roc curve
        fname_ypred='ypred_{0}.test'.format(model_name)
        np.savetxt(model_save_dir+fname_ypred,y_pred)
        
        ## Save the test data labels for roc curve 
        ### This is just the test data, but it is useful to save it, to make the analysis part simpler
        fname_ytest='ytest_{0}.test'.format(model_name)
        np.savetxt(model_save_dir+fname_ytest,test_y)
        

