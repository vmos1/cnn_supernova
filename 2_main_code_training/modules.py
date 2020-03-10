### Modules used for training 

import numpy as np
import os
from tensorflow.keras.models import load_model
import pickle
import matplotlib.pyplot as plt
import yaml


from sklearn.metrics import roc_curve, auc, roc_auc_score
from models import f_define_model
from tensorflow.keras import callbacks


def f_load_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config



def f_train_model(model,inpx,inpy,model_weights,num_epochs=5,batch_size=64,val_x=None,val_y=None):
    '''
    Train model. Returns just history.history
    '''
    if val_x is None or val_y is None: 
        cv_fraction=0.33 # Fraction of data for cross validation
        print("Using {0} % of training data as validation".format(cv_fraction))
    
#     callbacks_list=[]
    callbacks_lst=[callbacks.EarlyStopping(monitor='val_loss', patience=40, verbose=1)]
    callbacks_lst.append(callbacks.ModelCheckpoint(model_weights, save_best_only=True, monitor='val_loss', mode='min'))
    
    history=model.fit(x=inpx, y=inpy,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    verbose=1,
                    callbacks =callbacks_lst,
                    #validation_split=cv_fraction,
                    validation_data=(val_x,val_y),
                    shuffle=True)
    
    print("Number of parameters",model.count_params())
    
    return history.history


def f_test_model(model,xdata,ydata):
    '''
    Test model and return array with predictions
    '''
    
#     model.evaluate(xdata,ydata,sample_weights=wts,verbose=1)
    y_pred=model.predict(xdata,verbose=1)
    ### Ensure prediction has the same size as labelled data.
    assert(ydata.shape[0]==y_pred.shape[0]),"Data %s and prediction arrays %s are not of the same size"%(test_y.shape,y_pred.shape)
       
    ##Condition for the case when the prediction is a 2column array 
    ## This complicated condition is needed since the array has shape (n,1) when created, but share (n,) when read from file.
    if (len(y_pred.shape)==2 and y_pred.shape[1]==2) : y_pred=y_pred[:,1]

    return y_pred

