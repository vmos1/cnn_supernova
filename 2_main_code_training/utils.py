### Code containing classes used in the main code: train_cnn.py
### Author: Venkitesh Ayyar (vpa@lbl.gov)
### March 7, 2020

import numpy as np
import yaml
from tensorflow.keras.models import load_model
from tensorflow.keras import callbacks
import pickle
import os


def f_load_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config

def f_get_data(prefix,data_dir,pre_norm):
    '''
    Function to get data from .npy files into images, labels and IDs.
    '''
    
    img_name=data_dir+'prenorm_%s_x.npy'%(prefix) if pre_norm else data_dir+prefix+'_x.npy'
    print("Input file name",img_name)
    try:
        images=np.load(img_name)
        labels=np.load(data_dir+prefix+'_y.npy')
        ids=np.load(data_dir+prefix+'_idx.npy')
    except Exception as e:
        print("Encountered exception",e)
        raise SystemExit

    keys=['images','labels','ids']        
    values_dict=dict(zip(keys,[images,labels,ids]))
    
    return values_dict


###############################
### Classes used ###
###############################
class dataset:
    '''
    Class to store datasets. Example objects: train_data,val_data,test_data
    3 arguments : 
    - Name of dataset
    - dictionary containing x,y,ids
    - Index array to select specific rows
    '''
    
    ## Eg: dataset('train',data_dir)
    def __init__(self,name,data_dict,idx_arr):
        self.name=name

        self.x,self.y,self.id=data_dict['images'][idx_arr],data_dict['labels'][idx_arr],data_dict['ids'][idx_arr]


class cnn_model:
    '''
    Class to store features of cnn model such as : model_name, wts_filename, history_filename,
    '''
    
    def __init__(self,model_name,model_save_dir):
        
        ### Initialization ###
        self.name=model_name
        #Declare names of files for storing model, model weights, history
        self.fname_model=model_save_dir+'model_{0}.h5'.format(model_name)
        self.fname_model_wts=model_save_dir+'model_wts_{0}.h5'.format(model_name)
        self.fname_history=model_save_dir+'history_{0}.pickle'.format(model_name)
        self.fname_ypred=model_save_dir+'ypred_{0}.test'.format(model_name)
        self.fname_id_test=model_save_dir+'id_test_{0}.test'.format(model_name)
        self.fname_id_train=model_save_dir+'id_train_{0}.train'.format(model_name)
        self.fname_id_val=model_save_dir+'id_val_{0}.val'.format(model_name)
        self.fname_ytest=model_save_dir+'ytest_{0}.test'.format(model_name)
        
    def f_build_model(self,model):
        '''Store model in the class member. Reads in a keras model   '''
        self.cnn_model=model
        

    
    def f_train_model(self,train_data,val_data,num_epochs=5,batch_size=64):
#         model,inpx,inpy,model_weights):
        '''
        Train model. Returns just history.history
        '''
    
        def f_learnrate_sch(epoch,lr):
            ''' Module to schedule the learn rate'''
            step=10 ### learn rate is constant up to here
            #if epoch>step: lr=lr*np.exp(-0.2*(epoch-10)) # Exponential decay after 10
            if (epoch>=step and epoch%step==0): lr=lr/2.0
             
            return lr 
    
        callbacks_lst=[]
        callbacks_lst.append(callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=1))
        callbacks_lst.append(callbacks.ModelCheckpoint(self.fname_model_wts, save_best_only=True, monitor='val_loss', mode='min'))
        callbacks_lst.append(callbacks.LearningRateScheduler(f_learnrate_sch,verbose=1))
         
        model=self.cnn_model
        history=model.fit(x=train_data.x, y=train_data.y,
                        batch_size=batch_size,
                        epochs=num_epochs,
                        verbose=1,
                        callbacks=callbacks_lst,
                        #validation_split=cv_fraction,
                        validation_data=(val_data.x,val_data.y),
                        shuffle=True)
        
        print("Number of parameters",model.count_params())

        self.history=history.history       

    
    def f_save_model_history(self):
        ''' Save model and history'''
        
        self.cnn_model.save(self.fname_model)
        with open(self.fname_history, 'wb') as f:
            pickle.dump(self.history, f)
    
    def f_load_model_history(self):
        ''' For pre-trained model, load model and history'''
        
        ## Check if files exist
        assert os.path.exists(self.fname_model),"Model not saved: %s"%(self.fname_model)
        assert os.path.exists(self.fname_history),"History not saved: %s"%(self.fname_history)
        
        ## Load data from files
        self.cnn_model=load_model(self.fname_model)
        with open(self.fname_history,'rb') as f:  history= pickle.load(f)        
    
    def f_test_model(self,test_data):
        '''
        Test model and return array with predictions
        '''
        model=self.cnn_model
#         model.evaluate(test_data.x,test_data.y,sample_weights=wts,verbose=1)
        print(test_data.x.shape)
        y_pred=model.predict(test_data.x,verbose=1)
        ### Ensure prediction has the same size as labelled data.
        
        assert(test_data.y.shape[0]==y_pred.shape[0]),"Data %s and prediction arrays %s are not of the same size"%(test_data.y.shape,y_pred.shape)
        
        ##Condition for the case when the prediction is a 2column array 
        ## This complicated condition is needed since the array has shape (n,1) when created, but share (n,) when read from file.
        if (len(y_pred.shape)==2 and y_pred.shape[1]==2) : y_pred=y_pred[:,1]
        
        ### Store predictions to the class object
        self.y_pred=y_pred
    
    def f_save_predictions(self,test_data,train_data,val_data):
        ''' Save predictions for test data and the actual test data labels
            Also save IDs of train and validation data'''
        
        ## Save the predictions on test data for the labels, for roc curve
        np.savetxt(self.fname_ypred,self.y_pred)
        
        ## Save the test data labels and IDs for roc curve 
        ### This is just the test data, but it is useful to save it, to make the analysis part simpler
        np.savetxt(self.fname_ytest,test_data.y)
        ### Save IDs of test, train and validation data for reverse analysis
        np.savetxt(self.fname_id_test,test_data.id)
        np.savetxt(self.fname_id_train,train_data.id)
        np.savetxt(self.fname_id_val,val_data.id)


class trained_model:
    '''
    Class to extract data of trained model
    variables: model,history, y_pred (predictions of labels), fpr, tpr, threshold, auc
    functions: f_read_stored_model, f_compute_preds
    Example objects :  (models numbers) '1', '2', etc.
    '''
    
    def __init__(self,model_name,model_save_dir):
        
        self.tpr,self.fpr,self.threshold,self.auc=[],[],[],None
        self.precision,self.recall,self.threshold2,self.fscore,self.auc2=[],[],[],[],None
        self.f_read_stored_model(model_name,model_save_dir)
        
    def f_read_stored_model(self,model_name,model_save_dir):
        '''
        Read model, history and predictions
        '''
        
        fname_model='model_{0}.h5'.format(model_name)
        fname_history='history_{0}.pickle'.format(model_name)

        # Load model and history
        self.model=load_model(model_save_dir+fname_model)
        
        with open(model_save_dir+fname_history,'rb') as f:
            self.history= pickle.load(f)
        
        # Load predictions
        fname_ypred=model_save_dir+'ypred_{0}.test'.format(model_name)
        self.y_pred=np.loadtxt(fname_ypred)

        # Load true labels
        fname_ytest=model_save_dir+'ytest_{0}.test'.format(model_name)
        self.y_test=np.loadtxt(fname_ytest)
    
    
    
    def f_compute_preds(self):
        '''
        Module to use model and compute 
        '''
        
        y_pred=self.y_pred
        test_y=self.y_test
#         print(test_x.shape,test_y.shape,y_pred.shape)

        ## roc curve
        self.fpr,self.tpr,self.threshold=roc_curve(test_y,y_pred)
        # AUC 
        self.auc= auc(self.fpr, self.tpr)
        
        # calculate precision-recall curve
        self.precision, self.recall, self.thresholds2 = precision_recall_curve(test_y, y_pred)
#         self.precision, self.recall, self.fscore, support = precision_recall_fscore_support(test_y, y_pred, sample_weight=test_wts)
        
        # AUC2
        self.auc2= auc(self.recall, self.precision)