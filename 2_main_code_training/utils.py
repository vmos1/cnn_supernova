### Code containing classes used in the main code: train_cnn.py
### Author: Venkitesh Ayyar (vpa@lbl.gov)
### March 7, 2020

import numpy as np

def f_get_data(prefix,data_dir):
    '''
    Function to get data from .npy files into images, labels and IDs.
    '''
    try:
        images=np.load(data_dir+prefix+'_x.npy')
        labels=np.load(data_dir+prefix+'_y.npy')
        ids=np.load(data_dir+prefix+'_idx.npy')
    except Exception as e:
        print("Encountered exception",e)
        raise SystemExit

    keys=['images','labels','ids']
    values_dict=dict(zip(keys,[images,labels,ids]))
    
    return values_dict

class dataset:
    '''
    Class to store datasets. Example objects: train_data,val_data,test_data
    
    '''
    ## Eg: dataset('train',data_dir)
    def __init__(self,name,data_dict,start_idx,end_idx,):
        self.name=name

        self.x,self.y,self.id=data_dict['images'][start_idx:end_idx],data_dict['labels'][start_idx:end_idx],data_dict['labels'][start_idx:end_idx]

        
        
class cnn_model:
    '''
    Class to store features of cnn model such as : model_name, wts_filename, history_filename,
    '''
    def __init(self,name,wts,hist):
        self.name=name
        self.wts_fname=wts
        self.history_fname=hist
    
        self.cnn_model=f_define_model()

        

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