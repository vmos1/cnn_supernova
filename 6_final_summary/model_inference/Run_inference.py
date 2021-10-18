# Code to Run inference on stored CNN model
## Author: Venkitesh Ayyar (vpa@lbl.gov)



import numpy as np
import argparse
import os

## M-L modules
# import tensorflow.keras
# import tensorflow as tf
from tensorflow.keras.models import load_model



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train and test CNN for Supernova data")
    add_arg = parser.add_argument
    
    add_arg('--model','-m', type=str, default='models/model_1.h5',help='The name of stored model file.')
    add_arg('--input','-i', type=str, default='data/',help='The name of input image file. Format: (n_samples, 51, 51, 3)')
    add_arg('--output','-o', type=str, default='results/',help='Folder to place the inference results')

    return parser.parse_args()





if __name__=="__main__":
    
    args=parse_args()
    print(args)
    ip_file=args.input
    stored_model=args.model
    results_dir=args.output
    
    print("Reading model file",stored_model)
    print("Reading input data from",ip_file)
    
    ## Store data in numpy array
    ip_images=np.load(ip_file)
    
    ## Check if file exist
    assert os.path.exists(stored_model),"Model not saved: %s"%(stored_model)
    ## Load model from file
    Model=load_model(stored_model)
    
    #################################
    ### Model Inference ###
    y_pred=Model.predict(ip_images,verbose=1)
    
    ## Save prediction array
    op_file=results_dir+'y_pred.txt'
    np.savetxt(op_file,y_pred)

    print("Results saved in %s"%(op_file))