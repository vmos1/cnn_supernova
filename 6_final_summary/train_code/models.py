## modify this script to change the model.
### Add models with a new index.

from tensorflow.keras import layers, models, optimizers, callbacks  # or tensorflow.keras as keras
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K

### Import the modules for resnet50
from resnet50 import *
from resnet18 import *

### Defining all the models tried in the study


def f_model_prototype(shape,**model_dict):
    '''
    General prototype for layered CNNs
    
    Structure:
    For different conv layers:
        - Conv 1
        - Conv 2
        - Pooling
        - Dropout
    - Flatten
    - Dropout 
    - Dense 
    - Dense -> 1
    
    '''
   
    activ='relu' # activation
    inputs = layers.Input(shape=shape)
    h = inputs
    # Convolutional layers
    conv_sizes=model_dict['conv_size_list'] # Eg. [10,10,10]
    
    ### Striding
    if model_dict['strides']==1: 
        stride_lst=[1]*len(conv_sizes) # Default stride is 1 for each convolution.
    else : 
        stride_lst=model_dict['strides']
    
    conv_args = dict(kernel_size=model_dict['kernel_size'], activation=activ, padding='same')
    
    for conv_size,strd in zip(conv_sizes,stride_lst):
        h = layers.Conv2D(conv_size, strides=strd, **conv_args)(h)
        h = layers.BatchNormalization(epsilon=1e-5, momentum=0.99)(h)
        
        if model_dict['double_conv']: 
            h = layers.Conv2D(conv_size,strides=strd, **conv_args)(h)
            h = layers.BatchNormalization(epsilon=1e-5, momentum=0.99)(h)
        
        if not model_dict['no_pool']: h = layers.MaxPooling2D(pool_size=model_dict['pool_size'])(h)
        ## inner_dropout is None or a float
        if model_dict['inner_dropout']!=None: h = layers.Dropout(rate=model_dict['inner_dropout'])(h)
    h = layers.Flatten()(h)
    
    # Fully connected  layers
    if model_dict['outer_dropout']!=None: h = layers.Dropout(rate=model_dict['outer_dropout'])(h)
    
    h = layers.Dense(model_dict['dense_size'], activation=activ)(h)
    h = layers.BatchNormalization(epsilon=1e-5, momentum=0.99)(h)
    
    # Ouptut layer
    outputs = layers.Dense(1, activation=model_dict['final_activation'])(h)    
    return outputs,inputs
    
def f_define_model(config_dict,name='1'):
    '''
    Function that defines the model and compiles it. 
    Reads in a dictionary with parameters for CNN model prototype and returns a keral model
    '''
    ### Extract info from the config_dict
    shape=config_dict['model']['input_shape']
    loss_fn=config_dict['training']['loss']
    metrics=config_dict['training']['metrics']
    
    resnet=False ### Variable storing whether the models is resnet or not. This is needed for specifying the loss function.    
    custom_model=False ### Variable storing whether the models is a layer-by-layer build code (not using the protytype function).    
    
    # Choose model
    if name=='1': # Simple layered, with inner dropout
        model_par_dict={'conv_size_list':[80,80,80],'kernel_size':(3,3), 'no_pool':False,'pool_size':(2,2), 'strides':1, 'learn_rate':0.00002,
                        'inner_dropout':None, 'outer_dropout':0.3,'dense_size':51,'final_activation':'sigmoid','double_conv':False}
    if name=='2': # Simple layered, with inner dropout
        model_par_dict={'conv_size_list':[80,80],'kernel_size':(4,4), 'no_pool':False,'pool_size':(3,3), 'strides':1, 'learn_rate':0.00002,
                        'inner_dropout':None, 'outer_dropout':0.3,'dense_size':51,'final_activation':'sigmoid','double_conv':True}
    if name=='3': # Simple layered, with inner dropout
        model_par_dict={'conv_size_list':[120,120],'kernel_size':(4,4), 'no_pool':False,'pool_size':(3,3), 'strides':1, 'learn_rate':0.00002,
                        'inner_dropout':None, 'outer_dropout':0.3,'dense_size':51,'final_activation':'sigmoid','double_conv':True}   
    if name=='4': # Striding single conv
        model_par_dict={'conv_size_list':[40,60,80],'kernel_size':(6,6), 'no_pool':True,'pool_size':(2,2), 'strides':[2,2,1], 'learn_rate':0.00002,
                        'inner_dropout':0.1, 'outer_dropout':0.3,'dense_size':51,'final_activation':'sigmoid','double_conv':False}
    
    ############################################
    ### Add more models above
    ############################################
    ####### Compile model ######################
    ############################################

    if resnet:
        print("resnet model name",name)
        opt,loss_fn=optimizers.Adam(lr=learn_rate),'sparse_categorical_crossentropy'
    
    else : ## For non resnet models 
        if not custom_model:  ### For non-custom models, use prototype function
            outputs,inputs=f_model_prototype(shape,**model_par_dict)
            learn_rate=model_par_dict['learn_rate']    
        model = models.Model(inputs, outputs)
        opt=optimizers.Adam(lr=learn_rate)
    
    model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)
    #print("model %s"%name)

    return model

