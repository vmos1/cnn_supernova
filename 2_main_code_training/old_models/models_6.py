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

    if name=='3': # Simple layered, with inner dropout
        model_par_dict={'conv_size_list':[80,80,80],'kernel_size':(3,3), 'no_pool':False,'pool_size':(2,2), 'strides':1, 'learn_rate':0.00002,
                        'inner_dropout':None, 'outer_dropout':0.3,'dense_size':51,'final_activation':'sigmoid','double_conv':False}
        
    if name=='8': # Simple layered, with inner dropout
        model_par_dict={'conv_size_list':[80,80],'kernel_size':(4,4), 'no_pool':False,'pool_size':(3,3), 'strides':1, 'learn_rate':0.00002,
                        'inner_dropout':None, 'outer_dropout':0.3,'dense_size':51,'final_activation':'sigmoid','double_conv':True}
   
    if name=='16': # Striding single conv
        model_par_dict={'conv_size_list':[40,60,80],'kernel_size':(6,6), 'no_pool':True,'pool_size':(2,2), 'strides':[2,2,1], 'learn_rate':0.00002,
                        'inner_dropout':0.1, 'outer_dropout':0.3,'dense_size':51,'final_activation':'sigmoid','double_conv':False}        
     
    if name=='17': # Simple layered, with inner dropout
        model_par_dict={'conv_size_list':[400,400,400,400],'kernel_size':(4,4), 'no_pool':False,'pool_size':(2,2), 'strides':1, 'learn_rate':0.000002,
                        'inner_dropout':None, 'outer_dropout':0.5,'dense_size':40,'final_activation':'sigmoid','double_conv':False}
     
    if name=='18': # Simple layered, with inner dropout
        model_par_dict={'conv_size_list':[160,200,240,320],'kernel_size':(4,4), 'no_pool':False,'pool_size':(2,2), 'strides':1, 'learn_rate':0.000002,
                        'inner_dropout':None, 'outer_dropout':0.3,'dense_size':10,'final_activation':'sigmoid','double_conv':False}
        
    if name=='19': # Simple layered, with inner dropout
        model_par_dict={'conv_size_list':[160,320],'kernel_size':(4,4), 'no_pool':False,'pool_size':(3,3), 'strides':1, 'learn_rate':0.000002,
                        'inner_dropout':None, 'outer_dropout':0.3,'dense_size':10,'final_activation':'sigmoid','double_conv':True}
    
    if name=='20': # Simple layered, with inner dropout
        model_par_dict={'conv_size_list':[200,320],'kernel_size':(4,4), 'no_pool':False,'pool_size':(3,3), 'strides':1, 'learn_rate':0.000002,
                        'inner_dropout':None, 'outer_dropout':0.3,'dense_size':40,'final_activation':'sigmoid','double_conv':True}
       
    elif name=='0':
        custom_model=True
        learn_rate=0.001 
        
        inputs = layers.Input(shape=shape)
        h = inputs
        # Convolutional layers
        h = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(h)
        h = Conv2D(128, kernel_size=(3, 3), activation='relu', strides=2, padding='same')(h)
        h = Conv2D(256, kernel_size=(3, 3), activation='relu', strides=1, padding='same')(h)
        h = Conv2D(256, kernel_size=(3, 3), activation='relu', strides=2, padding='same')(h)
        h = Flatten()(h)
        h = Dense(512, activation='relu')(h)
        
        # Ouptut layer
        outputs = layers.Dense(1, activation='sigmoid')(h)
        
    elif name=='30':
        custom_model=True
        learn_rate=0.000005
        
        inputs = layers.Input(shape=shape)
        h = inputs        
        h = layers.Conv2D(80,kernel_size=(4, 4), strides=1, activation='relu', padding='same')(h)
        h = layers.BatchNormalization(epsilon=1e-5, momentum=0.99)(h)
        h = layers.Conv2D(160,kernel_size=(4, 4), strides=1, activation='relu', padding='same')(h)
        h = layers.BatchNormalization(epsilon=1e-5, momentum=0.99)(h)
        h = layers.MaxPooling2D(pool_size=(3,3))(h)
        h = layers.Conv2D(240,kernel_size=(4, 4), strides=1, activation='relu', padding='same')(h)
        h = layers.BatchNormalization(epsilon=1e-5, momentum=0.99)(h)
        h = layers.Conv2D(320,kernel_size=(4, 4), strides=1, activation='relu', padding='same')(h)
        h = layers.BatchNormalization(epsilon=1e-5, momentum=0.99)(h)
        h = layers.MaxPooling2D(pool_size=(3,3))(h)
        
        h = layers.Flatten()(h)
        h = layers.Dropout(rate=0.3)(h)
        h = layers.Dense(51, activation='relu')(h)
        h = layers.BatchNormalization(epsilon=1e-5, momentum=0.99)(h)

        # Ouptut layer
        outputs = layers.Dense(1, activation='sigmoid')(h)    
    
    elif name=='31':
        custom_model=True
        learn_rate=0.000002
        inputs = layers.Input(shape=shape)
        h = inputs
        # Convolutional layers     
        conv_sizes=[200,200]
        conv_args = dict(kernel_size=(4, 4), activation='relu', padding='same')
        for conv_size in conv_sizes:
            h = layers.Conv2D(conv_size, **conv_args)(h)
            h = layers.BatchNormalization(epsilon=1e-5, momentum=0.99)(h)
            h = layers.Conv2D(conv_size, **conv_args)(h)
            h = layers.BatchNormalization(epsilon=1e-5, momentum=0.99)(h)
            h = layers.MaxPooling2D(pool_size=(3, 3))(h)
        
        h = layers.Flatten()(h)
        h = layers.Dense(51, activation='relu')(h)
        h = layers.BatchNormalization(epsilon=1e-5, momentum=0.99)(h)
        h = layers.Dropout(rate=0.5)(h)
        outputs = layers.Dense(1, activation='sigmoid')(h)
        
    elif name=='100': # Resnet 50
        inputs = layers.Input(shape=shape)
        model = ResNet50(img_input=inputs)
        learn_rate=0.0005
        resnet=True
    
    elif name=='101': # Resnet 18
        inputs = layers.Input(shape=shape)
        model = ResNet18(img_input=inputs)
        learn_rate=0.0005
        resnet=True
    
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

