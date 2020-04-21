# Code to pre-normalize the image using the MAD method (Median absolute deviation)
# April 17, 2020
import numpy as np
import time


save_location='/global/project/projectdirs/dasrepo/vpa/supernova_cnn/data/gathered_data/input_npy_files/'
f1='full_x.npy'
f2='prenorm_full_x.npy'

### Read data from .npy file
ip_arr=np.load(save_location+f1)
print(ip_arr.shape)

def f_rescale_samples(samples):
    ''' Rescale individual images with MAD value of diff image
    '''
    def f_mad(arr):
        '''
        Compute MAD and std
        '''
        arr2=arr.flatten()
        MD=np.median(arr2)
    #     print(MD)
        mad=np.median(np.abs(arr2-MD))
        k=1.4826 ### For normal distribution
        sigma=mad*k

        return mad,sigma
    
    
    scaled_samples=np.ones_like(samples)
    lst_zeros=[] # List to store indices where the MAD-sigma value is zero
    for i,row in enumerate(samples):
        scale=f_mad(row[:,:,2])[1]
        if scale<1e-10:### Not applying scale for zero sigma
            #print("Small value",i,scale)
            lst_zeros.append(i)
            scale=1.0
        scaled_samples[i]=row*(1.0/scale)
    
    ### For every row, compute the MAD value for diff image (idx =2 ) and multiple its inverse to each sample
#     scaled_samples=np.array([(1.0/f_mad(i[:,:,2])[1]+1e-6)*i for i in samples])
    
    return scaled_samples,lst_zeros


t1=time.time()
rescaled_arr,zero_lst=f_rescale_samples(ip_arr)
t2=time.time()
print(t2-t1)
print('Number of zero median images',len(zero_lst))

print(rescaled_arr.shape)
### Save pre-normalized images
np.save(save_location+f2,rescaled_arr)


