# Code to find and write the location of gifs and Ids taken from the file name
### Feb 11,2020
### author: Venkitesh. vpa@lbl.gov

import numpy as np
import pandas as pd
import subprocess as sp
import time
import os


#####################
### Define locations
data_dir='/global/project/projectdirs/dasrepo/vpa/supernova_cnn/data/raw_data/'

fname='/global/project/projectdirs/dasrepo/vpa/supernova_cnn/data/gathered_data/gathered_filepaths_fits.csv'
### Write column names to file 
with open (fname,'w') as f: f.write('ID,filename,file path\n')


#####################
### Use os.walk() to get full path for every .gif or .fits file in the folder (recursive)
dir_name=data_dir
#dir_name='/global/project/projectdirs/dasrepo/vpa/supernova_cnn/data/raw_data/home2/SNWG/Archive/2013/Y1/20131218/'
t1=time.time()

# file_type='.gif'
file_type='.fits'

file_list=[]
for count,(root, dirs,files) in enumerate(os.walk(dir_name)):
    for file in files:
        if file.endswith(file_type):
            id=int(file.split(file_type)[0][4:])    # Extract the ID from the file name
            file_path=root+os.sep+file    # Extract full path
            file_list.append([id,file,file_path])
    #print("Count",count)
t2=time.time()
print("Time taken",t2-t1)

#######################
### Write data to a csv
### The code is fast enough, so there is not need to write to file in blocks
with open (fname,'a') as f:
    for item in file_list:
        f.write("%s,%s,%s\n"%(item[0],item[1],item[2]))