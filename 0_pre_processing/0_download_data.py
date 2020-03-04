# Code to copy data from https://portal.nersc.gov/project/dessn/autoscan/
# Venkitesh Ayyar, Jan 24, 2020

########################
import numpy as np
import subprocess as sp
import os

# General prototype of string
strg_prefix='http://portal.nersc.gov/project/dessn/autoscan/stamps/stamps_'
strg_suffix='.tar'


data_dir='/global/project/projectdirs/dasrepo/vpa/supernova_cnn/data/raw_data/'
files_list=[strg_prefix+str(i)+strg_suffix for i in np.arange(11)]

print("List of files",files_list) 
print("Are you sure you want to download all the file to the location:%s ? The total data size is over 60GB.\n Y or N ?"%(data_dir) )
proceed=input()

dry_run=False

if proceed=='Y':
    for count,fle in enumerate(files_list):
        print("Downloading file",fle)
        ### Download the file 
        cmd='wget -P {0} {1}'.format(data_dir,fle) 
        print(cmd)
        if not dry_run: sp.check_output(cmd,shell=True)
        ### Untar the file
        cmd='tar -xvf {0}stamps_{1}.tar'.format(data_dir,count)
        print(cmd)
        #if not dry_run: sp.check_output(cmd,shell=True)
    print("Download complete")

else:
    print("Did not download")
