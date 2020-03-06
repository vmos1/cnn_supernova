#!/bin/bash
#################
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --qos=regular
#SBATCH --job-name=supernova_extract_data_haswell
#SBATCH --output=slurm-%x-%j.out
#SBATCH --constraint=haswell
#SBATCH --account=nstaff
#################

echo "--start date" `date` `date +%s`
export OMP_NUM_THREADS=32
python 4c_extract_to_npy.py -b 1000 -c 32
echo "--end date" `date` `date +%s`
