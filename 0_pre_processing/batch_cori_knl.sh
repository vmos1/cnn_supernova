#!/bin/bash
#################
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --qos=regular
#SBATCH --job-name=supernova_extract_data_knl
#SBATCH --output=slurm-%x-%j.out
#SBATCH --constraint=knl
#SBATCH --account=nstaff
#################

echo "--start date" `date` `date +%s`
export OMP_NUM_THREADS=1
srun -n 1 -c 136 python 4_extract_to_npy.py -b 100 -c 136
echo "--end date" `date` `date +%s`
