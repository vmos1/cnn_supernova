#!/bin/bash
#################
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --qos=regular
#SBATCH --job-name=supernova_extract_data_knl
#SBATCH --output=slurm-%x-%j.out
#SBATCH --constraint=knl
#SBATCH --account=m3363
#################

echo "--start date" `date` `date +%s`
conda activate v3
export OMP_NUM_THREADS=1
srun -n 1 -c 136 python 4_extract_to_npy.py -b 100 -c 136
conda deactivate
echo "--end date" `date` `date +%s`
