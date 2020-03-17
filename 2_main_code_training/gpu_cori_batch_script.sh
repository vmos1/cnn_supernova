#!/bin/bash
#################
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --qos=regular
#SBATCH --job-name=supernova_cnn_train
#SBATCH --output=slurm-%x-%j.out
#SBATCH --constraint=gpu
#SBATCH --account=nstaff
#SBATCH --gres=gpu:1
#################


echo "--start date" `date` `date +%s`
echo '--hostname ' $HOSTNAME
export HDF5_USE_FILE_LOCKING=FALSE
module unload esslurm
module load tensorflow/gpu-1.15.0-py37
#module load python 
#conda activate v_py3
### Actual script to run
python main.py --train --config config_cori.yaml --gpu cori --model_list $1
### To test existing model
#python main.py --config config_cori.yaml --gpu cori --model_list $1
#conda deactivate v_py3
echo "--end date" `date` `date +%s`
