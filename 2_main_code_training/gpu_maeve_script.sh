
echo "--start date" `date` `date +%s`
echo '--hostname ' $HOSTNAME
export HDF5_USE_FILE_LOCKING=FALSE
# Limit to one GPU
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$1
### Actual script to run
python main.py --train --config config_maeve.yaml --gpu maeve --model_list $2
echo "--end date" `date` `date +%s`
