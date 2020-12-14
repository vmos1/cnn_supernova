# Description

This folder contains files to run inference on the best CNN models.
The notebook `Model_inference.ipynb` performs the model inference.

## Input data
The input data needs to be in the following form: 
- Three `.npy` files of the form '<prefix>_idx.npy','<prefix>_x.npy', '<prefix>_y.npy' 
corresponding to sample indices, sample images and sample labels.
For example, 'full_idx.npy','full_x.npy', 'full_y.npy'.
These need to be specified in the code using `data_dir` and `prefix'.

The code requires a saved model file of the type `model1.h5` and a history file `history_1.pickle`

## Results 
Inference results are stored in the folder `results_dir`

The results contain 3 files:
1. `id_test_<model_number>.test` : The labels of the original samples.
2. `ypred_<model_number.test>`: The predictions for each sample.
3. `ytest_<model_number.test>`: The original sample predictions (for easy comparison).



