Jan 24, 2020
# Introduction
This repository contains codes used to train 2D layered CNNs for performing classification on supernova data (Artifact vs non-artifact).

Tools for use for other datasets:
1. Data visualization
2. Training CNN
3. Viewing model results: learning curves and predictions

# Dataset
The data can be found https://portal.nersc.gov/project/dessn/autoscan/
For further information refer: https://arxiv.org/abs/1504.02936v3

# Repository information
The Table below describes the content of various folders in this repository

| Description | Location |
| --- | ---|
| Data pre-processing | [0_pre_processing](https://github.com/vmos1/cnn_supernova/tree/master/0_pre_processing) | 
| Main code | [2_main_code_training](https://github.com/vmos1/cnn_supernova/tree/master/2_main_code_training) |
| Analysis code | [3_analyze_results_jpt_notebooks](https://github.com/vmos1/cnn_supernova/tree/master/3_analyze_results_jpt_notebooks) |
| Code for inference | [6_final_summary](https://github.com/vmos1/cnn_supernova/tree/master/6_final_summary) |

# Running Inference

To run inference on the best CNN model, use the code in the folder `6_final_summary_Model_inference.ipynb`
The best CNN model along with sample data is stored at https://portal.nersc.gov/project/m3363/vayyar_des_cnn/

# Conda environment
The best way to run these codes is using conda environment. To build the conda environment use the files in ther folder
`conda_env/`. The file `conda_env/environment.yml` can be used to build the environment. The file `conda_env/additional.txt` contains additional packages that need to be manually installed inside the conda environment.


