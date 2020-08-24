# Description of data extraction
Feb 26, 2020


Steps:
1. **Download data**
From the website https://portal.nersc.gov/project/dessn/autoscan/

    a. Download the raw data (.gif files) using the file `0_download_data.py`
    
    b. Download the file `autoscan_features.3.csv` which contains the summary information. 
    
2. **Gather file paths** Use `1_preprocessing.py` to write gathered file paths as a csv (gathered_filepaths.csv)
3. **Add labels** Add the labels and write it to a file (summary_label_files.csv)
4. **Check extraction 1** Check that the above extraction is correct using the notebook `3_Data_check_1.ipynb`
5. **Extract images to .npy file** Extract the data using `4_extract_to_npy.py` which goes through the dataframe and writes each .gif to .npy files in the folder input_npy_files.

    * Use `4_extract_to_npy.py` 
    
    * This process is very slow, so it has been parallelized to use mutliple cores. Temporary files are created and these are merged to get the big files.
    
    * There are batch files `batch_cori_haswell.sh` and `batch_cori_knl.sh` to run this on compute nodes. 
    
    * The time taken is roughly 2 hours on haswell after parallelization.
    
6. **Check extraction 2** Check this extraction process using `5_Data_check_2.ipynb` 
7. **Apply normalization** If required, apply the normalization (MAD normalization) to all samples using the file `6_pre_norm.py`. This creates a new .npy file.
