# Description of data extraction
Feb 26, 2020


Steps:
1. From the website https://portal.nersc.gov/project/dessn/autoscan/
    a. Download the data using the file 0_download_data.py
    b. Download the file autoscan_features.3.csv 
2. Use 1_preprocessing.py to write gathered file paths as a csv (gathered_filepaths.csv)
3. Add the labels and write it to a file (summary_label_files.csv)
4. Check that the above extraction is correct using the notebook 3_Data_check_1.ipynb
5. Extract the data using 4_extract_to_npy.py which goes through the dataframe and writes each .gif to .npy files in the folder input_npy_files.
    a)Use 4_extract_to_npy.py 
    b) This process is very slow, so it has been parallelized to use use mutliple cores. Temporary files are created and these are merged to get the big files.
    c) There are batch files 'batch_cori_haswell.sh' and 'batch_cori_knl.sh' to run this on compute nodes. 
    d) The time taken is roughly <> hours on haswell and knl.
6. Check this extraction process using 5_Data_check_2.ipynb 

