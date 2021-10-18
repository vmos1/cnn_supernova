module load tensorflow

#python Run_inference.py -m saved_models/model_1.h5 -i sample_test_data/input_x.npy -o results_inference/
main_dir=/global/cfs/cdirs/dasrepo/vpa/supernova_cnn/data/results_data/results/final_summary_data_folder/
python Run_inference.py -m ${main_dir}saved_models/model_1.h5 -i ${main_dir}sample_test_data/temp_bigger_data/transpose_input_x.npy -o ${main_dir}results_inference/

