import os
import sys
import pickle
import joblib
import pandas as pd
import numpy as np


if __name__ == '__main__':

    ## Features. The columns are to be given to the code in this order !!
    FEATS = ['gflux', 'diffsumrn', 'numnegrn', 'mag',
         'a_image', 'b_image', 'flags', 'mag_ref',
         'mag_ref_err', 'a_ref', 'b_ref', 'bandnum',
         'n2sig3', 'n3sig3', 'n2sig5', 'n3sig5',
         'n3sig5shift', 'n2sig5shift', 'n2sig3shift',
         'n3sig3shift', 'ellipticity', 'nn_dist_renorm',
         'maglim', 'min_distance_to_edge_in_new', 'ccdid',
         'snr', 'mag_from_limit', 'magdiff', 'spread_model',
         'spreaderr_model', 'flux_ratio', 'lacosmic',
         'gauss', 'scale', 'amp', 'colmeds', 'maskfrac',
         'l1']
    
    
    classifier_path = '/global/cfs/cdirs/dasrepo/vpa/supernova_cnn/data/autoscan_data/data/ml3.2.classifier/ml3.2.comp'
    imputer_path = '/global/cfs/cdirs/dasrepo/vpa/supernova_cnn/data/autoscan_data/data/imputerml3.2.obj'
    scaler_path = '/global/cfs/cdirs/dasrepo/vpa/supernova_cnn/data/autoscan_data/data/scalerml3.2.obj'
    
    clf = joblib.load(classifier_path)
    imputer = pickle.load(open(imputer_path, 'rb'))
    scaler = pickle.load(open(scaler_path, 'rb'))

    # load in the features
    print(sys.argv[1])
    features = pd.read_csv(sys.argv[1],sep=',',comment='#')
    features = features.rename(columns={'BAND':'BANDNUM'})
    
    run_ids = features['ID']
    ### Need to use the column names in order given in FEATS above. Input df has columns in Caps.
    col_list=[strg.upper() for strg in FEATS]
    features = features[col_list]
    
    print(features.columns)
    
    ### Convert BAND to integers
    bm = {'g':0, 'r':1, 'i':2, 'z':3}
    features['BANDNUM']=np.array([bm[i] for i in features.BANDNUM.values])
    
    imputed = imputer.transform(features)
    scaled = scaler.transform(imputed)

    clf.n_jobs = 1
    probs = clf.predict_proba(scaled)[:, 1]
    df = pd.DataFrame({'run_id': run_ids, 'score': probs})
    
    ## Save results to .csv file
    save_dir='/global/cfs/cdirs/dasrepo/vpa/supernova_cnn/data/autoscan_data/results/'
    outfile_name=save_dir+'random-forest_scores.csv'
    print(outfile_name)
    df.to_csv(outfile_name, index=False)
