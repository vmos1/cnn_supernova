import os
import sys
import pickle
import joblib
import pandas as pd

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

FEATS=['AMP', 'A_IMAGE', 'A_REF', 'B_IMAGE', 'B_REF', 'CCDID', 'COLMEDS', 
 'DIFFSUMRN', 'ELLIPTICITY', 'FLAGS', 'FLUX_RATIO', 
 'GAUSS', 'GFLUX', 'L1', 'LACOSMIC', 'MAG', 'MAGDIFF', 'MAGLIM', 
 'MAG_FROM_LIMIT', 'MAG_REF', 'MAG_REF_ERR', 'MASKFRAC', 
 'MIN_DISTANCE_TO_EDGE_IN_NEW', 'N2SIG3', 'N2SIG3SHIFT', 
 'N2SIG5', 'N2SIG5SHIFT', 'N3SIG3', 'N3SIG3SHIFT', 
 'N3SIG5', 'N3SIG5SHIFT', 'NN_DIST_RENORM', 'NUMNEGRN', 
 'SCALE',  'SNR', 'SPREADERR_MODEL', 'SPREAD_MODEL']


if __name__ == '__main__':

#     classifier_path = os.getenv('ML_CLASSIFIER')
#     imputer_path = os.getenv('ML_IMPUTER')
#     scaler_path = os.getenv('ML_SCALER')
    
    classifier_path = '/global/cfs/cdirs/dasrepo/vpa/supernova_cnn/data/autoscan_data/data/ml3.2.classifier/ml3.2.comp'
    imputer_path = '/global/cfs/cdirs/dasrepo/vpa/supernova_cnn/data/autoscan_data/data/imputerml3.2.obj'
    scaler_path = '/global/cfs/cdirs/dasrepo/vpa/supernova_cnn/data/autoscan_data/data/scalerml3.2.obj'
    
    clf = joblib.load(classifier_path)
    imputer = pickle.load(open(imputer_path, 'rb'))
    scaler = pickle.load(open(scaler_path, 'rb'))

    # load in the features
    print(sys.argv[1])
    features = pd.read_csv(sys.argv[1],sep=',',comment='#')

    numbers = features['ID']
    features = features[FEATS]

    imputed = imputer.transform(features)
    scaled = scaler.transform(imputed)

    clf.n_jobs = 1
    probs = clf.predict_proba(scaled)[:, 1]
    df = pd.DataFrame({'number': numbers, 'score': probs})
    outfile_name = sys.argv[1].replace('.features.csv', '.scored.csv')
    df.to_csv(outfile_name, index=False)
