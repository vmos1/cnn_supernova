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


if __name__ == '__main__':

    classifier_path = os.getenv('ML_CLASSIFIER')
    imputer_path = os.getenv('ML_IMPUTER')
    scaler_path = os.getenv('ML_SCALER')

    clf = joblib.load(classifier_path)
    imputer = pickle.load(open(imputer_path, 'rb'))
    scaler = pickle.load(open(scaler_path, 'rb'))

    # load in the features
    features = pd.read_csv(sys.argv[1])

    numbers = features['number']
    features = features[FEATS]

    imputed = imputer.transform(features)
    scaled = scaler.transform(imputed)

    clf.n_jobs = 1
    probs = clf.predict_proba(scaled)[:, 1]
    df = pd.DataFrame({'number': numbers, 'score': probs})
    ## the line below is dangerous. it can accidentally modify source file. don't run!!
#     outfile_name = sys.argv[1].replace('.features.csv', '.scored.csv')
    df.to_csv(outfile_name, index=False)