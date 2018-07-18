import os
import numpy as np
import pickle

BASE = '/homes/du113/scratch/satire-models'

with open(os.path.join(BASE, 'features.p'), 'rb') as fid:
    feat_data = pickle.load(fid)

features = feat_data['coefs']
names = feat_data['fn']


for i, coef in enumerate(features, 1):
    toppc = np.argsort(coef)[-20:]
    topnc = np.argsort(coef)[:20]
    # topcoef = np.hstack([topnc, toppc])

    print('top positive features for estimator {}'.format(i))
    print(names[toppc])
    print('top negative features for estimator {}'.format(i))
    print(names[topnc])
