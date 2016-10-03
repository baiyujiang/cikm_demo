import os, os.path

import scipy.io
import scipy.sparse as ss

from jsonToMat import jsonToMat
from label_shrink import label_shrink
from label_conflict import label_conflict
from team_formation import *
import cPickle as pkl


def maximum (A, B):
    BisBigger = A-B
    BisBigger.data = np.where(BisBigger.data < 0, 1, 0)
    return A - A.multiply(BisBigger) + B.multiply(BisBigger)

if __name__ == "__main__":

    #Inputs
    jsonfile = 'data/dblp-5000.json'
    jsonToMat(jsonfile)
    mat = scipy.io.loadmat(os.path.splitext(jsonfile)[0])

    aa = mat['aa']
    aa = maximum(aa, aa.T)
    aa = ss.triu(aa, 1) + ss.tril(aa, -1)
    # aa = aa[0:250, 0:250]
    L = mat['soft_label']
    L = np.array(L.todense())
    # L = L[0:250, :]

    features = list(mat['features'])
    features = [f.strip() for f in features]
    names = mat['names']
    groups = mat['groups']


    output = 'data/dblp_5000_networkx_output.pkl'

    if not os.path.isfile(output):
        print "starting converting..."
        convert2Networkx(aa, L, output)


