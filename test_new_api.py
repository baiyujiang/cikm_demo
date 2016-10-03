import os, os.path

import scipy.io
import scipy.sparse as ss

from jsonToMat import jsonToMat
from label_shrink import label_shrink
from label_conflict import label_conflict
from team_formation import *
import cPickle as pkl

import time


def maximum (A, B):
    BisBigger = A-B
    BisBigger.data = np.where(BisBigger.data < 0, 1, 0)
    return A - A.multiply(BisBigger) + B.multiply(BisBigger)

if __name__ == "__main__":

    #Inputs
    #jsonfile = ('data/brad_pitt.json')
    did = 'dblp-subgraph'
    jsonfile = './data/' + did + '.json'
    matfile = './data/' + did + '.mat'

    # currentTeam = [200, 382, 144, 377, 194, 228, 426, 368, 440, 555]
    currentTeam = [200, 144, 194, 228]

    #Load data
    if not os.path.isfile(matfile):
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

    ##################### team shrinkage
    currentTeam = np.array(currentTeam)
    top_user_id = label_shrink(aa, L, currentTeam)
    print top_user_id


    ##################### team conflict resolution
    a = 200
    b = 228
    newTeam = label_conflict(aa, L, currentTeam, a, b)
    print newTeam

    #################### team formation

    T = [1,2, 3] # T is the skill set required by the task
    output = 'dblp_subgraph_networkx_output.pkl'
    #output = 'brad_networkx_output.pkl'
    if not os.path.isfile(output):
        convert2Networkx(aa, L, output)

    t1 = time.time()
    networkx_output = pkl.load(open(output, 'rb'))
    print time.time() - t1
    S_attribute = networkx_output['S_attribute']
    all_pair_shortest_length = networkx_output['all_pair_shortest_length']
    all_pair_shortest_path = networkx_output['all_pair_shortest_path']


    t1 = time.time()
    start_team = teamformation_rarestFirst(T, S_attribute, all_pair_shortest_length, all_pair_shortest_path)
    print start_team
    print time.time() - t1



