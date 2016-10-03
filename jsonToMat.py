import json
import numpy as np
import scipy.io as sio
from scipy.sparse import coo_matrix
import os

def jsonToMat(jsonfile):
    f = open(jsonfile, 'rb')
    obj = json.loads(f.read())
    nodes = obj['nodes']
    links = obj['links']
    features = obj['features']

    n = len(nodes)

    row = []
    col = []
    data = []

    for link in links:
        source = link['source']
        target = link['target']
        value = link['value']
        row.append(int(source[4:])-1)
        col.append(int(target[4:])-1)
        data.append(value)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    aa = coo_matrix((data,(row,col)), shape=(n,n))

    row = []
    col = []
    data = []
    names = []
    groups=[]

    for node in nodes:
        id = node['id']
        name = node['name']
        group = '0'#node['group']
        names.append(name)
        groups.append(group)
        for fea_ind, fea in enumerate(features):
            if (node[fea]!=0.0):
                row.append(int(id[4:])-1)
                col.append(fea_ind)
                data.append(node[fea])

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    soft_label = coo_matrix((data,(row,col)), shape=(n,len(features)), dtype=np.double)



    sio.savemat(os.path.splitext(jsonfile)[0]+'.mat', {'features':features,'names':names, 'groups':groups, 'aa':aa, 'soft_label':soft_label})



if __name__ == "__main__":
    '''
        provide the json file location
    '''
    jsonToMat('./data/dblp.json')