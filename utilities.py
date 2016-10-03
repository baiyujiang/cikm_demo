import numpy as np
from collections import OrderedDict
import json
from label_direct_recommend import *
from structure_only import *
import scipy.sparse as ss
from label_only import *
import sys
import networkx as nx

def getEgonet(center, aa):

    weights = np.array(aa[center,:].todense())[0]
    max_index = np.argsort(weights)[::-1]
    pos_index = np.nonzero(weights)[0]
    neighbors = np.intersect1d(max_index, pos_index)

    teammembers = np.insert(neighbors[0:15],0,center)
    egonet = aa[teammembers][:,teammembers]
    egonet = np.array(egonet.todense())

    return teammembers,egonet

def getgraph(newTeam,aa):

    return np.array(aa[newTeam-1][:,newTeam-1].todense())

def dumptojson(teamgraph, teammembers, features, names, groups, labels, filename):

    #link list
    links = []
    for i in range(np.count_nonzero(teamgraph)):

        s = teamgraph.nonzero()[0][i]
        t = teamgraph.nonzero()[1][i]

        if s!=t:
            link = OrderedDict()
            link["source"]="user"+str(teammembers[s])
            link["target"]="user"+str(teammembers[t])
            link["value"]=teamgraph[s,t]
            links.append(link)


    #node list, each node is a dictionary
    m,n = labels.shape
    nodes = []
    for person_ind in range(m):
        node = OrderedDict()
        node["id"] = "user" + str(teammembers[person_ind])
        node["name"] = names[person_ind].strip()
        node["group"] = groups[person_ind].strip()
        for feature_ind in range(n):
            node[features[feature_ind]] = labels[person_ind,feature_ind]
        nodes.append(node)

    vis_data = OrderedDict()
    vis_data["features"] = features
    vis_data["nodes"] = nodes
    vis_data["links"] = links
    with open(filename+'.json','wb') as f:
        f.write(json.dumps(vis_data, indent = ' '))

def assemble(teamgraph, teammembers, features, names, groups, labels):

    #link list
    links = []
    for i in range(np.count_nonzero(teamgraph)):

        s = teamgraph.nonzero()[0][i]
        t = teamgraph.nonzero()[1][i]

        if s!=t:
            link = OrderedDict()
            link["source"]="user"+str(teammembers[s])
            link["target"]="user"+str(teammembers[t])
            link["value"]=teamgraph[s,t]
            links.append(link)

    #node list, each node is a dictionary
    m,n = labels.shape
    nodes = []
    for person_ind in range(m):
        node = OrderedDict()
        node["id"] = "user"+str(teammembers[person_ind])
        node["name"] = names[person_ind].strip()
        node["group"] = groups[person_ind].strip()
        for feature_ind in range(n):
            node[features[feature_ind]] = labels[person_ind,feature_ind]
        nodes.append(node)

    vis_data = OrderedDict()
    vis_data["features"] = features
    vis_data["nodes"] = nodes
    vis_data["links"] = links
    return vis_data

def teamassemble(aa,currentTeam,CV,EV,L):
    n = aa.shape[0]
    virtualperson = n+1

    row = np.array([0]*len(CV))
    col = np.array(currentTeam-1)
    w = ss.csc_matrix((np.array(CV), (row, col)), shape=(1, n))
    aa = ss.vstack([aa, w])
    w = ss.csc_matrix((np.array(CV), (col, row)), shape=(n+1, 1))
    aa = ss.hstack([aa, w])
    aa = ss.csc_matrix(aa)

    L = np.vstack([L,EV])
    newTeam = np.append(currentTeam,virtualperson)
    score = label_direct_recommend(aa, L, newTeam, virtualperson)
    return gettopcandidates(score)

def teamassemble_v2(aa, currentTeam, CV, EV, L, a, b):
    n = aa.shape[0]
    virtualperson = n+1

    row = np.array([0]*len(CV))
    col = np.array(currentTeam-1)
    w = ss.csc_matrix((np.array(CV), (row, col)), shape=(1, n))
    aa = ss.vstack([aa, w])
    w = ss.csc_matrix((np.array(CV), (col, row)), shape=(n+1, 1))
    aa = ss.hstack([aa, w])
    aa = ss.csc_matrix(aa)

    L = np.vstack([L,EV])
    newTeam = np.append(currentTeam,virtualperson)
    CV.append(0)

    return teamrefine_v2(aa, newTeam, CV, EV, L, virtualperson, a, b)

def teamrefine(aa, currentTeam, CV,EV, L, id):
    aa[id-1, currentTeam-1] = np.array(CV)
    aa[currentTeam-1, id-1] = np.array(CV).reshape((len(CV),1))

    L[id-1,:] = EV
    score = label_direct_recommend(aa, L, currentTeam, id)
    return gettopcandidates(score)

def teamrefine_v2(aa, currentTeam, CV, EV, L, id, a, b):
    aa[id - 1, currentTeam - 1] = np.array(CV)
    aa[currentTeam - 1, id - 1] = np.array(CV).reshape((len(CV),1))
    L[id-1, :] = EV
    score_expertise = get_expertise_score(L, EV)
    score_structure = get_structure_score(aa, currentTeam, CV, id)
    scores = a*score_expertise + b*score_structure
    sorted_index = scores.argsort()[::-1] + 1
    sorted_index = [i for i in sorted_index if i not in currentTeam]
    return sorted_index[0:30]


def multiple_replacement(aa, currentTeam, L, uid):
    if len(currentTeam) == len(uid): # get first candidate using expertise match
        ev = L[currentTeam[0]-1, :]
        score = get_expertise_score(L, ev)
        for c in score.argsort()[::-1] + 1:
            if c not in currentTeam:
                top_c = c
                break
        old_team = [currentTeam[0]]
        new_team = [top_c]
        uid = [u for u in uid if u!= currentTeam[0]]

    else:
        team = np.array([t for t in currentTeam if t not in uid])
        old_team = team.copy()
        new_team = team.copy()

    for id in uid:
        old_team = np.append(old_team, id)
        new_team = np.append(new_team, id)
        cv = np.array(aa[id-1, old_team - 1].todense())[0]
        ev = L[id-1, :]
        can = teamrefine_v2(aa, new_team, cv, ev, L, id, 0.1, 1)
        for c in can:
            if c not in currentTeam:
                top_c = c
                break
        new_team[-1] = top_c
    return new_team


def virtual_to_actual(aa, currentTeam, L, CV, EV, uid):
    n = aa.shape[0]

    actual_nodes = [i for i in currentTeam if i > 0]
    virtual_nodes = [i for i in currentTeam if i < 0]
    newTeam = currentTeam.copy()

    temp = 0
    for i, t in enumerate(newTeam):
        if t < 0:
            newTeam[i] = n + 1 + temp
            temp += 1

    for i in range(0, len(virtual_nodes)):
        connection = np.array(CV[i])
        row = np.array([0]*len(connection[currentTeam > 0]))
        col = np.array(np.array(actual_nodes) - 1)
        w = ss.csc_matrix((np.array(connection[currentTeam > 0]), (row, col)), shape=(1, n))
        aa = ss.vstack([aa, w])

    for i in range(0, len(virtual_nodes)):
        connection = np.array(CV[i])
        row = np.array(newTeam - 1)
        col = np.array([0] * len(newTeam))
        w = ss.csc_matrix((np.array(connection), (row, col)), shape=(n+len(virtual_nodes), 1))
        aa = ss.hstack([aa, w])
        aa = ss.csc_matrix(aa)

    L = np.vstack([L,EV])

    for i, u in enumerate(uid):
        if u in virtual_nodes:
            uid[i] = virtual_nodes.index(u) + 1 + n

    return aa, newTeam, uid, L

def update_graph(aa, currentTeam, L, CV, EV, uid):
    for i,u in enumerate(uid):
        aa[u-1, currentTeam-1] = np.array(CV[i])
        aa[currentTeam-1, u-1] = np.array(CV[i]).reshape((len(CV[i]),1))
        L[u-1, :] = EV[i]

    return aa, L






