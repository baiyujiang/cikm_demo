import numpy as np
import networkx as nx
import cPickle as pkl
import sys
from scipy.sparse import find

def convert2Networkx(aa, L, output):
    """
    this function preprocess the graph for input to team formation algorithm
    1) the weights should reflect the communication cost
    2) convert it to Networkx format
    """
    # convert aa into a new weight matrix using Jaccard distance

    # build attribute dictionary
    print "building attribute dictionary ..."
    S_attribute = {}
    dn = L.shape[1]
    for i in range(dn):
        Si = L[:, i].nonzero()
        S_attribute[i+1] = Si[0] + 1


    print "converting to networkx format"

    # build Networkx Graph
    [I, J, _] = find(aa > 2)
    # V = A[I, J]
    G = nx.Graph()
    I =I + 1
    J =J + 1
    # G.add_weighted_edges_from(zip(I, J, V))
    G.add_edges_from(zip(I, J))
    # all_pair_shortest_length = nx.all_pairs_dijkstra_path_length(G)
    # all_pair_shortest_path = nx.all_pairs_dijkstra_path(G)

    print "Runnig shortest path algorithms..."
    all_pair_shortest_length = nx.all_pairs_shortest_path_length(G, cutoff = 2)
    all_pair_shortest_path = nx.all_pairs_shortest_path(G, cutoff = 2)
    pkl.dump({'S_attribute' : S_attribute, 'all_pair_shortest_length' : all_pair_shortest_length, 'all_pair_shortest_path' : all_pair_shortest_path}, open(output,'wb'), protocol=pkl.HIGHEST_PROTOCOL);



def teamformation_rarestFirst(T, S_attribute, all_pair_shortest_length, all_pair_shortest_path):
    """
    The RarestFirst algorithm for team formation
    Reference: Finding a Team of Experts in Social Netowrks, KDD 09
    """
    a_rare = None
    min_cover = sys.maxint
    for a in T:
        S_a = S_attribute[a]
        if len(S_a) < min_cover:
            min_cover = len(S_a)
            a_rare = a

    gen = (x for x in T if x != a_rare)
    min_Ri = sys.maxint
    i_star = None
    for i in S_attribute[a_rare]:
        max_Ria = 0
        for a in gen:
            S_a = S_attribute[a]
            min_d = sys.maxint
            for j in S_a:
                try: 
                    min_d = min(min_d, all_pair_shortest_length[i][j])
                except:
                    continue

            R_ia = min_d
            max_Ria = max(max_Ria, R_ia)
        if min_Ri > max_Ria:
            min_Ri = max_Ria
            i_star = i 
    team = [i_star]
    for a in T:
        S_a = S_attribute[a]
        min_d = sys.maxint
        i_prime = None
        for j in S_a:
            try:
                if min_d > all_pair_shortest_length[i_star][j]:
                    min_d = all_pair_shortest_length[i_star][j]
                    i_prime = j
            except:
                continue
        try: 
            path = all_pair_shortest_path[i_star][i_prime]
            team = team + path
        except:
            continue
    return list(set(team))