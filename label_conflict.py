"""
This utility is for resolving the conflict between two team members
author:Liangyue Li (jackiey99@gmail.com)
date: Aug 29, 2016
"""

import numpy as np
from label_direct_recommend import label_direct_recommend



def label_conflict(aa, L, currentTeam, a, b):
    """
	conflict resolution using graph kernel on labeled graph
	aa: people-people network
	L: label matrices
	currentTeam: current team members
    a, b: the two conflicting team members

    return: a new team without the conflict
	"""

    scores_replace_a = label_direct_recommend(aa, L, currentTeam, a)
    s_rep_a, id_rep_a = gettopscoreandid(scores_replace_a)
    scores_replace_b = label_direct_recommend(aa, L, currentTeam, b)
    s_rep_b, id_rep_b = gettopscoreandid(scores_replace_b)
    
    # now consider removing a or b
    dn = L.shape[1]

    n = aa.shape[0]

    n1 = len(currentTeam)

    A1 = np.array(aa[currentTeam - 1][:, currentTeam - 1].todense())
    A1 = BLin_W2P(A1)

    cand = [a, b] 
    
    c = 0.00000001
    score_remove = np.zeros((len(cand), 2))

    for _i, can in enumerate(cand):
        newTeam = np.array([i for i in currentTeam if i != can])

        LL = np.zeros((n1 * (n1 - 1), 1))

        for j in range(dn):
            LL = LL + np.kron(L[currentTeam - 1, j], L[newTeam - 1, j])

        LL = np.diag(np.array(LL).T[0])

        A2 = np.array(aa[newTeam - 1][:, newTeam - 1].todense())
        A2 = BLin_W2P(A2)

        score_remove[_i, 0] = label_gs(A1, A2, LL, c)
        score_remove[_i, 1] = can
    s_remove, id_remove = gettopscoreandid(score_remove)

    max_s = max(s_rep_a, s_rep_b, s_remove)
    if max_s == s_rep_a:
        newTeam = np.array([i for i in currentTeam if i != a])
        newTeam = np.append(newTeam, id_rep_a)
    elif max_s == s_rep_b:
        newTeam = np.array([i for i in currentTeam if i != b])
        newTeam = np.append(newTeam, id_rep_b)
    else:
        newTeam = np.array([i for i in currentTeam if i != id_remove])

    return newTeam



def BLin_W2P(W):
    """
	Set up transition matrix for random walk
	The diagnal of W is set zero
	"""

    W = np.triu(W, 1) + np.tril(W, -1)
    n = W.shape[0]

    D0 = np.sum(W, axis=1)
    D0 = np.maximum(D0, 0.00000000001)
    D0 = 1. / D0
    D0 = np.diag(D0)

    return np.dot(D0, W)


def label_gs(A, B, L, c):
    n1 = A.shape[0]
    n2 = B.shape[0]

    p1 = np.ones((n1, 1)) / n1
    p2 = np.ones((n2, 1)) / n2
    q1 = np.ones((n1, 1)) / n1
    q2 = np.ones((n2, 1)) / n2

    X = np.kron(A, B)
    qx = np.kron(q1, q2)
    px = np.kron(p1, p2)
    return np.dot(np.dot(np.dot(qx.T, np.linalg.inv(np.eye(n1 * n2) - c * np.dot(L, X))), L), px)


def gettopscoreandid(score):

    score = score[score[:, 0].argsort()[::-1]]
    return score[0,0], int(score[0, 1])



