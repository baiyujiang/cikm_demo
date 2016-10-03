"""
author:Liangyue Li (jackiey99@gmail.com)
date: Dec 29,2013
"""

import numpy as np


def label_direct_recommend(aa, L, currentTeam, i0):
    """
	replacement using graph kernel on labeled graph
	aa: people-people network
	L: label matrices
	currentTeam: current team members
	i0: the member to be replaced

	return score: a n*2 matrix, each row is [score, user id]
	"""

    #L = L.todense()
    dn = L.shape[1]

    n = aa.shape[0]
    remainTeam = np.array([i for i in currentTeam if i != i0])
    currentTeam = np.append(remainTeam, i0)
    n1 = len(currentTeam)

    A1 = np.array(aa[currentTeam - 1][:, currentTeam - 1].todense())
    A1 = BLin_W2P(A1)


    cand = np.array([i for i in range(1, n + 1) if i not in currentTeam])
    cand = cand[np.array(aa[cand - 1][:, remainTeam - 1].sum(axis=1) > 0).flatten()]

    c = 0.00000001
    score = np.zeros((len(cand), 2))

    for i, can in enumerate(cand):
        newTeam = np.append(remainTeam, can)

        LL = np.zeros((n1 ** 2, 1))

        for j in range(dn):
            LL = LL + np.kron(L[currentTeam - 1, j], L[newTeam - 1, j])

        LL = np.diag(np.array(LL).T[0])

        A2 = np.array(aa[newTeam - 1][:, newTeam - 1].todense())
        A2 = BLin_W2P(A2)

        score[i, 0] = label_gs(A1, A2, LL, c)
        score[i, 1] = can
    return score


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


def gettopcandidates(score):

    score = score[score[:, 0].argsort()[::-1]]
    return map(int, score[:,1][0:10])

    #print '\n'.join(map(str, map(int, score[:, 1][0:11])))



