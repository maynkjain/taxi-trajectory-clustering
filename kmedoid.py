import numpy as np
import math

def kMedoids(D, k, tmax=100):
    m, n = D.shape

    np.fill_diagonal(D, math.inf)

    if k > n:
        raise Exception('too many medoids')
    M = np.arange(n)
    np.random.shuffle(M)
    M = np.sort(M[:k])
    Mnew = np.copy(M)
    C = {}
    for t in range(tmax):
        J = np.argmin(D[:,M], axis=1)

        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]

    np.fill_diagonal(D, 0)

    return M, C