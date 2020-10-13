#n=1, SIC "POVM", collective measurement
import cvxopt
from cvxopt import matrix, solvers
import numpy as np
import scipy as sp
from itertools import chain
import cmath
import copy

def Amatrix(n):
    A = np.zeros([2**n*2**n, 2**(n-1)*(2**(n)-1)])
    l = 0
    for j in range(2**n):
        if j>0:
            for k in range(j):
                A[2**n*j+k, l] = 1
                A[2**n*k+j, l] = -1
                l+=1
    return A

def kronecker_prod(rp):
    r = np.array([[1]])
    for i in range(len(rp)):
        r = np.kron(r, rp[i])
    return r

def SDP(rho, q, n):
    c = np.eye(2**n).flatten().tolist()
    c = matrix(c)
    A, b = [matrix(Amatrix(n).tolist())], [matrix(np.zeros((1, 2**(n-1)*(2**n-1))).tolist())]
    A, b = cvxopt.matrix(A), cvxopt.matrix(b)
    G = [matrix((-1*np.eye(2**n*2**n)).tolist())]
    h = [ matrix((-q[len(rho)-1]*kronecker_prod(rho[len(rho)-1])).tolist()) ]
    for _ in range(len(rho)-1):
        G += [matrix((-1*np.eye(2**n*2**n)).tolist())]
        h += [ matrix((-q[_]*kronecker_prod(rho[_])).tolist()) ]
    solvers.options['show_progress'] = False
    sol = solvers.sdp(c, Gs=G, hs=h, A=A, b=b)
    v = []
    for _ in range(2**n*2**n):
        v.append(sol['x'][_])
    value = np.array(v)@np.eye(2**n).flatten()
    return value

