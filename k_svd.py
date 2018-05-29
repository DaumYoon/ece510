from numpy import matrix
from stopping_rule import stopping_rule
from pursuit import pursuit
import numpy as np
#T_0 is the upper bound of 0-norm of x_i. Y is the data samples of size nxN. K is the K of K-SVD
def k_svd(K,T_0,Y):
    
    size_Y = Y.shape
    n = size_Y[0]
    N = size_Y[1]

    D=np.ones([n, K])
    temp_norm = np.linalg.norm(D[:,0])
    D/=temp_norm

	#set numIter for stopping_rule(J,numIter)
    numIter = 100

    J=1

    #provide a stopping rule. It returns false when while loop should stop.
    while stopping_rule(J,numIter):
        #provide a pursuit program
        X = pursuit(D,T_0,Y)
        omega = []

        for k in range (0, K):

            #build omega
            for i in range (0,N):
                if X[k][i]!=0:
                    omega.append(i)

            #build E
            E = Y
            for j in [j for j in range (0, K) if j != k]:
                E -= D[:,j]@X[j,:]

            #build E restricted
            E = np.delete(E, omega, 1)

            u, s, vh = np.linalg.svd(E, full_matrices=True)
            D[:,k] = u[:,0]
            
	    #construct X[k:] from SVD. Since s[0]*vh[0,:] is reduced row vector, this needs to be elongated before inserting in X[k:]
            x_r = s[0]*vh[0,:]
            for i in range (0,N):
                if i in omega:
                    X[k,i] = 0
                    omega.pop(0)
                else:
                    X[k,i] = x_r.pop(0)

        J += 1

    return D, X

