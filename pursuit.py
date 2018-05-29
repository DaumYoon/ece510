import numpy as np

def pursuit(D,T_0,Y):
	
	size_Y=Y.shape
	size_D=D.shape
	N=size_Y[1]
	k=size_D[1]
	X=np.zeros([k,N])
	#set threshold
	threshold = 0.001

	for col in range (0,N):
		R=Y[:,col]
		for n in range (0,T_0):
			if np.linalg.norm(R) < threshold:
				break
			max_d=0;
			max_inner=0;
			for d in range (0,k):
				temp_inner = R.transpose()@D[:,d]
				if temp_inner > max_inner:
					max_d=d
					max_inner=temp_inner
			X[max_d,col] = R.transpose()@D[:,max_d]/D[:,max_d].tranpose()*D[:,max_d]
			R-=X[max_d,col]*D[:,max_d]

	return X
    
