import numpy as np

def stopping_rule(J,numIter):
	if J<=numIter:
		return True
	else:
		return False

