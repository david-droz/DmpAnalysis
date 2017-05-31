'''

Make the histogram of ML predictions

'''

import sys
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

	ID = sys.argv[1]
	
	predictions = np.load('./results/' + str(ID) + '/predictions.npy')
	truth = np.load('./results/Y_Val.npy')
	
	elecs = []
	prots = []
	
	for i in xrange(truth.shape[0]):
		if truth[i] == 1:
			elecs.append(predictions[i])
		else:
			prots.append(predictions[i])
			
	fig1 = plt.figure()
	plt.hist(elecs,20,label='e')
	plt.hist(prots,20,label='p')
	plt.legend(loc='best')
	plt.yscale('log')
	plt.show()
	
