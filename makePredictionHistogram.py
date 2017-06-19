'''

Make the histogram of ML predictions

'''
from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use('Agg')
import sys
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

	ID = sys.argv[1]
	
	predictions = np.load('./results/' + str(ID) + '/predictions.npy')
	truth = np.load('./results/Y_Val.npy')
	
	elecs = []
	prots = []
	
	for i in range(truth.shape[0]):
		if truth[i] == 1:
			elecs.append(predictions[i])
		else:
			prots.append(predictions[i])
			
	fig1 = plt.figure()
	plt.hist(elecs,50,label='e',alpha=0.5,histtype='step',color='green')
	plt.hist(prots,50,label='p',alpha=0.5,histtype='step',color='red')
	plt.legend(loc='best')
	plt.yscale('log')
	plt.savefig('predictionHistogram')
	#~ plt.show()
	
