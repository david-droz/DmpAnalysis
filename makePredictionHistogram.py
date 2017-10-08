'''

Make the histogram of ML predictions

'''
from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use('Agg')
import sys
import numpy as np
import matplotlib.pyplot as plt

def getClassifierScore(truth,pred):
	elecs = pred[truth.astype(bool)]
	prots = pred[~truth.astype(bool)]
			
	return elecs, prots


if __name__ == '__main__':

	ID = sys.argv[1]
	
	try:
		predictions = np.load('./results/' + str(ID) + '/predictions.npy')[:,0]
	except IndexError:
		predictions = np.load('./results/' + str(ID) + '/predictions.npy')
	truth = np.load('./results/Y_Val.npy')
	
	elecs = []
	prots = []
	
	#~ for i in range(truth.shape[0]):
		#~ if truth[i] == 1:
			#~ elecs.append(predictions[i])
		#~ else:
			#~ prots.append(predictions[i])
			
	
	elecs_p, prots_p = getClassifierScore(truth,predictions)
	binList = [x/50 for x in range(0,51)]
	fig4 = plt.figure()
	plt.hist(elecs_p,bins=binList,label='e',alpha=1.,histtype='step',color='green')
	plt.hist(prots_p,bins=binList,label='p',alpha=1.,histtype='step',color='red',ls='dashed')
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.legend(loc='upper center')
	plt.yscale('log')
	plt.savefig('predHisto_' + str(ID))
	plt.close(fig4)
	
