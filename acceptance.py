'''

Calculation of acceptance

'''

from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use('Agg')
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score, precision_score, average_precision_score, precision_recall_curve, recall_score
from sklearn.metrics import f1_score

if __name__ == '__main__':

	ID = sys.argv[1]
	
	try:
		predictions = np.load('./results/' + str(ID) + '/predictions.npy')[:,0]
	except IndexError:
		predictions = np.load('./results/' + str(ID) + '/predictions.npy')
	
	truth = np.load('../dataset_validate.npy')
	X_val = truth[:,0:-2]
	Y_val = truth[:,-1]
	del truth
	
	bin_edges = np.logspace( 5 , 7 , num=5)			# 4 logarithmic bins from 100 GeV to 10 TeV
	bin_centers = [ (bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
	
	
	for i in range(len(bin_edges)-1):
		
		binpred = []
		bintruth = []
		
		for j in range(len(Y_val)):
			if X_val[j,30] > bin_edges[i] and X_val[j,30] <= bin_edges[i+1]:
				binpred.append( predictions[j] )
				bintruth.append( Y_val[j] )
		
		l_precision, l_recall, l_thresholds = precision_recall_curve(bintruth,binpred)

		generation_area_radius = 1.38
		generation_area = np.pi * generation_area_radius * generation_area_radius * 100 * 100 	# in cm^2
		acceptance = [ 2 * np.pi * generation_area * x for x in l_recall  ]
		bkg_fraction = [ 1 - x for x in l_precision ] 
		
		fig = plt.figure()
		plt.plot(bkg_fraction,acceptance)
		plt.xscale('log')
		plt.xlabel('Background fraction')
		plt.ylabel('Acceptance')
		plt.legend(loc='best')
		
		figname = 'acceptance_' + str(int(bin_edges[i])/1000.) + '-' + str(int(bin_edges[i+1])/1000) + '.png'
		figtitle = 'Acceptance: ' + str(int(bin_edges[i])/1000.) + ' GeV - ' + str(int(bin_edges[i+1])/1000) + ' GeV'
		
		plt.title(figtitle)
		plt.savefig(figname)
