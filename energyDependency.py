'''

Energy dependency of precision and recall

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
	
	bin_edges = np.logspace( 5 , 7 , num=7)			# 6 logarithmic bins from 100 GeV to 10 TeV
	bin_centers = [ (bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
	
	recall_at_prec95 = []
	recall_at_prec96 = []
	recall_at_prec97 = []
	recall_at_prec98 = []
	precision_at_prec95 = []
	precision_at_prec96 = []
	precision_at_prec97 = []
	precision_at_prec98 = []
	
	for i in range(len(bin_edges)-1):
		
		binpred = []
		bintruth = []
		
		for j in range(len(Y_val)):
			if X_val[j,30] > bin_edges[i] and X_val[j,30] <= bin_edges[i+1]:
				binpred.append( predictions[i] )
				bintruth.append( Y_val[i] )
		
		l_precision, l_recall, l_thresholds = precision_recall_curve(bintruth,binpred)


		prec_95 = 0
		rc_95 = 0
		f1_95 = 0
		prec_96 = 0
		rc_96 = 0
		f1_96 = 0
		prec_97 = 0
		rc_97 = 0
		f1_97 = 0
		prec_98 = 0
		rc_98 = 0
		f1_98 = 0
		for i in range(len(l_precision)):
			
			# 95%
			if l_precision[i] > 0.95:
				temp_f1 = 2*(l_precision[i]*l_recall[i])/(l_precision[i]+l_recall[i])
				if temp_f1 > f1_95:
					prec_95 = l_precision[i]
					rc_95 = l_recall[i]
					f1_95 = temp_f1
			# 96%
			if l_precision[i] > 0.96:
				temp_f1 = 2*(l_precision[i]*l_recall[i])/(l_precision[i]+l_recall[i])
				if temp_f1 > f1_96:
					prec_96 = l_precision[i]
					rc_96 = l_recall[i]
					f1_96 = temp_f1
			# 97%
			if l_precision[i] > 0.97:
				temp_f1 = 2*(l_precision[i]*l_recall[i])/(l_precision[i]+l_recall[i])
				if temp_f1 > f1_97:
					prec_97 = l_precision[i]
					rc_97 = l_recall[i]
					f1_97 = temp_f1
			# 98%
			if l_precision[i] > 0.98:
				temp_f1 = 2*(l_precision[i]*l_recall[i])/(l_precision[i]+l_recall[i])
				if temp_f1 > f1_98:
					prec_98 = l_precision[i]
					rc_98 = l_recall[i]
					f1_98 = temp_f1
					
		precision_at_prec95.append(prec_95)
		recall_at_prec95.append(rc_95)	
		precision_at_prec96.append(prec_96)
		recall_at_prec96.append(rc_96)
		precision_at_prec97.append(prec_97)
		recall_at_prec97.append(rc_97)
		precision_at_prec98.append(prec_98)
		recall_at_prec98.append(rc_98)

	
	fig1 = plt.figure()
	plt.plot(bin_centers,recall_at_prec95,'.',label='precision > 0.95')
	plt.plot(bin_centers,recall_at_prec96,'.',label='precision > 0.96')
	plt.plot(bin_centers,recall_at_prec97,'.',label='precision > 0.97')
	plt.plot(bin_centers,recall_at_prec98,'.',label='precision > 0.98')
	plt.xscale('log')
	plt.xlabel('Energy [MeV]')
	plt.ylabel('Recall')
	plt.legend(loc='best')
	plt.savefig('energy_recall')

	fig2 = plt.figure()
	plt.plot(bin_centers,precision_at_prec95,'.',label='precision > 0.95')
	plt.plot(bin_centers,precision_at_prec96,'.',label='precision > 0.96')
	plt.plot(bin_centers,precision_at_prec97,'.',label='precision > 0.97')
	plt.plot(bin_centers,precision_at_prec98,'.',label='precision > 0.98')
	plt.xscale('log')
	plt.xlabel('Energy [MeV]')
	plt.ylabel('Precision')
	plt.legend(loc='best')
	plt.savefig('energy_precision')
