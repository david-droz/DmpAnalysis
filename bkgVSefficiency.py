'''

Given the ID of a machine learning run, compute the "Background/Efficiency" curve,
i.e. on the X-axis is signal efficiency and on the Y axis is the residual background fraction

and saves it as a pickle file for later use/plotting

'''

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import sys

from sklearn.metrics import roc_auc_score

# Stephan on Slack (10 Aug. 2017) :
# on the x-axis you have signal efficiency
# and on the y-axis: residual background fraction 
#  ~ the fraction of protons in my signal region ?

def getcounts(truth,pred,threshold):
	tp, fp, tn, fn = [0,0,0,0]
	for i in range(truth.shape[0]):
		if truth[i] == 1. :
			if pred[i] >= threshold:
				tp += 1
			else:
				fn += 1
		elif truth[i] == 0. :
			if pred[i] >= threshold:
				fp += 1
			else:
				tn += 1
	return tp,fp,tn,fn
	
def getcountsFast(truth,pred,threshold):
	
	pred_e = pred[truth.astype(bool)]
	pred_p = pred[~truth.astype(bool)]
	
	tp = pred_e[ pred_e >= threshold].shape[0]
	fn = pred_e[ pred_e < threshold].shape[0]
	fp = pred_p[ pred_p >= threshold].shape[0]
	tn = pred_p[ pred_p < threshold].shape[0]
	
	del pred_e, pred_p
	return tp, fp, tn, fn
	
	
ID = sys.argv[1]

try:
	Y_val = np.load('results/Y_Val.npy')
except:
	Y_val = np.load('results/'+ID+'/Y_Val.npy')
predictions = np.load('results/'+ID+'/predictions.npy')

l_bkg = []
l_efficiency = []
l_thresholds = []
npoints = 1000

print('ROC AUC score: ', roc_auc_score(Y_val,predictions))

for i in range(npoints):
	thr = i * (1./npoints)
	tp,fp,tn,fn = getcountsFast(Y_val,predictions,thr)
	
	try:
		l_bkg.append( fp / (tp + fp) )
	except ZeroDivisionError :
		l_bkg.append( 1 )
	l_efficiency.append( tp / (tp + fn) )
	l_thresholds.append( thr )


with open('data_roc_'+ID+'.pick','wb') as f:
	pickle.dump([l_bkg,l_efficiency],f,protocol=2)
