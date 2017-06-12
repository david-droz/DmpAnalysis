'''

Finds the 10 best AUC (Area Under Curve) for the ROC curve and the Precision-Recall curve, then compute the curves themselves
and saves them as pickle to be plotted later

'''


from __future__ import print_function, division, absolute_import

import numpy as np
import pickle
import sys
import os
import glob

from sklearn.metrics import roc_curve, roc_auc_score, precision_score, average_precision_score, precision_recall_curve, recall_score
from sklearn.metrics import f1_score
import heapq


Y_val = np.load('./results/Y_Val.npy')

IDlist = glob.glob('./results/*/')
IDlist.sort()
l_AUC_ROC = []
l_AUC_PR = []

for ID in IDlist:
	
	Y_pred = np.load( ID + 'predictions.npy' )
	
	l_AUC_ROC.append(roc_auc_score(Y_val,Y_pred))
	
	l_AUC_PR.append(average_precision_score(Y_val,Y_pred))

bestROC_values = heapq.nlargest(10, l_AUC_ROC)
bestROC_indices = [ l_AUC_ROC.index(x) for x in bestROC_values ]

bestPR_values = heapq.nlargest(10, l_AUC_PR)
bestPR_indices = [ l_AUC_PR.index(x) for x in bestPR_values ] 


l_ROC = [   roc_curve( Y_val,  np.load(IDlist[i]+'predictions.npy') , pos_label=1 ) for i in bestROC_indices ]
l_ROC_ID = [ IDlist[i] for i in bestROC_indices ]

l_PR = [  precision_recall_curve( Y_val,  np.load(IDlist[i]+'predictions.npy') , pos_label=1 ) for i in bestPR_indices ]
l_PR_ID = [ IDlist[j] for j in bestPR_indices ]

with open('metric_ROC.pick','wb') as f:
	pickle.dump([L_ROC,l_ROC_ID],f,protocol=2)
with open('metric_PR.pick','wb') as g:
	pickle.dump([L_PR,l_PR_ID],g,protocol=2)

