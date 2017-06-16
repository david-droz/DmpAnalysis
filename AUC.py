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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score, precision_score, average_precision_score, precision_recall_curve, recall_score
from sklearn.metrics import f1_score
import heapq


Y_val = np.load('./results/Y_Val.npy')

IDlist = glob.glob('./results/*/')
IDlist.sort()
l_AUC_ROC = []
l_AUC_PR = []

for ID in IDlist:
	
	try:
		Y_pred = np.load( ID + 'predictions.npy' )
	except IOError:
		continue
	
	l_AUC_ROC.append(roc_auc_score(Y_val,Y_pred))
	
	l_AUC_PR.append(average_precision_score(Y_val,Y_pred))

bestROC_values = heapq.nlargest(10, l_AUC_ROC)
bestROC_indices = [ l_AUC_ROC.index(x) for x in bestROC_values ]

bestPR_values = heapq.nlargest(10, l_AUC_PR)
bestPR_indices = [ l_AUC_PR.index(x) for x in bestPR_values ] 

l_ROC = [   roc_curve( Y_val,  np.load(IDlist[i]+'/predictions.npy') , pos_label=1 ) for i in bestROC_indices ]
l_ROC_ID = [ IDlist[i] for i in bestROC_indices ]

l_PR = [  precision_recall_curve( Y_val,  np.load(IDlist[i]+'/predictions.npy') , pos_label=1 ) for i in bestPR_indices ]
l_PR_ID = [ IDlist[j] for j in bestPR_indices ]


##############
best_AUC_l_precision, best_AUC_l_recall, best_AUC_l_thresholds = l_PR[0]
fscore_best = 0
fscore_best_index = 0

for i in range(len(best_AUC_l_precision)):
	fscore_temp = 2 * best_AUC_l_precision[i] * best_AUC_l_recall[i] / (best_AUC_l_precision[i]+best_AUC_l_recall[i])
	if fscore_temp > fscore_best:
		fscore_best = fscore_temp
		fscore_best_index = i
best_f1_prec = best_AUC_l_precision[fscore_best_index]
best_f1_recall = best_AUC_l_recall[fscore_best_index]

print('-- Best F1 score on best AUC --')
print('F1:',fscore_best)
print('Precision:',best_f1_prec)
print('Recall:',best_f1_recall)
print('AUC:',bestPR_values[0])

################

fscore_best = 0
precision_best = 0
recall_best = 0

for j in range(len(l_PR)):
	a,b,c = l_PR[j]
	for i in range(len(a)):
		fscore_temp = 2 * a[i] * b[i] / (a[i]+b[i])
		if fscore_temp > fscore_best:
			fscore_best = fscore_temp
			precision_best = a[i]
			recall_best = b[i]

print('-- Best F1 score on top 10 AUC --')
print('F1:',fscore_best)
print('Precision:',precision_best)
print('Recall:',recall_best)

#################

precision_average = precision_score( Y_val,  np.load(IDlist[bestPR_indices[0]]+'/predictions.npy') )
recall_average = recall_score( Y_val,  np.load(IDlist[bestPR_indices[0]]+'/predictions.npy') )
f1_average = f1_score( Y_val,  np.load(IDlist[bestPR_indices[0]]+'/predictions.npy') )

print('-- Average on best AUC --')
print('F1:',f1_average)
print('Precision',precision_average)
print('Recall',recall_average)

#################
with open('metric_ROC.pick','wb') as f:
	pickle.dump([l_ROC,l_ROC_ID],f,protocol=2)
with open('metric_PR.pick','wb') as g:
	pickle.dump([l_PR,l_PR_ID],g,protocol=2)

#################
fig1 = plt.figure()
for i in range(len(l_ROC)):
	plt.plot(l_ROC[i][0],l_ROC[i][1],label=l_ROC_ID[i].replace('./results/',''))
plt.legend(loc='best')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC')
plt.savefig('ROC')

fig2 = plt.figure()
for i in range(len(l_PR)):
	plt.plot(l_PR[i][0],l_PR[i][1],label=l_PR_ID[i].replace('./results/',''))
plt.legend(loc='best')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('Precision-Recall')
plt.savefig('PR')
