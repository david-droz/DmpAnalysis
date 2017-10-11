'''

QDA.py

Quadratic Discriminant Analysis

~ sort of PCA that tries to maximise separation between classes



'''

from __future__ import print_function, division, absolute_import

import numpy as np
import time
import pickle
import sys
import os
import random
import hashlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, average_precision_score, precision_recall_curve, recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV

def XY_split(fname):
	arr = np.load(fname)
	X = arr[:,0:-2]				# Last two columns are timestamp and particle ID
	Y = arr[:,-1]
	return X,Y
def load_training(fname='../dataset_train.npy'): return XY_split(fname)
def load_validation(fname='../dataset_validate.npy'): return XY_split(fname)
def load_test(fname='../dataset_test.npy'): return XY_split(fname)

def getClassifierScore(truth,pred):
	elecs = pred[truth.astype(bool)]
	prots = pred[~truth.astype(bool)]
			
	return elecs, prots

def _run():
	
	if not os.path.isdir('pics'): os.mkdir('pics')
	
	outdir = 'pics/LDA'
	if not os.path.isdir(outdir): os.mkdir(outdir)
	
	TRAIN_E_PATH = '/home/drozd/analysis/ntuples/MC-skim-fullBGO-NUD-HET-30Aug17/data_train_elecs_under_1.npy'
	TRAIN_P_PATH = '/home/drozd/analysis/ntuples/MC-skim-fullBGO-NUD-HET-30Aug17/data_train_prots_under_1.npy'
	VAL_E_PATH = '/home/drozd/analysis/ntuples/MC-skim-fullBGO-NUD-HET-30Aug17/data_validate_elecs_under_1.npy'
	VAL_P_PATH = '/home/drozd/analysis/ntuples/MC-skim-fullBGO-NUD-HET-30Aug17/data_validate_prots_under_1.npy'
	
	train_e = np.load(TRAIN_E_PATH)
	train_p = np.load(TRAIN_P_PATH)
	train = np.concatenate(( train_e, train_p ))
	np.random.shuffle(train)
	
	X_train = train[:,0:-2]
	Y_train = train[:,-1]
	del train_e,train_p, train
	
	val_e = np.load(VAL_E_PATH) 
	val_p = np.load(VAL_P_PATH)[0:val_e.shape[0],:]
	val = np.concatenate(( val_e, val_p ))
	del val_e, val_p

	X_val = val[:,0:-2]
	Y_val = val[:,-1]
	del val
	
	#~ X_train = StandardScaler().fit_transform(X_train)
	#~ X_val = StandardScaler().fit_transform(X_val)

	p = LinearDiscriminantAnalysis()
	#~ p = QuadraticDiscriminantAnalysis()
	p.fit(X_train,Y_train)
	
	predictions_binary = p.predict(X_val)			# Array of 0 and 1
	predictions_proba = p.predict_proba(X_val)[:,1]		# Array of numbers [0,1]
	
	purity = precision_score(Y_val,predictions_binary)			# Precision:  true positive / (true + false positive). Purity (how many good events in my prediction?)
	completeness = recall_score(Y_val,predictions_binary)		# Recall: true positive / (true positive + false negative). Completeness (how many good events did I find?)
	F1score = f1_score(Y_val,predictions_binary)				# Average of precision and recall
	
	l_precision, l_recall, l_thresholds = precision_recall_curve(Y_val,predictions_proba)
	
	np.save('predictions.npy',predictions_proba)
	np.save('truth.npy',Y_val)
	
	
	# 1 - precision = 1 - (TP/(TP + FP)) = (TP + FP)/(TP + FP) - (TP / (TP+FP)) = FP/(TP+FP) = FPR
	
	prec_95 = 0
	recall_95 = 0
	fscore_best = 0
	fscore_best_index = 0
	
	for i in range(len(l_precision)):
		fscore_temp = 2 * l_precision[i] * l_recall[i] / (l_precision[i]+l_recall[i])
		if fscore_temp > fscore_best:
			fscore_best = fscore_temp
			fscore_best_index = i
	
	prec_95 = l_precision[fscore_best_index]
	recall_95 = l_recall[fscore_best_index]
	
	if prec_95 < 0.6 or recall_95 < 0.1 :
		prec_95 = purity
		recall_95 = completeness
	
	for i in range(len(l_precision)):
		if l_precision[i] > 0.95 :
			if prec_95 is None:
				prec_95 = l_precision[i]
				recall_95 = l_recall[i]
			else:
				if l_precision[i] < prec_95:
					prec_95 = l_precision[i]
					recall_95 = l_recall[i]		
	
	print("Precision: ", prec_95)
	print("Recall: ", recall_95)
	
	
	elecs_p, prots_p = getClassifierScore(Y_val,predictions_proba)
	binList = [x/50 for x in range(0,51)]
	fig4 = plt.figure()
	plt.hist(elecs_p,bins=binList,label='e',alpha=0.7,histtype='step',color='green')
	plt.hist(prots_p,bins=binList,label='p',alpha=0.7,histtype='step',color='red')
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.title('Prediction histogram')
	plt.legend(loc='upper center')
	plt.grid(True)
	plt.ylim((9,1e+6))
	plt.yscale('log')
	plt.savefig('predHisto')
	plt.close(fig4)
	
	print('AUC: ',roc_auc_score(Y_val,predictions_proba))
	
		
		
	
if __name__ == '__main__' :
	
	_run()


	
