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


def _run():
	
	if not os.path.isdir('pics'): os.mkdir('pics')
	
	outdir = 'pics/LDA'
	if not os.path.isdir(outdir): os.mkdir(outdir)

	X_train, Y_train = load_training()
	X_val,Y_val = load_validation('/home/drozd/analysis/fraction1/dataset_validate_1.npy')
	
	X_train = StandardScaler().fit_transform(X_train)
	X_val = StandardScaler().fit_transform(X_val)

	#p = LinearDiscriminantAnalysis()
	p = QuadraticDiscriminantAnalysis()
	p.fit(X_train,Y_train)
	
	predictions_binary = p.predict(X_val)			# Array of 0 and 1
	predictions_proba = p.predict_proba(X_val)[:,1]		# Array of numbers [0,1]
	
	purity = precision_score(Y_val,predictions_binary)			# Precision:  true positive / (true + false positive). Purity (how many good events in my prediction?)
	completeness = recall_score(Y_val,predictions_binary)		# Recall: true positive / (true positive + false negative). Completeness (how many good events did I find?)
	F1score = f1_score(Y_val,predictions_binary)				# Average of precision and recall
	
	l_precision, l_recall, l_thresholds = precision_recall_curve(Y_val,predictions_proba)
	
	
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
	
	
	#~ electrons = p.transform( np.load('/home/drozd/analysis/fraction1/data_test_elecs_1.npy')[:,0:-2]  )
	#~ protons = p.transform( np.load('/home/drozd/analysis/fraction1/data_test_prots_1.npy')[:,0:-2]  )
	electrons = np.load('/home/drozd/analysis/data_test_elecs.npy')[:,0:-2]
	protons = np.load('/home/drozd/analysis/data_test_prots.npy')[:,0:-2]

	electrons = StandardScaler().fit_transform(electrons)
	protons = StandardScaler().fit_transform(protons)
		
	fig1 = plt.figure()
	plt.hist(p.predict_proba(electrons)[:,1],50,histtype='step',label='e')
	plt.hist(p.predict_proba(protons)[:,1],50,histtype='step',label='p')
	plt.legend(loc='best')
	plt.yscale('log')
	plt.savefig(outdir+'/QDA')
	
	fig2 = plt.figure()
	
	l_e = []
	l_p = []
	for i in range(Y_val.shape[0]):
		if Y_val[i] == 0:
			l_p.append(predictions_proba[i])
		else:
			l_e.append(predictions_proba[i])
	plt.hist(l_e,50,label='e',alpha=0.5,histtype='step',color='green')
	plt.hist(l_p,50,label='p',alpha=0.5,histtype='step',color='red')
	plt.yscale('log')
	plt.legend(loc='best')
	plt.savefig(outdir+'/QDA_hist')
	
	#plt.hist(p.predict_proba(protons),50,histtype='step',label='p')
	#plt.yscale('log')
	#plt.savefig(outdir+'/QDA_p')
	
		
		
	
if __name__ == '__main__' :
	
	_run()


	
