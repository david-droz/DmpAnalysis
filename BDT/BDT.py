'''

BDT.py

Trains a Scikit-learn Gradient-Boosted Tree on the DAMPE electron-proton separation problem.

- Runs five times, each time on a random set of parameters
- Parameters are chosen randomly with getRandomParams(), which contains lists of possible parameters. Set of parameters saved in ./models
- Results are saved in ./results/*ID*/   where ID is a hash number corresponding to the set of parameters
- Main results are printed to screen:  Precision (purity) and Recall (completeness)



'''

from __future__ import print_function, division, absolute_import

import numpy as np
import time
import pickle
import sys
import os
import random
import hashlib
import sys

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, average_precision_score, precision_recall_curve, recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV

def ParamsID(a):
	'''
	Using hash function to build an unique identifier for each dictionary
	'''
	ID = 1
	for x in a.keys():
		ID = ID + int(hashlib.sha256(str(a[x]).encode('utf-8')).hexdigest(), 16)

	return (ID % 10**10)

def getRandomParams():
	'''
	Returns a dictionary containing all the parameters to train the Scikit BDT.
	'''
	p_lr = [1.0,0.1,0.01,0.001,0.0001]
	p_n = [25,50,100,150,200,250,300,400,500]
	p_max = [2,3,4,5,6,3]
	p_leaves = [1,10,50,100,0.001,0.0001,1,1]
	
						
	mydic = {}
	mydic['lr'] = random.choice(p_lr)
	mydic['n'] = random.choice(p_n)
	mydic['max'] = random.choice(p_max)
	mydic['leaves'] = random.choice(p_leaves)

	return mydic

def XY_split(fname):
	arr = np.load(fname)
	X = arr[:,0:-2]				# Last two columns are timestamp and particle ID
	Y = arr[:,-1]
	return X,Y
def load_training(fname='../dataset_train.npy'): return XY_split(fname)
def load_validation(fname='../dataset_validate.npy'): return XY_split(fname)
def load_test(fname='../dataset_test.npy'): return XY_split(fname)


def _run():
	

	t0 = time.time()
	
	if len(sys.argv) == 1:
		X_train, Y_train = load_training()
		X_val, Y_val = load_validation()
	else:
		X_train, Y_train = load_training(sys.argv[1])
		X_val, Y_val = load_validation(sys.argv[2])
	
	
	while True:
		params = getRandomParams()
		ID = ParamsID(params)										# Get an unique ID associated to the parameters
		if not os.path.isfile('results/' + str(ID) + '/purity_completeness.txt'):		# Check if set of params has already been tested. Don't write file yet because what's below can get interrupted
			break
	

	if not os.path.isdir('models'): os.mkdir('models')
	
	
	with open('models/params_' + str(ID) + '.pick','wb') as f:	# Save the parameters into a file determined by unique ID
		pickle.dump(params,f)
	
	clf = GradientBoostingClassifier(n_estimators=params['n'], learning_rate=params['lr'],max_depth=params['max'], min_samples_leaf=params['leaves'])
	clf.fit(X_train, Y_train)
	
	
	predictions_binary = clf.predict(X_val)			# Array of 0 and 1
	predictions_proba = clf.predict_proba(X_val)[:,1]		# Array of numbers [0,1]
	
	purity = precision_score(Y_val,predictions_binary)			# Precision:  true positive / (true + false positive). Purity (how many good events in my prediction?)
	completeness = recall_score(Y_val,predictions_binary)		# Recall: true positive / (true positive + false negative). Completeness (how many good events did I find?)
	F1score = f1_score(Y_val,predictions_binary)				# Average of precision and recall
	
	l_precision, l_recall, l_thresholds = precision_recall_curve(Y_val,predictions_proba)
	
	
	# 1 - precision = 1 - (TP/(TP + FP)) = (TP + FP)/(TP + FP) - (TP / (TP+FP)) = FP/(TP+FP) = FPR
	
	prec_95 = None
	recall_95 = None
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
	print("Iteration run time: ", time.strftime('%H:%M:%S', time.gmtime(time.time() - t0))	)
	
	if not os.path.isdir('results'): os.mkdir('results')
	if not os.path.isdir('results/' + str(ID)) : os.mkdir('results/' + str(ID))
	with open('results/' + str(ID) + '/results.pick','wb') as f:
		pickle.dump([l_precision,l_recall,l_thresholds],f)
	np.save('results/' + str(ID) + '/predictions.npy',predictions_proba)
	np.save('results/Y_Val.npy',Y_val)
	with open('results/' + str(ID) + '/purity_completeness.txt','w') as g:
		g.write("Precision: "+str(prec_95)+'\n')
		g.write("Recall: "+str(recall_95)+'\n')
	
		
		
	
if __name__ == '__main__' :
	
	for x in range(5):
		print('------------ ', x, ' ----------------')
		_run()


	
