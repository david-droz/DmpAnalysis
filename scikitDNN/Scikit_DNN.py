'''

Scikit_DNN.py

Trains a Scikit-learn deep neural network (Multi-Layer Perceptron) on the DAMPE electron-proton separation problem.

- Runs two times, each time on a random set of parameters
- Parameters are chosen randomly with getRandomParams(), which contains lists of possible parameters. Set of parameters saved in ./models
- Results are saved in ./results/*ID*/   where ID is a hash number corresponding to the set of parameters
- Main results are printed to screen:  Precision (purity) and Recall (completeness)

Looks for datasets in ../dataset_train.npy ; ../dataset_validate.npy
	Can change that by suppling them as arguments to the program

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

from scipy.stats import randint as sp_randint

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, average_precision_score, precision_recall_curve, recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

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
	Returns a dictionary containing all the parameters to train the Scikit DNN.
	'''
	
	p_lr = ['constant', 'invscaling', 'adaptive']
	p_n = [(50,), (100,), (200,), (300,), (400,), (500,), (100,50), (30,15,7), (100,50,25), (200,100)]
	p_epoch = [50, 100, 200, 300,400]
	p_alpha = [1,1e-1,1e-2,1e-3, 1e-4,1e-5]
	p_algo = ['adam','sgd','lbfgs']
					
	mydic = {}
	mydic['lr'] = random.choice(p_lr)
	mydic['n'] = random.choice(p_n)
	mydic['epoch'] = random.choice(p_epoch)
	mydic['alpha'] = random.choice(p_alpha)
	mydic['algo'] = random.choice(p_algo)

	return mydic

def getParticleSet(fname):
	arr = np.load(fname)
	X = arr[:,0:-2]				### Last two columns are timestamp and particle id
	Y = arr[:,-1]
	X = StandardScaler().fit_transform(X)
	r = np.concatenate(( X, Y.reshape(( Y.shape[0], 1 )) ) , axis=1)
	del arr, X, Y
	return r


def _run():

	t0 = time.time()
	
	if len(sys.argv) == 1:
		train_e = getParticleSet('/home/drozd/analysis/data_train_elecs.npy')
		train_p = getParticleSet('/home/drozd/analysis/data_train_prots.npy')
		val_e = getParticleSet('/home/drozd/analysis/data_validate_elecs.npy') 
		val_p = getParticleSet('/home/drozd/analysis/data_validate_prots.npy') 
	else:
		train_e = getParticleSet(sys.argv[1])
		train_p = getParticleSet(sys.argv[2])
		val_e = getParticleSet(sys.argv[3]) 
		val_p = getParticleSet(sys.argv[4])
		
	train = np.concatenate(( train_e, train_p ))
	np.random.shuffle(train)
	X_train = train[:,0:-1]
	Y_train = train[:,-1]
	del train_e,train_p, train
	
	val = np.concatenate(( val_e, val_p ))
	np.random.shuffle(val)
	X_val = val[:,0:-1]
	Y_val = val[:,-1]
	del val_e, val_p, val
	
	while True:
		params = getRandomParams()
		ID = ParamsID(params)										# Get an unique ID associated to the parameters
		if not os.path.isfile('results/' + str(ID) + '/purity_completeness.txt'):		# Check if set of params has already been tested. Don't write file yet because what's below can get interrupted
			break

	if not os.path.isdir('models'): os.mkdir('models')
	
	with open('models/params_' + str(ID) + '.pick','wb') as f:	# Save the parameters into a file determined by unique ID
		pickle.dump(params,f,protocol=2)
	
	clf = MLPClassifier(hidden_layer_sizes=params['n'],solver=params['algo'],alpha=params['alpha'],learning_rate=params['lr'],max_iter=params['epoch'] )
	
	clf.fit(X_train, Y_train)
	
	
	predictions_binary = clf.predict(X_val)			# Array of 0 and 1
	predictions_proba = clf.predict_proba(X_val)[:,1]		# Array of numbers [0,1]
	
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
					
	print("Precision: ", prec_95)
	print("Recall: ", recall_95)
	print("Iteration run time: ", time.strftime('%H:%M:%S', time.gmtime(time.time() - t0))	)
	
	if not os.path.isdir('results'): os.mkdir('results')
	if not os.path.isdir('results/' + str(ID)) : os.mkdir('results/' + str(ID))
	with open('results/' + str(ID) + '/results.pick','wb') as f:
		pickle.dump([l_precision,l_recall,l_thresholds],f,protocol=2)
	np.save('results/' + str(ID) + '/predictions.npy',predictions_proba)
	np.save('results/Y_Val.npy',Y_val)
	with open('results/' + str(ID) + '/purity_completeness.txt','w') as g:
		g.write("Precision: "+str(prec_95)+'\n')
		g.write("Recall: "+str(recall_95)+'\n')
		
	del X_train, X_val, Y_train, Y_val, predictions_binary, predictions_proba
	
		
		
	
if __name__ == '__main__' :
	
	for x in range(2):
		print('------------ ', x, ' ----------------')
		_run()

	
