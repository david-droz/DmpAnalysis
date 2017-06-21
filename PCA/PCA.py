'''

PCA.py

PCA decomposition of DAMPE data



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

from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, average_precision_score, precision_recall_curve, recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

def XY_split(fname):
	arr = np.load(fname)
	X = arr[:,0:-2]				# Last two columns are timestamp and particle ID
	Y = arr[:,-1]
	return X,Y
def load_training(fname='../dataset_train.npy'): return XY_split(fname)
def load_validation(fname='../dataset_validate.npy'): return XY_split(fname)
def load_test(fname='../dataset_test.npy'): return XY_split(fname)


def _run(n):
	
	if not os.path.isdir('pics'): os.mkdir('pics')
	
	outdir = 'pics/'+str(n)+'/'
	if not os.path.isdir(outdir): os.mkdir(outdir)

	X_train, Y_train = load_training()
	X_train = StandardScaler().fit_transform(X_train)

	p = PCA(n_components=n)
	p.fit(X_train)
	
	electrons = p.transform( StandardScaler().fit_transform(np.load('/home/drozd/analysis/fraction1/data_test_elecs_1.npy')[:,0:-2])  )
	protons = p.transform( StandardScaler().fit_transform(np.load('/home/drozd/analysis/fraction1/data_test_prots_1.npy')[:,0:-2])  )
	
	for i in range(n):
		fig1 = plt.figure()
		plt.hist(electrons[:,i],50,histtype='step',label='e')
		plt.hist(protons[:,i],50,histtype='step',label='p')
		plt.legend(loc='best')
		plt.title('PCA - PC component ' + str(i))
		plt.savefig(outdir+'pc'+str(i))
		
	X_train_new = p.transform(X_train)
	
	new_out = outdir + 'train_'
	for i in range(n):
		l_e = []
		l_p = []
		for j in range(len(Y_train)):
			if Y_train[j] == 1:
				l_e.append(X_train[j,i])
			else:
				l_p.append(X_train[j,i])
		
		fig2 = plt.figure()
		plt.hist(l_e,50,histtype='step',label='e')
		plt.hist(l_p,50,histtype='step',label='p')
		plt.legend(loc='best')
		plt.title('PCA - PC component ' + str(i))
		plt.savefig(new_out+'pc'+str(i))
		
		
	print(p.explained_variance_ratio_)
	
		
		
	
if __name__ == '__main__' :
	
	_run(9)


	
