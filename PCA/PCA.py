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

from mpl_toolkits.mplot3d import Axes3D

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
	
	for i in range(14):
		X_train[:,i] = X_train[:,i]/X_train[:,30]
	X_train = X_train[:,0:28]
	
	X_train = StandardScaler().fit_transform(X_train)
	
	
	X_t_elecs = np.load('../data_train_elecs.npy')
	for i in range(14): X_t_elecs[:,i] = X_t_elecs[:,i]/X_t_elecs[:,30]
	X_t_elecs = X_t_elecs[0:28]
	
	p = PCA(n_components=n)
	#~ p.fit(X_train)
	p.fit(X_t_elecs)
	
	
	electrons = np.load('/home/drozd/analysis/fraction1/data_test_elecs_1.npy')
	protons = np.load('/home/drozd/analysis/fraction1/data_test_prots_1.npy')
	
	for d in [electrons,protons]:
		for i in range(14):
			d[:,i] = d[:,i]/d[:,30]
		d = d[:,0:28]
		d = p.transform( StandardScaler().fit_transform(d) )
	
	for i in range(n):
		fig1 = plt.figure()
		plt.hist(electrons[:,i],50,histtype='step',label='e')
		plt.hist(protons[:,i],50,histtype='step',label='p')
		plt.legend(loc='best')
		plt.title('PCA - PC component ' + str(i))
		plt.savefig(outdir+'test_split_pc'+str(i))
		plt.close(fig1)
		
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
		plt.close(fig2)
		
	
	X_validate_all = np.load('/home/drozd/analysis/fraction1/dataset_validate_1.npy')
	for i in range(14):
		X_validate_all[:,i] = X_validate_all[:,i]/X_validate_all[:,30]
	X_validate_all = X_validate_all[:,0:28]
	X_validate_all = p.transform( StandardScaler().fit_transform(X_validate_all) )
	
	Y_val = np.load('/home/drozd/analysis/fraction1/dataset_validate_1.npy')[:,-1]
		
	new_out = outdir + 'validate_'
	for i in range(n):
		l_e = []
		l_p = []
		for j in range(len(Y_val)):
			if Y_val[j] == 1:
				l_e.append(X_validate_all[j,i])
			else:
				l_p.append(X_validate_all[j,i])
		
		fig3 = plt.figure()
		plt.hist(l_e,50,histtype='step',label='e')
		plt.hist(l_p,50,histtype='step',label='p')
		plt.legend(loc='best')
		plt.title('PCA - PC component ' + str(i))
		plt.savefig(new_out+'pc'+str(i))
		plt.close(fig3)
	
	l_e_1 = []
	l_e_2 = []
	l_e_3 = []
	l_p_1 = []
	l_p_2 = []
	l_p_3 = []
	for j in range(len(Y_val)):
		if Y_val[j] == 1:
			l_e_1.append(X_validate_all[j,0])
			l_e_2.append(X_validate_all[j,1])
			l_e_3.append(X_validate_all[j,2])
		else:
			l_p_1.append(X_validate_all[j,0])
			l_p_2.append(X_validate_all[j,1])
			l_p_3.append(X_validate_all[j,2])
	fig4 = plt.figure()
	plt.scatter(l_e_1,l_e_2,label='e',alpha=0.3)
	plt.scatter(l_p_1,l_p_2,label='p',alpha=0.3)
	plt.xlabel("1st eigenvector")
	plt.ylabel("2nd eigenvector")
	plt.legend(loc='best')
	plt.savefig(outdir+'val2D')
	plt.close(fig4)
	
	
	
	fig5 = plt.figure()
	ax = Axes3D(fig5)
	ax.scatter(l_e_1,l_e_2,l_e_3,label='e')
	ax.scatter(l_p_1,l_p_2,l_p_3,label='p')
	ax.set_title("First three PCA directions")
	ax.set_xlabel("1st eigenvector")
	ax.w_xaxis.set_ticklabels([])
	ax.set_ylabel("2nd eigenvector")
	ax.w_yaxis.set_ticklabels([])
	ax.set_zlabel("3rd eigenvector")
	ax.w_zaxis.set_ticklabels([])
	plt.legend(loc='best')
	plt.savefig(outdir+'val3D')
	plt.close(fig5)

	del X_validate_all, Y_val
	del electrons, protons
	
	X_test_all = np.load('/home/drozd/analysis/fraction1/dataset_test_1.npy')
	for i in range(14):
		X_test_all[:,i] = X_test_all[:,i]/X_test_all[:,30]
	X_test_all = X_test_all[:,0:28]
	X_test_all = p.transform( StandardScaler().fit_transform(X_test_all) )
		
	Y_test = np.load('/home/drozd/analysis/fraction1/dataset_test_1.npy')[:,-1]
	new_out = outdir + 'test_'
	for i in range(n):
		l_e = []
		l_p = []
		for j in range(len(Y_test)):
			if Y_test[j] == 1:
				l_e.append(X_test_all[j,i])
			else:
				l_p.append(X_test_all[j,i])
		
		fig3 = plt.figure()
		plt.hist(l_e,50,histtype='step',label='e')
		plt.hist(l_p,50,histtype='step',label='p')
		plt.legend(loc='best')
		plt.title('PCA - PC component ' + str(i))
		plt.savefig(new_out+'pc'+str(i))
		plt.close(fig3)
	
	l_e_1 = []
	l_e_3 = []
	l_p_1 = []
	l_p_3 = []
	for j in range(len(Y_test)):
		if Y_test[j] == 1:
			l_e_1.append(X_test_all[j,1])
			l_e_3.append(X_test_all[j,3])
		else:
			l_p_1.append(X_test_all[j,1])
			l_p_3.append(X_test_all[j,3])
	fig4 = plt.figure()
	plt.scatter(l_e_1,l_e_3,label='e',alpha=0.3)
	plt.scatter(l_p_1,l_p_3,label='p',alpha=0.3)
	plt.xlabel("2nd eigenvector")
	plt.ylabel("4th eigenvector")
	plt.legend(loc='best')
	plt.savefig(outdir+'test2D')
	plt.close(fig4)
		
	print(p.explained_variance_ratio_)
	
	print(sum(p.explained_variance_ratio_))
	
	
		
		
	
if __name__ == '__main__' :
	
	_run(8)


	
