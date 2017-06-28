'''

Sorts variables by KS statistics, then performs training on a gradually decreasing set of variables

'''

from __future__ import print_function, division, absolute_import

import numpy as np
import time
import pickle
import sys
import os
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import heapq

from sklearn.metrics import roc_curve, roc_auc_score, precision_score, average_precision_score, precision_recall_curve, recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler, ReduceLROnPlateau
from keras.constraints import maxnorm
from keras.layers.advanced_activations import PReLU, ELU, LeakyReLU

from scipy import stats


def XY_split(fname):
	arr = np.load(fname)
	X = arr[:,0:-2]				# Last two columns are timestamp and particle ID
	Y = arr[:,-1]
	return X,Y
	
def getModel(X_train):
	model = Sequential()
	#~ model.add(Dense(40,
					#~ input_shape=(X_train.shape[1],),
					#~ kernel_initializer='uniform',
					#~ activation='relu'))
	#~ model.add(Dense(20,kernel_initializer='uniform',activation='relu'))
	#~ model.add(Dense(10,kernel_initializer='uniform',activation='relu'))
	#~ model.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
	
	model.add(Dense(200,input_shape=(X_train.shape[1],),kernel_initializer='he-uniform',activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(100,kernel_initializer='uniform',activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(50,kernel_initializer='uniform',activation='relu'))
	model.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
	
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
	return model
	
	# Best imba: {u'loss': u'binary_crossentropy', u'optimizer': u'sgd', u'dropout': 0.10000000000000001, u'metrics': [u'binary_accuracy'], u'init': u'glorot_uniform', u'acti_out': u'sigmoid', u'architecture': [300, 200, 100, 50, 1], u'batchnorm': False, u'activation': u'softplus'}
	# Best balanced: {u'loss': u'binary_crossentropy', u'optimizer': u'adam', u'dropout': 0.10000000000000001, u'metrics': [u'binary_accuracy'], u'init': u'he_uniform', u'acti_out': u'sigmoid', u'architecture': [200, 100, 50, 1], u'batchnorm': False, u'activation': u'relu'}


def run(balanced):
	
	np.random.seed(5)
	
	if balanced:	
		X_train, Y_train = XY_split('/home/drozd/analysis/fraction1/dataset_train.npy')
		X_val, Y_val = XY_split('/home/drozd/analysis/fraction1/dataset_validate_1.npy')	
		arr_elecs = np.load('/home/drozd/analysis/fraction1/data_train_elecs.npy')[:,0:-2]
		arr_prots = np.load('/home/drozd/analysis/fraction1/data_train_prots.npy')[:,0:-2]
	else:
		X_train, Y_train = XY_split('/home/drozd/analysis/dataset_train.npy')
		X_val, Y_val = XY_split('/home/drozd/analysis/dataset_validate.npy')	
		arr_elecs = np.load('/home/drozd/analysis/data_train_elecs.npy')[:,0:-2]
		arr_prots = np.load('/home/drozd/analysis/data_train_prots.npy')[:,0:-2]	
	
	X_train = StandardScaler().fit_transform(X_train)
	X_val = StandardScaler().fit_transform(X_val)
	
	
	l_pvalue = []
	l_KS = []
	for i in range(X_train.shape[1]):
		KS_statistic, p_value = stats.ks_2samp(arr_elecs[:,i],arr_prots[:,i])	# Kolmogorov-Smirnov test
		l_pvalue.append(p_value)												# If p-value is high, then the two distributions are likely the same
		l_KS.append(KS_statistic)												# If K-S statistic is high, then the two distributions are likely different.
	del arr_elecs, arr_prots
	
	if not os.path.isdir('results'):os.mkdir('results')
	if balanced: 
		if not os.path.isdir('results/balanced'): os.mkdir('results/balanced')
		outdir = './results/balanced/'
	else:
		if not os.path.isdir('results/imbalanced'): os.mkdir('results/imbalanced')
		outdir = './results/imbalanced/'
		
	############################################################################################################
	############################################################################################################
	############################################################################################################
	
	
		
	for n in range(1,X_train.shape[1]+1):
		
		touched='touch_ba_'+str(n)
		if not balanced: touched = touched.replace('ba','imba')
		
		outfile = outdir + "%02d" % (n,) + '.pick'
		
		if os.path.isfile(outfile): continue
		
		if os.path.isfile(touched): continue
		
		with open(touched,'w') as fg:
			fg.write('a')
		
		try:
			bestP_values = heapq.nlargest(n,l_pvalue)
			bestP_indices = [ l_pvalue.index(x) for x in bestP_values]		# Those are the n variables we keep in step number n
			
			X_train_new = X_train[:,bestP_indices]
			X_val_new = X_val[:,bestP_indices]
			
			model = getModel(X_train_new)
			
			history = model.fit(X_train_new,Y_train,batch_size=100,epochs=40,verbose=0,callbacks=[],validation_data=(X_val_new,Y_val))
		
			predictions_proba = model.predict(X_val_new)
			predictions_binary = np.around(predictions_proba)
			del X_train_new, X_val_new
			
			l_precision, l_recall, l_thresholds = precision_recall_curve(Y_val,predictions_proba)
			
			l_f1 = []
			for i in range(len(l_precision)):
				l_f1.append( 2*(l_precision[i] * l_recall[i])/(l_precision[i] + l_recall[i])    )
			mf1 = max(l_f1)
					
			AUC = average_precision_score(Y_val,predictions_proba)
			
			with open(outfile,'wb') as f:
				pickle.dump([n,AUC,mf1],f,protocol=2)
		except IndexError:
			os.remove(touched)
			continue
		except:
			os.remove(touched)
			raise
		
		os.remove(touched)
		del history, predictions_binary, predictions_proba, l_precision, l_recall, l_thresholds
	# end for
	
	listofPicks = glob.glob(outdir+'*.pick')
	listofPicks.sort()
	
	del X_train, X_val, Y_train, Y_val
	
	nrofvariables = []
	l_AUC = []
	for f in listofPicks:
		a,b,c = pickle.load(open(f,'rb'))
		nrofvariables.append(a)
		l_AUC.append(b)
	
	fig1 = plt.figure()
	plt.plot(nrofvariables,l_AUC,'o-')
	plt.xlabel('Nr of variables')
	plt.ylabel('Precision-Recall AUC')
	
	if balanced:
		plt.savefig('EndResult_balanced')
	else:
		plt.savefig('EndResult_imbalanced')
	
	
	
		
		
	
if __name__ == '__main__' :
	
	if len(sys.argv) == 1:
		run(True)
	else:
		if 'alse' in sys.argv[1]:
			run(False)
		else:
			run(True)
	
