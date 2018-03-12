'''

This program was used to run several normalisation methods and compare them with each other. The methods were:

	- Standardisation: move to mean 0 and deviation 1
	- Shift: only move to mean 0, no scaling
	- scikit-learn standard scaler
	- Max to 1: divide by maximum value (~ scale to [0;1])
	- No normalisation at all
	
	Each could be applied on both training and validation. This could also be applied by class (i.e. norm e⁻ and p independently)
	or on the whole sample. Considering that on validation, I do not know the actual labels.
	
	Extensive testing led to the following methods being best:
		- Train: max to 1, combined. Validation: max to 1, combined
		- Train: standardisation, combined, then saving the values to apply them on validation.
	First one being slightly better, but keeping in mind the second one
	
	
	Then the code got adapted to produce a plot of background fraction as a function of electron/proton ratio in validation
		
	Ran on :
	baobab:	/home/drozd/analysis/norm/
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

from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, average_precision_score, precision_recall_curve, recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler, ReduceLROnPlateau
from keras.constraints import maxnorm
from keras.layers.advanced_activations import PReLU, ELU, LeakyReLU

from scipy import stats


def getParticleSet(fname):
	arr = np.load(fname)
	X = arr[:,0:-2]				### Last two columns are timestamp and particle id
	Y = arr[:,-1]
	X = _normalise(X)
	r = np.concatenate(( X, Y.reshape(( Y.shape[0], 1 )) ) , axis=1)
	del arr, X, Y
	return r
	
def getClassifierScore(truth,pred):
	elecs = pred[truth.astype(bool)]
	prots = pred[~truth.astype(bool)]
			
	return elecs, prots
	
def getModel(X_train):
	model = Sequential()
	model.add(Dense(300,input_shape=(X_train.shape[1],),kernel_initializer='he_uniform',activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(150,kernel_initializer='he_uniform',activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(70,kernel_initializer='he_uniform',activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(1,kernel_initializer='he_uniform',activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
	return model
	

def _normalise(arr):
	for i in range(arr.shape[1]):
		arr[:,i] = (arr[:,i] - np.mean(arr[:,i])) / np.std(arr[:,i])	
	return arr

def run(n=1):
	
	n_epochs = 50
	
	figTitle = "Train-MaxTo1_Test-MaxTo1_ratio"+str(n)+'_epochs'+str(n_epochs)
	figName = './imba/'+figTitle
	
	
	
	modelName = 'model'+str(n_epochs)+'.h5'
	
	
	train_e = np.load('/home/drozd/analysis/newData/data_train_elecs_under_1.npy')
	train_p = np.load('/home/drozd/analysis/newData/data_train_prots_under_1.npy')
	train = np.concatenate(( train_e, train_p ))
	np.random.shuffle(train)
	
	
	X_train = train[:,0:-2]
	X_max = X_train.max(axis=0)
	
	X_train = X_train/X_max
	
	Y_train = train[:,-1]

	del train_e,train_p, train
		
	val_p = np.load('/home/drozd/analysis/newData/data_validate_prots_under_1.npy')[0:int(7.5e+5),:]
	val_e = np.load('/home/drozd/analysis/newData/data_validate_elecs_under_1.npy')[0:int(val_p.shape[0]/n),:]
	val = np.concatenate(( val_e, val_p ))
	
	X_val = val[:,0:-2]/(val[:,0:-2].max(axis=0))
	
	Y_val = val[:,-1]
	
	del val_e, val_p, val
	
	
	
	#####
	
	if not os.path.isfile(modelName):
		model = getModel(X_train)
		history = model.fit(X_train,Y_train,batch_size=150,epochs=n_epochs,verbose=0,callbacks=[],validation_split=0.1)
		model.save(modelName)
	else:
		model = load_model(modelName)
	
	del X_train, Y_train
	
	
	# Prediction histogram
	
	predictions_proba = model.predict(X_val)
	
	elecs_p, prots_p = getClassifierScore(Y_val,predictions_proba)
	
	binList = [x/50 for x in range(0,51)]
	fig4 = plt.figure()
	plt.hist(elecs_p,bins=binList,label='e',alpha=0.7,histtype='step',color='green')
	plt.hist(prots_p,bins=binList,label='p',alpha=0.7,histtype='step',color='red')
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.title(figTitle)
	plt.legend(loc='upper center')
	plt.yscale('log')
	#~ plt.savefig(figName)
	plt.close(fig4)
	
	n_elecs_top = elecs_p[ elecs_p >= 0.95 ].shape[0]
	n_prots_top = prots_p[ prots_p >= 0.95 ].shape[0]
	rejection = n_elecs_top / n_prots_top
	
	return rejection
	
	
		
	############################################################################################################
	############################################################################################################
	############################################################################################################
	
		
	
if __name__ == '__main__' :
	
	
	fracs = [1,2,5,10,20,50,100,200,500]
	
	rej = []
	
	for n in fracs:
		rej.append(run(n))
		
	fig = plt.figure()
	plt.plot(fracs,rej,'o')
	plt.xlabel('proton/electron ratio')
	plt.ylabel('electron/proton in signal region (0.95)')
	plt.yscale('log')
	plt.xscale('log')
	plt.savefig('rejection')
	
	with open('rejectionData.pickle','wb') as f:
		pickle.dump([fracs,rej],f)
