'''

Train a model, and then plot the histogram of classifier score on several energy bins

'''

from __future__ import division, print_function, absolute_import

import numpy as np
import time
import pickle
import sys
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler, ReduceLROnPlateau
from keras.constraints import maxnorm
from keras.layers.advanced_activations import PReLU, ELU, LeakyReLU

from sklearn.metrics import roc_curve, roc_auc_score, precision_score, average_precision_score, precision_recall_curve, recall_score
from sklearn.metrics import f1_score


##############################################


def _normalise(arr):
	for i in range(arr.shape[1]):
		arr[:,i] = (arr[:,i] - np.mean(arr[:,i])) / np.std(arr[:,i])	
	return arr

def getClassifierScore(truth,pred):
	elecs = pred[truth.astype(bool)]
	prots = pred[~truth.astype(bool)]
			
	return elecs, prots

def getParticleSet(fname):
	arr = np.load(fname)
	X = arr[:,0:-2]				### Last two columns are timestamp and particle id
	Y = arr[:,-1]
	X = _normalise(X)
	r = np.concatenate(( X, Y.reshape(( Y.shape[0], 1 )) ) , axis=1)
	del arr, X, Y
	return r

	
def run():
	
	train_e = getParticleSet('/home/drozd/analysis/data_train_elecs.npy')
	train_p = getParticleSet('/home/drozd/analysis/data_train_prots.npy')
	train = np.concatenate(( train_e, train_p ))
	np.random.shuffle(train)
	
	X_train = train[:,0:-1]
	Y_train = train[:,-1]
	del train_e,train_p, train

	val_e = np.concatenate((getParticleSet('/home/drozd/analysis/fraction1/data_validate_elecs_1.npy') , getParticleSet('/home/drozd/analysis/fraction1/data_test_elecs_1.npy') ))
	val_p = np.concatenate((getParticleSet('/home/drozd/analysis/fraction1/data_validate_prots_1.npy') , getParticleSet('/home/drozd/analysis/fraction1/data_test_prots_1.npy') ))
	
	val = np.concatenate(( val_e, val_p ))
	np.random.shuffle(val)
	
	X_val = val[:,0:-1]
	Y_val = val[:,-1]
	
	del val_e, val_p, val
	
	modelName = 'myModel.model'
	
	if not os.path.isfile(modelName):
		model = Sequential()
		model.add(Dense(300,input_shape=(X_train.shape[1],),kernel_initializer='he_uniform',activation='relu'))
		model.add(Dropout(0.1))
		model.add(Dense(150,kernel_initializer='he_uniform',activation='relu'))
		model.add(Dropout(0.1))
		model.add(Dense(70,kernel_initializer='he_uniform',activation='relu'))
		model.add(Dropout(0.1))
		model.add(Dense(1,kernel_initializer='he_uniform',activation='sigmoid'))
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
	
		rdlronplt = ReduceLROnPlateau(monitor='loss',patience=3,min_lr=0.001)
		earl = EarlyStopping(monitor='loss',min_delta=0.0001,patience=5)
		callbacks = [rdlronplt,earl]
		
		history = model.fit(X_train,Y_train,batch_size=150,epochs=40,verbose=0,callbacks=callbacks,validation_data=(X_val,Y_val))
		model.save(modelName)
	else:
		model = load_model(modelName)
	
	# --------------------------------
	
	bin_edges = np.unique( np.concatenate(( np.logspace(5,6,7) , np.logspace(6,6.7,3) )) )
	
	for i in range(bin_edges.shape[0]-1):
		
		graphName = 'predHisto_'+str(int(bin_edges[i]))+'-'+str(int(bin_edges[i+1]))
		if os.path.isfile(graphName+'.png'):
			continue
		
		arr_e = np.concatenate((np.load('/home/drozd/analysis/fraction1/data_validate_elecs_1.npy'),np.load('/home/drozd/analysis/fraction1/data_test_elecs_1.npy') ))
		X = arr[:,0:-2]
		X = X[ X[:,30] > bin_edges[i] ] 
		X = X[ X[:,30] < bin_edges[i+1] ]
		Y = arr[:,-1]
		X = _normalise(X)
		arr_e = np.concatenate(( X, Y.reshape(( Y.shape[0], 1 )) ) , axis=1)
		del X, Y
		
		arr_p = np.concatenate((np.load('/home/drozd/analysis/fraction1/data_validate_prots_1.npy'),np.load('/home/drozd/analysis/fraction1/data_test_prots_1.npy') ))
		X = arr[:,0:-2]
		X = X[ X[:,30] > bin_edges[i] ] 
		X = X[ X[:,30] < bin_edges[i+1] ]
		Y = arr[:,-1]
		X = _normalise(X)
		arr_p = np.concatenate(( X, Y.reshape(( Y.shape[0], 1 )) ) , axis=1)
		del X, Y
		
		bigarr = np.concatenate(( arr_e, arr_p ))
		del arr_e, arr_p
		np.random.shuffle(bigarr)
		X_val = bigarr[:,0:-1]
		Y_val = bigarr[:,-1]
		del bigarr
		
		predictions = model.predict(X_val)
		elecs_p, prots_p = getClassifierScore(Y_val,predictions)	
		
		Nbins = 50
		binList = [x/Nbins for x in range(0,Nbins+1)]		
		fig = plt.figure()
		plt.hist(elecs_p,bins=binList,label='e',alpha=0.7,histtype='step',color='green')
		plt.hist(prots_p,bins=binList,label='p',alpha=0.7,histtype='step',color='red')
		plt.xlabel('Classifier score')
		plt.ylabel('Number of events')
		plt.title(str(int(bin_edges[i]))+' - '+str(int(bin_edges[i+1])))
		plt.legend(loc='upper center')
		plt.yscale('log')
		plt.savefig(graphName)
		plt.close(fig)
	
	
if __name__ == '__main__' :
	
	run()

