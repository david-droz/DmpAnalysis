'''

Check model persistence with Keras

Result: It works!

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
	
	model = Sequential()
	model.add(Dense(100,input_shape=(X_train.shape[1],),kernel_initializer='he_uniform',activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(50,kernel_initializer='he_uniform',activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(1,kernel_initializer='he_uniform',activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

	callbacks = []
	
	history = model.fit(X_train,Y_train,batch_size=150,epochs=20,verbose=0,callbacks=callbacks,validation_data=(X_val,Y_val))
	
	# --------------------------------
	
	model.save("model.model")
	np.save("X_val",X_val)
	np.save("Y_val",Y_val)
	
	predictions_balanced = model.predict(X_val)
	
	del X_val, X_train
	
	print("----- AUC -----")
	print(average_precision_score(Y_val,predictions_balanced))
	print("----- F1 -----")
	print(f1_score(Y_val,np.around(predictions_balanced)))
	print("----- Precision/Recall -----")
	print(precision_score(Y_val,np.around(predictions_balanced)), " / ", recall_score(Y_val,np.around(predictions_balanced)))
		
	Nbins = 50
	binList = [x/Nbins for x in range(0,Nbins+1)]
	
	elecs_b, prots_b = getClassifierScore(Y_val,predictions_balanced)
	fig4 = plt.figure()
	plt.hist(elecs_b,bins=binList,label='e',alpha=0.7,histtype='step',color='green')
	plt.hist(prots_b,bins=binList,label='p',alpha=0.7,histtype='step',color='red')
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.title('Balanced validation set')
	plt.legend(loc='best')
	plt.yscale('log')
	plt.savefig('predHisto_before')
	
	
	
if __name__ == '__main__' :
	
	run()
	
	DNN = load_model('model.model')
	X = np.load("X_val.npy")
	Y = np.load("Y_val.npy")
	
	pred = DNN.predict(X)
	
	print("----- AUC -----")
	print(average_precision_score(Y,pred))
	print("----- F1 -----")
	print(f1_score(Y,np.around(pred)))
	print("----- Precision/Recall -----")
	print(precision_score(Y,np.around(pred)), " / ", recall_score(Y,np.around(pred)))
		
	Nbins = 50
	binList = [x/Nbins for x in range(0,Nbins+1)]
	
	elec, prot = getClassifierScore(Y,pred)
	fig4 = plt.figure()
	plt.hist(elec,bins=binList,label='e',alpha=0.7,histtype='step',color='green')
	plt.hist(prot,bins=binList,label='p',alpha=0.7,histtype='step',color='red')
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.title('Balanced validation set')
	plt.legend(loc='best')
	plt.yscale('log')
	plt.savefig('predHisto_after')

