'''

For the same trained model, compare results in imbalanced bootstrap versus imbalanced no bootstrap

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

from keras.models import Sequential
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

	
def run(bootStrap):
	
	if bootStrap:
		train_e = getParticleSet('/home/drozd/analysis/data_train_elecs.npy')
		train_p = getParticleSet('/home/drozd/analysis/data_train_prots.npy')
	else:
		train_e = getParticleSet('/home/drozd/analysis/NoBootStrap/data_train_elecs_under_100.npy')
		train_p = getParticleSet('/home/drozd/analysis/NoBootStrap/data_train_prots_under_100.npy')
	train = np.concatenate(( train_e, train_p ))
	np.random.shuffle(train)
	
	X_train = train[:,0:-1]
	Y_train = train[:,-1]
	del train_e,train_p, train

	if bootStrap:
		val_e = getParticleSet('/home/drozd/analysis/data_validate_elecs.npy') 
		val_p = getParticleSet('/home/drozd/analysis/data_validate_prots.npy') 
	else:
		val_e = getParticleSet('/home/drozd/analysis/NoBootStrap/data_validate_elecs_under_100.npy') 
		val_p = getParticleSet('/home/drozd/analysis/NoBootStrap/data_validate_prots_under_100.npy') 
	
	val = np.concatenate(( val_e, val_p ))
	np.random.shuffle(val)
	
	X_val = val[:,0:-1]
	Y_val = val[:,-1]
	
	
	del val_e, val_p, val
	
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
	
	# --------------------------------
	
	predictions_balanced = model.predict(X_val)
	
	del X_val, X_train
	
	AUC = average_precision_score(Y_val,predictions_balanced)
	F = f1_score(Y_val,np.around(predictions_balanced))
	PREC = precision_score(Y_val,np.around(predictions_balanced))
	REC = recall_score(Y_val,np.around(predictions_balanced))
	
	ROC_AUC = roc_auc_score(Y_val,np.around(predictions_balanced))
	
	return AUC, F, PREC, REC,ROC_AUC
	
	
if __name__ == '__main__' :
	
	AUC_1, F_1, PREC_1, REC_1 = [[],[],[],[]]
	AUC_0, F_0, PREC_0, REC_0 = [[],[],[],[]]
	
	ROCAUC_0, ROCAUC_1 = [[],[]]
	
	# Normalise to 1
	for x in range(5):
		a = run(True)
		AUC_1.append(a[0])
		F_1.append(a[1])
		PREC_1.append(a[2])
		REC_1.append(a[3])
		ROCAUC_1.append(a[4])
	
	# Normalise to 0
	for x in range(5):
		run(False)
		AUC_0.append(a[0])
		F_0.append(a[1])
		PREC_0.append(a[2])
		REC_0.append(a[3])
		ROCAUC_0.append(a[4])
		
	print('----- BootStrap -----')
	print('ROC AUC:', np.mean(ROCAUC_1), '+/-', np.std(ROCAUC_1))
	print('AUC:', np.mean(AUC_1), '+/-', np.std(AUC_1))
	print('F1:', np.mean(F_1), '+/-', np.std(F_1))
	print('Precision:', np.mean(PREC_1), '+/-', np.std(PREC_1))
	print('Recall:', np.mean(REC_1), '+/-', np.std(REC_1))
	
	print('----- No BootStrap -----')
	print('ROC AUC:', np.mean(ROCAUC_0), '+/-', np.std(ROCAUC_0))
	print('AUC:', np.mean(AUC_0), '+/-', np.std(AUC_0))
	print('F1:', np.mean(F_0), '+/-', np.std(F_0))
	print('Precision:', np.mean(PREC_0), '+/-', np.std(PREC_0))
	print('Recall:', np.mean(REC_0), '+/-', np.std(REC_0))
