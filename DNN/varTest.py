'''

Test DNN performances in the case of a full-variable run or in the case of only a couple of variables

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

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler, ReduceLROnPlateau
from keras.constraints import maxnorm
from keras.layers.advanced_activations import PReLU, ELU, LeakyReLU

def XY_split(fname):
	arr = np.load(fname)
	X = arr[:,0:-2]				# Last two columns are timestamp and particle ID
	Y = arr[:,-1]
	return X,Y

def run(fullVar):
	
	t0 = time.time()
	
	balanced = False
	
	if balanced:	
		X_train, Y_train = XY_split('/home/drozd/analysis/fraction1/dataset_train.npy')
		X_val, Y_val = XY_split('/home/drozd/analysis/fraction1/dataset_validate_1.npy')	
	else:
		X_train, Y_train = XY_split('/home/drozd/analysis/dataset_train.npy')
		X_val, Y_val = XY_split('/home/drozd/analysis/dataset_validate.npy')		
	
	if not fullVar:
		varsToRemove = [6,10,32,33,38,39,41,42,44,47,48,49,50,51,53,54,56,57]
		varsToKeep = [i for i in range(X_train.shape[1]) if not i in varsToRemove]
		
		X_train = X_train[:,varsToKeep]
		X_val = X_val[:,varsToKeep]
		
	
	X_train = StandardScaler().fit_transform(X_train)
	X_val = StandardScaler().fit_transform(X_val)
	
	if fullVar:
		print('----- All variables -----')
	else:
		print('----- Selected variables -----')
	
	model = Sequential()
	model.add(Dense(40,
					input_shape=(X_train.shape[1],),
					kernel_initializer='uniform',
					activation='relu'))
	model.add(Dense(20,kernel_initializer='uniform',activation='relu'))
	model.add(Dense(10,kernel_initializer='uniform',activation='relu'))
	model.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

	callbacks = []
	
	history = model.fit(X_train,Y_train,batch_size=100,epochs=50,verbose=0,callbacks=callbacks,validation_data=(X_val,Y_val))
	
	predictions_binary = np.around(model.predict(X_val))		# Array of 0 and 1
	predictions_proba = model.predict_proba(X_val)				# Array of numbers [0,1]
	
	purity = precision_score(Y_val,predictions_binary)			# Precision:  true positive / (true + false positive). Purity (how many good events in my prediction?)
	completeness = recall_score(Y_val,predictions_binary)		# Recall: true positive / (true positive + false negative). Completeness (how many good events did I find?)
	F1score = f1_score(Y_val,predictions_binary)				# Average of precision and recall
	
	l_precision, l_recall, l_thresholds = precision_recall_curve(Y_val,predictions_proba)
	

	# Precision and Recall that maximise F1 score
	l_f1 = []
	for i in range(len(l_precision)):
		l_f1.append( 2*(l_precision[i] * l_recall[i])/(l_precision[i] + l_recall[i])    )
	mf1 = max(l_f1)
	for i in range(len(l_f1)):
		if l_f1[i] == mf1:
			prec_95 = l_precision[i]
			recall_95 = l_recall[i]
			
	AUC = average_precision_score(Y_val,predictions_proba)
	
	print("-- On best F1:")				
	print("Precision:", prec_95)
	print("Recall:", recall_95)
	print("Max F1:", mf1)
	print("AUC:", AUC)
	
	prec_95 = 0
	rc_95 = 0
	f1_95 = 0
	
	for i in range(len(l_precision)):
		if l_precision[i] > 0.95:
			temp_f1 = 2*(l_precision[i]*l_recall[i])/(l_precision[i]+l_recall[i])
			if temp_f1 > f1_95:
				prec_95 = l_precision[i]
				rc_95 = l_recall[i]
				f1_95 = temp_f1
	
	print('-- On 95%')
	print('F1:',f1_95)
	print('Precision',prec_95)
	print('Recall',rc_95)
	
	del X_train, X_val, Y_train, Y_val, history, predictions_binary, predictions_proba
	
		
		
	
if __name__ == '__main__' :
	
	run(True,True)
	run(True,False)
	run(False,True)
	run(False,False)
