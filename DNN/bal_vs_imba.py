'''

For the same trained model, compare results in balanced vs imbalanced

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

def XY_split(fname):
	arr = np.load(fname)
	X = arr[:,0:-2]				### Last two columns are timestamp and particle id
	Y = arr[:,-1]
	return X,Y

def _normalise(arr):
	for i in range(arr.shape[1]):
		if np.all(arr[:,i] > 0) :
			arr[:,i] = (arr[:,i] - np.mean(arr[:,i]) + 1.) / np.std(arr[:,i])		# Mean = 1 if all values are strictly positive (from paper)
		else:
			arr[:,i] = (arr[:,i] - np.mean(arr[:,i])) / np.std(arr[:,i])	
	return arr
	
def run():
	
	X_train, Y_train = XY_split('/home/drozd/analysis/dataset_train.npy')
	X_val, Y_val = XY_split('/home/drozd/analysis/fraction1/dataset_validate_1.npy')
	X_val_imba, Y_val_imba = XY_split('/home/drozd/analysis/dataset_validate.npy')
	
	X_train = _normalise(X_train)
	X_val = _normalise(X_val)
	X_val_imba = _normalise(X_val_imba)
	
	model = Sequential()
	model.add(Dense(200,input_shape=(X_train.shape[1],),kernel_initializer='he_uniform',activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(100,kernel_initializer='he_uniform',activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(50,kernel_initializer='he_uniform',activation='relu'))
	model.add(Dense(1,kernel_initializer='he_uniform',activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

	callbacks = []
	
	history = model.fit(X_train,Y_train,batch_size=100,epochs=25,verbose=2,callbacks=callbacks,validation_data=(X_val,Y_val))
	
	
	predictions_balanced = model.predict(X_val)
	predictions_imba = model.predict(X_val_imba)
	
	l_precision_b, l_recall_b, l_thresholds_b = precision_recall_curve(Y_val,predictions_balanced)
	l_precision_i, l_recall_i, l_thresholds_i = precision_recall_curve(Y_val_imba,predictions_imba)
	
	l_fpr_b, l_tpr_b, l_roc_thresholds_b = roc_curve(Y_val,predictions_balanced)
	l_fpr_i, l_tpr_i, l_roc_thresholds_i = roc_curve(Y_val_imba,predictions_imba)
	
	fig1 = plt.figure()
	plt.plot(l_precision_b, l_recall_b,label='balanced')
	plt.plot(l_precision_i, l_recall_i,label='imbalanced')
	plt.xlabel('Precision')
	plt.ylabel('Recall')
	plt.legend(loc='best')
	plt.savefig('PR')
	
	fig2 = plt.figure()
	plt.plot(l_fpr_b, l_tpr_b,label='balanced')
	plt.plot(l_fpr_i, l_tpr_i, l_recall_i,label='imbalanced')
	plt.xlabel('False Positive')
	plt.ylabel('True Positive')
	plt.legend(loc='best')
	plt.savefig('ROC')
	



if __name__ == '__main__' :
	
	run()

