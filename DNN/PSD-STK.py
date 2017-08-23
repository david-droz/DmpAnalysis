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

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler, ReduceLROnPlateau
from keras.constraints import maxnorm
from keras.layers.advanced_activations import PReLU, ELU, LeakyReLU
	
def getClassifierScore(truth,pred):
	elecs = pred[truth.astype(bool)]
	prots = pred[~truth.astype(bool)]
			
	return elecs, prots
	
def getModel(X_train):
	model = Sequential()
	model.add(Dense(300,input_shape=(X_train.shape[1],),kernel_initializer='uniform',activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(150,kernel_initializer='uniform',activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(75,kernel_initializer='uniform',activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
	return model


def run(subDetector):
	
	train_e = np.load('/home/drozd/analysis/newData/data_train_elecs_under_1.npy')
	train_p = np.load('/home/drozd/analysis/newData/data_train_prots_under_1.npy')
	
	train = np.concatenate(( train_e, train_p ))
	np.random.shuffle(train)
	
	X_train = train[:,0:-1]
	X_max = X_train.max(axis=0)
	
	X_train = X_train / X_max
	
	if subDetector == 'STK' :
		X_train = X_train[:,40:58]
	else:
		X_train = X_train[:,32:40]
	
	Y_train = train[:,-1]

	del train_e,train_p, train
	
	model = getModel(X_train)
	
	rdlronplt = ReduceLROnPlateau(monitor='loss',patience=3,min_lr=0.0001)
	callbacks = [rdlronplt]
	
	history = model.fit(X_train,Y_train,batch_size=100,epochs=100,verbose=2,callbacks=callbacks,validation_split=0.1)
	del X_train, Y_train
	
	
	val_e = np.load('/home/drozd/analysis/newData/data_validate_elecs_under_1.npy') 
	val_p = np.load('/home/drozd/analysis/newData/data_validate_prots_under_1.npy')[0:val_e.shape[0],:]
	val = np.concatenate(( val_e, val_p ))
	del val_e, val_p

	X_val = val[:,0:-1]
	
	X_val = X_val / X_val.max(axis=0)
	Y_val = val[:,-1]
	del val
	
	if subDetector == 'STK' :
		X_val = X_val[:,40:58]
	else:
		X_val = X_val[:,32:40]
	
	

	predictions_proba = model.predict(X_val)
	predictions_binary = np.around(predictions_proba)
	np.save(subDetector+"_predictions.npy",predictions_proba)
	np.save(subDetector+"_truth.npy",Y_val)
	del X_val
	
	
	# Prediction histogram
	elecs_p, prots_p = getClassifierScore(Y_val,predictions_proba)
	binList = [x/50 for x in range(0,51)]
	fig4 = plt.figure()
	plt.hist(elecs_p,bins=binList,label='e',alpha=0.7,histtype='step',color='green')
	plt.hist(prots_p,bins=binList,label='p',alpha=0.7,histtype='step',color='red')
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.title('Prediction histogram')
	plt.legend(loc='upper center')
	plt.grid(True)
	plt.ylim((9,1e+6))
	plt.yscale('log')
	plt.savefig(subDetector+'_predHisto')
	plt.close(fig4)

	with open(subDetector+'_graphData.pickle','wb') as f:
		pickle.dump([elecs_p,prots_p],f,protocol=2)
	
	try:
		n_elecs_top = elecs_p[ elecs_p > 0.9 ].shape[0]
		n_prots_top = prots_p[ prots_p > 0.9 ].shape[0]
		contamination = n_prots_top / (n_elecs_top + n_prots_top)
		
		n_elecs_top_95 = elecs_p[ elecs_p > 0.95 ].shape[0]
		n_prots_top_95 = prots_p[ prots_p > 0.95 ].shape[0]
		contamination_95 = n_prots_top_95 / (n_elecs_top_95 + n_prots_top_95)
	except ZeroDivisionError:
		contamination = 1.
		contamination_95 = 1.
		
	l_precision, l_recall, l_thresholds = precision_recall_curve(Y_val,predictions_proba)
	
	with open(subDetector+"_PR-curve.pick","wb") as f:
		pickle.dump([l_precision,l_recall],f,protocol=2)

	l_f1 = []
	for i in range(len(l_precision)):
		l_f1.append( 2*(l_precision[i] * l_recall[i])/(l_precision[i] + l_recall[i])    )
	mf1 = max(l_f1)
			
	AUC = average_precision_score(Y_val,predictions_proba)
	
	try:
		pr = precision_score(Y_val,predictions_proba)
		rc = recall_score(Y_val,predictions_proba)			
	except:
		pr = precision_score(Y_val,predictions_binary)
		rc = recall_score(Y_val,predictions_binary)			
		
	del Y_val, predictions_proba, predictions_binary
		
	############################################################################################################
	############################################################################################################
	############################################################################################################
	
		
	
if __name__ == '__main__' :
	
	try:
		subDetector = sys.argv[1]
	except IndexError:
		subDetector = "STK"
		
	if subDetector not in ["STK","PSD"]:
		print("WARNING! Subdetector keyword not identified, assigning to STK")
		subDetector = "STK"
	
	run(subDetector)
	
	
