'''

For the same trained model, compare results in balanced vs imbalanced

--> Result: Training should be balanced. Validation, not necessarily. 
	Save normalisation applied on training for future use

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

def getCounts(truth,pred,threshold):
	'''
	Returns true positive, false positive, true negative, false negative
	'''
	tp, fp, tn, fn = [0,0,0,0]
	
	for i in range(truth.shape[0]):
		if truth[i] == 1. :
			if pred[i] >= threshold:
				tp += 1
			else:
				fn += 1
		elif truth[i] == 0. :
			if pred[i] >= threshold:
				fp += 1
			else:
				tn += 1
		else:
			raise Exception("Bad true labels")
	return tp,fp,tn,fn
		
def getROC(truth,pred,npoints=20):
	
	l_fpr = []
	l_tpr = []
	l_thresholds = []
	
	for i in range(npoints):	
		thr = i * (1./npoints)
		tp,fp,tn,fn = getCounts(truth,pred,thr)
		l_fpr.append( fp/(fp + tn) )
		l_tpr.append( tp/(tp + fn) )
		l_thresholds.append( thr )
		
	return l_fpr,l_tpr,l_thresholds

def getPR(truth,pred,npoints=20):

	l_pre = []
	l_rec = []
	l_thresholds = []
	
	for i in range(npoints):	
		thr = i * (1./npoints)
		tp,fp,tn,fn = getCounts(truth,pred,thr)
		l_pre.append( tp / (tp + fp) )
		l_rec.append( tp / (tp + fn) )
		l_thresholds.append( thr )
	
	#~ thr = 1.
	#~ tp,fp,tn,fn = getCounts(truth,pred,thr)
	#~ l_pre.append( tp / (tp + fp) )
	#~ l_rec.append( tp / (tp + fn) )
	#~ l_thresholds.append( thr )
	
	return l_pre,l_rec,l_thresholds
		
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
	
	val_imba = np.concatenate(( val_e[0:int(val_p.shape[0]/100)], val_p ))
	np.random.shuffle(val_imba)
	X_val_imba = val_imba[:,0:-1]
	Y_val_imba = val_imba[:,-1]
	
	del val_e, val_p, val, val_imba
	
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
	predictions_imba = model.predict(X_val_imba)
	predictions_train = model.predict(X_train)
	
	del X_val, X_val_imba, X_train
	
	sk_l_precision_b, sk_l_recall_b, sk_l_thresholds_b = precision_recall_curve(Y_val,predictions_balanced)
	sk_l_precision_i, sk_l_recall_i, sk_l_thresholds_i = precision_recall_curve(Y_val_imba,predictions_imba)
	sk_l_precision_t, sk_l_recall_t, sk_l_thresholds_t = precision_recall_curve(Y_train,predictions_train)
	
	sk_l_fpr_b, sk_l_tpr_b, sk_l_roc_thresholds_b = roc_curve(Y_val,predictions_balanced)
	sk_l_fpr_i, sk_l_tpr_i, sk_l_roc_thresholds_i = roc_curve(Y_val_imba,predictions_imba)
	sk_l_fpr_t, sk_l_tpr_t, sk_l_roc_thresholds_t = roc_curve(Y_train,predictions_train)
	
	man_l_precision_b, man_l_recall_b, man_l_thresholds_b = getPR(Y_val,predictions_balanced,100)
	man_l_precision_i, man_l_recall_i, man_l_thresholds_i = getPR(Y_val_imba,predictions_imba,100)
	
	man_l_fpr_b, man_l_tpr_b, man_l_roc_thresholds_b = getROC(Y_val,predictions_balanced,100)
	man_l_fpr_i, man_l_tpr_i, man_l_roc_thresholds_i = getROC(Y_val_imba,predictions_imba,100)
	
	print("----- AUC -----")
	print("Train:", average_precision_score(Y_train,predictions_train))
	print("Validate:", average_precision_score(Y_val,predictions_balanced))
	print("----- F1 -----")
	print("Train:", f1_score(Y_train,np.around(predictions_train)))
	print("Validate:", f1_score(Y_val,np.around(predictions_balanced)))
	print("----- Precision/Recall -----")
	print("Train:", precision_score(Y_train,np.around(predictions_train)), " / ", recall_score(Y_train,np.around(predictions_train)))
	print("Validate:", precision_score(Y_val,np.around(predictions_balanced)), " / ", recall_score(Y_val,np.around(predictions_balanced)))
	
	fig1 = plt.figure()
	plt.plot(sk_l_precision_b, sk_l_recall_b,label='balanced')
	plt.plot(sk_l_precision_i, sk_l_recall_i,label='imbalanced')
	#~ plt.plot(sk_l_precision_t, sk_l_recall_t,label='training set')
	#~ plt.plot(man_l_precision_b, man_l_recall_b,'o',label='balanced, hand')
	#~ plt.plot(man_l_precision_i, man_l_recall_i,'o',label='imbalanced, hand')
	plt.xlabel('Precision')
	plt.ylabel('Recall')
	plt.legend(loc='best')
	plt.savefig('PR')
	
	fig1b = plt.figure()
	plt.plot(sk_l_precision_b, sk_l_recall_b,label='validation set')
	plt.plot(sk_l_precision_t, sk_l_recall_t,label='training set')
	plt.xlabel('Precision')
	plt.ylabel('Recall')
	plt.legend(loc='best')
	plt.savefig('PRb')
	
	fig2 = plt.figure()
	plt.plot(sk_l_fpr_b, sk_l_tpr_b,label='balanced')
	plt.plot(sk_l_fpr_i, sk_l_tpr_i,label='imbalanced')
	#~ plt.plot(man_l_fpr_b, man_l_tpr_b,'o',label='balanced, hand')
	#~ plt.plot(man_l_fpr_i, man_l_tpr_i,'o',label='imbalanced, hand')
	plt.xlabel('False Positive')
	plt.ylabel('True Positive')
	plt.legend(loc='best')
	plt.savefig('ROC')
	
	fig2b = plt.figure()
	plt.plot(sk_l_fpr_b, sk_l_tpr_b,label='validation set')
	plt.plot(sk_l_fpr_t, sk_l_tpr_t,label='training set')
	plt.xlabel('False Positive')
	plt.ylabel('True Positive')
	plt.legend(loc='best')
	plt.savefig('ROCb')
	
	Nbins = 50
	binList = [x/Nbins for x in range(0,Nbins+1)]
	
	elecs_t, prots_t = getClassifierScore(Y_train,predictions_train)
	fig3 = plt.figure()
	plt.hist(elecs_t,bins=binList,label='e',alpha=0.7,histtype='step',color='green')
	plt.hist(prots_t,bins=binList,label='p',alpha=0.7,histtype='step',color='red')
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.title('Training set')
	plt.legend(loc='best')
	plt.yscale('log')
	plt.savefig('predHisto_train')
	
	fig3b = plt.figure()
	plt.hist(elecs_t,bins=binList,label='e',alpha=0.7,histtype='step',color='green',normed=True)
	plt.hist(prots_t,bins=binList,label='p',alpha=0.7,histtype='step',color='red',normed=True)
	plt.xlabel('Classifier score')
	plt.ylabel('Fraction of events')
	plt.title('Training set - normalised')
	plt.legend(loc='best')
	plt.yscale('log')
	plt.savefig('predHisto_train_n')	
	del elecs_t, prots_t, Y_train, predictions_train
	
	elecs_b, prots_b = getClassifierScore(Y_val,predictions_balanced)
	fig4 = plt.figure()
	plt.hist(elecs_b,bins=binList,label='e',alpha=0.7,histtype='step',color='green')
	plt.hist(prots_b,bins=binList,label='p',alpha=0.7,histtype='step',color='red')
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.title('Balanced validation set')
	plt.legend(loc='best')
	plt.yscale('log')
	plt.savefig('predHisto_bal')
	
	fig4b = plt.figure()
	plt.hist(elecs_b,bins=binList,label='e',alpha=0.7,histtype='step',color='green',normed=True)
	plt.hist(prots_b,bins=binList,label='p',alpha=0.7,histtype='step',color='red',normed=True)
	plt.xlabel('Classifier score')
	plt.ylabel('Fraction of events')
	plt.title('Balanced validation set - normalised')
	plt.legend(loc='best')
	plt.yscale('log')
	plt.savefig('predHisto_bal_n')	
	del elecs_b, prots_b, Y_val, predictions_balanced
	
	elecs_i, prots_i = getClassifierScore(Y_val_imba,predictions_imba)
	fig5 = plt.figure()
	plt.hist(elecs_i,bins=binList,label='e',alpha=0.7,histtype='step',color='green')
	plt.hist(prots_i,bins=binList,label='p',alpha=0.7,histtype='step',color='red')
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.legend(loc='best')
	plt.title('Imbalanced validation set')
	plt.yscale('log')
	plt.savefig('predHisto_imba')
	
	fig5b = plt.figure()
	plt.hist(elecs_i,bins=binList,label='e',alpha=0.7,histtype='step',color='green',normed=True)
	plt.hist(prots_i,bins=binList,label='p',alpha=0.7,histtype='step',color='red',normed=True)
	plt.xlabel('Classifier score')
	plt.ylabel('Fraction of events')
	plt.title('Imbalanced validation set - normalised')
	plt.legend(loc='best')
	plt.yscale('log')
	plt.savefig('predHisto_imba_n')	
	
	
if __name__ == '__main__' :
	
	run()

