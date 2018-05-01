'''

Train DNN on Xin's ntuples. Go for three methods:

- Train a single model on full spectrum, with weights
- Train one model per energy bin, no weights
- Train a single model on full spectrum, no weights


DmlNtup_allElectron-v6r0p0_1GeV_10TeV_merged_100GeV_1TeV.npy (1 291 379, 51)
DmlNtup_allElectron-v6r0p0_1GeV_10TeV_merged_1TeV_10TeV.npy (1 403 014, 51)
DmlNtup_allElectron-v6r0p0_1GeV_10TeV_merged_20GeV_100GeV.npy (2 721 011, 51)

DmlNtup_allProton-v6r0p0_1GeV_100TeV_merged_100GeV_1TeV.npy (7 973 173, 51)
DmlNtup_allProton-v6r0p0_1GeV_100TeV_merged_1TeV_10TeV.npy (4 687 308, 51)
DmlNtup_allProton-v6r0p0_1GeV_100TeV_merged_20GeV_100GeV.npy (5 059 310, 51)

'''

from __future__ import print_function, division, absolute_import

import numpy as np
import time
import pickle
import sys
import os
import glob
from uncertainties import ufloat
from argparse import ArgumentParser

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score, precision_score, average_precision_score, precision_recall_curve, recall_score
from sklearn.metrics import f1_score

# Keras deep neural networks
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler, ReduceLROnPlateau


def getModel(X_train):
	model = Sequential()
	model.add(Dense(300,input_shape=(X_train.shape[1],),kernel_initializer='he_uniform',activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(150,kernel_initializer='he_uniform',activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(75,kernel_initializer='he_uniform',activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(1,kernel_initializer='he_uniform',activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
	return model
	
def getLinearModel(X_train,model):
	model2 = Sequential()
	model2.add(Dense(300,input_shape=(X_train.shape[1],),kernel_initializer='he_uniform',activation='relu'))
	model2.add(Dropout(0.1))
	model2.add(Dense(150,kernel_initializer='he_uniform',activation='relu'))
	model2.add(Dropout(0.1))
	model2.add(Dense(75,kernel_initializer='he_uniform',activation='relu'))
	model2.add(Dropout(0.1))
	model2.add(Dense(1,kernel_initializer='he_uniform'))
	model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
	
	for i,x in enumerate(model.layers):
		weights = x.get_weights()
		model2.layers[i].set_weights(weights)
		
	return model2
	
def getClassifierScore(truth,pred):
	elecs = pred[truth.astype(bool)]
	prots = pred[~truth.astype(bool)]
	return elecs, prots
		
	
def trainOne(weights=True):
	
	ne = int(4e+5)				# 400k events _per energy range_
	ntest = ne + int(3e+5)
	
	
	###
	# LOAD TRAINING
	###
	arr_e = np.load('DmlNtup_allElectron-v6r0p0_1GeV_10TeV_merged_20GeV_100GeV.npy')[0:ne]
	arr_p = np.load('DmlNtup_allProton-v6r0p0_1GeV_100TeV_merged_20GeV_100GeV.npy')[0:ne]
	
	for erange in ['100GeV_1TeV','1TeV_10TeV']:
		try:
			arr_e = np.concatenate(( arr_e, np.load('DmlNtup_allElectron-v6r0p0_1GeV_10TeV_merged_'+erange+'.npy')[0:ne] ))
			arr_p = np.concatenate(( arr_p, np.load('DmlNtup_allProton-v6r0p0_1GeV_100TeV_merged_'+erange+'.npy')[0:ne] ))
		except MemoryError :
			print(erange)
			raise
	
	train = np.concatenate(( arr_e , arr_p ))
	np.random.shuffle(train)
	del arr_e, arr_p
	
	X_train = train[:,0:47]
	Y_train = train[:,-1]
	weight_train = train[:,-3]		
	del train
	
	###
	# LOAD TEST
	###
	arr_e = np.load('DmlNtup_allElectron-v6r0p0_1GeV_10TeV_merged_20GeV_100GeV.npy')[ne:ntest]
	arr_p = np.load('DmlNtup_allProton-v6r0p0_1GeV_100TeV_merged_20GeV_100GeV.npy')[ne:ntest]
	for erange in ['100GeV_1TeV','1TeV_10TeV']:
		arr_e = np.concatenate(( arr_e, np.load('DmlNtup_allElectron-v6r0p0_1GeV_10TeV_merged_'+erange+'.npy')[ne:ntest] ))
		arr_p = np.concatenate(( arr_p, np.load('DmlNtup_allProton-v6r0p0_1GeV_100TeV_merged_'+erange+'.npy')[ne:ntest] ))
	test = np.concatenate(( arr_e , arr_p ))
	np.random.shuffle(test)
	del arr_e, arr_p
	
	X_test = test[:,0:47]
	Y_test = test[:,-1]
	weight_test = test[:,-3]		
	del test
	
	###
	# PROCEED TO TRAINING
	###
	
	X_max = X_train.max(axis=0)
	X_train = X_train / X_max
	X_test = X_test / X_max
	np.save('out/Xmax_full.npy',X_max)
	
	model = getModel(X_train)
	rdlronplt = ReduceLROnPlateau(monitor='loss',patience=3,min_lr=0.0001)	
	if weights :
		history = model.fit(X_train,Y_train,batch_size=20,epochs=100,verbose=0,callbacks=[rdlronplt],validation_data=(X_test,Y_test),sample_weight=weight_train)
		model2 = getLinearModel(X_train,model)
		model2.save('out/model_full_weighted.h5')
	else:
		history = model.fit(X_train,Y_train,batch_size=20,epochs=100,verbose=0,callbacks=[rdlronplt],validation_data=(X_test,Y_test))
		model2 = getLinearModel(X_train,model)
		model2.save('out/model_full_unweighted.h5')
	
	fig1 = plt.figure()
	plt.plot(history.history['loss'],label='loss')
	plt.plot(history.history['val_loss'],label='val_loss')
	plt.legend(loc='best')
	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.title('train history')
	if weights: plt.savefig('plots/history_full_weighted')
	else: plt.savefig('plots/history_full_unweighted')
	plt.close(fig1)
	
	for m, n in [ [model,'sigmoid'],[model2,'linear'] ] :
		predictions = m.predict(X_test)
		elecs_p, prots_p = getClassifierScore(Y_test,predictions)
		weights_elec = weight_test[ Y_test.astype(bool) ]
		weights_prot = weight_test[ ~Y_test.astype(bool) ]
		
		fig2 = plt.figure()
		binList = [x/50 for x in range(0,51)]
		plt.hist(elecs_p,bins=binList,label='e',histtype='step',color='green')
		plt.hist(prots_p,bins=binList,label='p',histtype='step',color='red')
		plt.xlabel('Classifier score')
		plt.ylabel('Number of events')
		plt.legend(loc='upper center')
		plt.grid(True)
		#~ plt.ylim((0.09,1e+5))
		plt.yscale('log')
		if weights:  plt.savefig('plots/classScore_full_weighted_'+n)
		else: plt.savefig('plots/classScore_full_unweighted_'+n)
		plt.close(fig2)
	
	
	
def trainThree():
	
	for erange in ['20GeV_100GeV','100GeV_1TeV','1TeV_10TeV']:
		
		try:
			arr_e = np.load('DmlNtup_allElectron-v6r0p0_1GeV_10TeV_merged_'+erange+'.npy')
			arr_p = np.load('DmlNtup_allProton-v6r0p0_1GeV_100TeV_merged_'+erange+'.npy')
		except MemoryError :
			print(erange)
			raise
		
		n_e = int( 0.6* arr_e.shape[0] )
		train_e = arr_e[ 0:n_e ]
		train_p = arr_p[ 0:n_e ]
		test_e = arr_e[ n_e:-1 ]
		test_p = arr_p[ n_e:-1 ]
		
		del arr_e , arr_p 
		
		train = np.concatenate(( train_e , train_p ))
		np.random.shuffle(train)
		del train_e, train_p 
		
		X_train = train[:,0:47]
		Y_train = train[:,-1]
		weight_train = train[:,-3]		
		del train
		
		test = np.concatenate(( test_e , test_p ))
		X_test = test[:,0:47]
		Y_test = test[:,-1]
		weight_test = test[:,-3]		
		del test, test_e, test_p
		
		
		X_max = X_train.max(axis=0)
		X_train = X_train / X_max
		X_test = X_test / X_max
		np.save('out/Xmax_'+erange+'.npy',X_max)
		
		model = getModel(X_train)
		
		rdlronplt = ReduceLROnPlateau(monitor='loss',patience=3,min_lr=0.0001)	
		callbacks = [rdlronplt]
		history = model.fit(X_train,Y_train,batch_size=20,epochs=100,verbose=0,callbacks=callbacks,validation_data=(X_test,Y_test))
		
		model2 = getLinearModel(X_train,model)
		
		model2.save('out/model_'+erange+'.h5')
		
		fig1 = plt.figure()
		plt.plot(history.history['loss'],label='loss')
		plt.plot(history.history['val_loss'],label='val_loss')
		plt.legend(loc='best')
		plt.xlabel('epochs')
		plt.ylabel('loss')
		plt.title('train history')
		plt.savefig('plots/history_'+erange)
		plt.close(fig1)
		
		for m, n in [ [model,'sigmoid'],[model2,'linear'] ] : 
			predictions = m.predict(X_test)
			elecs_p, prots_p = getClassifierScore(Y_test,predictions)
			
			fig2 = plt.figure()
			binList = [x/50 for x in range(0,51)]
			plt.hist(elecs_p,bins=binList,label='e',histtype='step',color='green')
			plt.hist(prots_p,bins=binList,label='p',histtype='step',color='red')
			plt.xlabel('Classifier score')
			plt.ylabel('Number of events')
			plt.legend(loc='upper center')
			plt.grid(True)
			#~ plt.ylim((0.09,1e+5))
			plt.yscale('log')
			plt.savefig('plots/classScore_'+erange+'_'+n)
			plt.close(fig2)


if __name__ == '__main__':
	
	for d in ['out','plots']:
		if not os.path.isdir(d): os.mkdir(d)
	
	if not os.path.isfile('out/model_full_weighted.h5'):
		trainOne()
	
	if not os.path.isfile('out/model_1TeV_10TeV.h5'):
		trainThree()
	if not os.path.isfile('out/model_full_unweighted.h5'):
		trainOne(False)
