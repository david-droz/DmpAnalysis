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
from keras import regularizers
from keras.optimizers import Adam


def getModel(X_train):
	model = Sequential()
	model.add(Dense(250,input_shape=(X_train.shape[1],),kernel_initializer='he_uniform',activation='relu',kernel_regularizer=regularizers.l2(0.),activity_regularizer=regularizers.l2(0.)))
	model.add(Dropout(0.3))
	model.add(Dense(150,kernel_initializer='he_uniform',activation='relu',kernel_regularizer=regularizers.l2(0.)))
	model.add(Dropout(0.3))
	model.add(Dense(75,kernel_initializer='he_uniform',activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(1,kernel_initializer='he_uniform',activation='sigmoid'))
	#~ model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
	model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01), metrics=['binary_accuracy'])
	return model
	
def getLinearModel(X_train,model):
	model2 = Sequential()
	model2.add(Dense(250,input_shape=(X_train.shape[1],),kernel_initializer='he_uniform',activation='relu',kernel_regularizer=regularizers.l2(0.),activity_regularizer=regularizers.l2(0.)))
	model2.add(Dropout(0.3))
	model2.add(Dense(150,kernel_initializer='he_uniform',activation='relu',kernel_regularizer=regularizers.l2(0.)))
	model2.add(Dropout(0.3))
	model2.add(Dense(75,kernel_initializer='he_uniform',activation='relu'))
	model2.add(Dropout(0.2))
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

def testIfGPU():
	print("--- GPU CHECK ---")
	
	try:
		from theano import function, config, shared, tensor
	except ImportError:
		import tensorflow as tf
		sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
		return
	vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
	iters = 1000
	rng = np.random.RandomState(22)
	x = shared(np.asarray(rng.rand(vlen), config.floatX))
	f = function([], tensor.exp(x))
	print(f.maker.fgraph.toposort())
	for i in range(iters):
		r = f()
	if np.any([isinstance(x.op, tensor.Elemwise) and
	              ('Gpu' not in type(x.op).__name__)
	              for x in f.maker.fgraph.toposort()]):
	    print('Used the cpu')
	else:
	    print('Used the gpu')
	
	print('-------')

class TimeHistory(Callback):
	'''
	https://stackoverflow.com/questions/43178668/record-the-computation-time-for-each-epoch-in-keras-during-model-fit
	'''
	def on_train_begin(self, logs={}):
		self.times = []

	def on_epoch_begin(self, batch, logs={}):
		self.epoch_time_start = time.time()

	def on_epoch_end(self, batch, logs={}):
		self.times.append(time.time() - self.epoch_time_start)
		print("Epoch time: ", time.time() - self.epoch_time_start )


	
def trainOne(weights=True):
	
	#~ ne = int(3e+5)				# 300k events _per energy range_
	#~ ntest = ne + int(2e+5)
	
	ne = int(4e+4)
	ntest = ne + int(3e+4)
	
	
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
	
	
	model = getModel(X_train)
	rdlronplt = ReduceLROnPlateau(monitor='loss',patience=2,min_lr=0.00005)	
	time_c = TimeHistory()
	if weights :
		np.save('out/Xmax_full_w.npy',X_max)
		history = model.fit(X_train,Y_train,batch_size=40,epochs=50,verbose=2,callbacks=[rdlronplt,time_c],validation_data=(X_test,Y_test),sample_weight=weight_train)
		model2 = getLinearModel(X_train,model)
		model2.save('out/model_full_weighted.h5')
	else:
		np.save('out/Xmax_full_uw.npy',X_max)
		history = model.fit(X_train,Y_train,batch_size=40,epochs=50,verbose=2,callbacks=[rdlronplt,time_c],validation_data=(X_test,Y_test))
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
		if n == 'sigmoid':
			binList = [x/50 for x in range(0,51)]
		else :
			binList = [x for x in range(-60,60,2)]
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
	
	
	
def trainThree(er=None,rerun=False):
	
	for erange in ['20GeV_100GeV','100GeV_1TeV','1TeV_10TeV']:
		
		if os.path.isfile('out/model_'+erange+'.h5') and not rerun : continue
		
		if er is not None and er != erange: continue
		
		try:
			arr_e = np.load('DmlNtup_allElectron-v6r0p0_1GeV_10TeV_merged_'+erange+'.npy')
			arr_p = np.load('DmlNtup_allProton-v6r0p0_1GeV_100TeV_merged_'+erange+'.npy')
		except MemoryError :
			print(erange)
			raise
		np.random.shuffle(arr_e)
		np.random.shuffle(arr_p)
		
		#~ n_e = min( [int( 0.6* arr_e.shape[0]) , int(6e+5) ] )
		n_e = int(1e+5)
		train_e = arr_e[ 0:n_e ]
		train_p = arr_p[ 0:n_e ]
		
		#~ n_t = min( [n_e + int(4e+5) , arr_e.shape[0]] )
		n_t = int(2e+5)
		
		test_e = arr_e[ n_e:n_t ]
		test_p = arr_p[ n_e:n_t ]
		
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
		del X_max
		
		model = getModel(X_train)
		
		rdlronplt = ReduceLROnPlateau(monitor='loss',patience=2,min_lr=0.00005)	
		#~ time_c = TimeHistory()
		callbacks = [rdlronplt]
		history = model.fit(X_train,Y_train,batch_size=30,epochs=50,verbose=2,callbacks=callbacks,validation_data=(X_test,Y_test))
		
		model2 = getLinearModel(X_train,model)
		
		del X_train, Y_train
		
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
			
		model2.save('out/model_'+erange+'.h5')


if __name__ == '__main__':
	
	#~ try:
		#~ testIfGPU()
	#~ except Exception as Ex:
		#~ print("GPU Test failed")
		#~ print(str(Ex))
	
	for d in ['out','plots']:
		if not os.path.isdir(d): os.mkdir(d)
	
	# There HAS to be a smarter way to do that	
	if len(sys.argv) > 1 :
		if sys.argv[1] == '1':
			trainOne()
		elif sys.argv[1] == '2':
			trainOne(False)
		elif sys.argv[1] == '3':
			trainThree()
		elif sys.argv[1] == '4':
			trainThree(er='20GeV_100GeV',rerun=True)
		elif sys.argv[1] == '5':
			trainThree(er='100GeV_1TeV',rerun=True)
		elif sys.argv[1] == '6':	
			trainThree(er='1TeV_10TeV',rerun=True)
		else:
			print("Doing nothing")
	
	else:
		if not os.path.isfile('out/model_full_weighted.h5'):
			trainOne()
		
		if not os.path.isfile('out/model_1TeV_10TeV.h5'):
			trainThree()
		if not os.path.isfile('out/model_full_unweighted.h5'):
			trainOne(False)
	
