'''

miniDNN.py

Trains a simple DNN to test if it works

'''


from __future__ import division, print_function, absolute_import

import numpy as np
import time
import pickle
import sys
import os
import random
import hashlib
import sys

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler, ReduceLROnPlateau
from keras.constraints import maxnorm
from keras.layers.advanced_activations import PReLU, ELU, LeakyReLU

from sklearn.metrics import roc_curve, roc_auc_score, precision_score, average_precision_score, precision_recall_curve, recall_score
from sklearn.metrics import f1_score

import getModel

##############################################

# Possible parameters

def ParamsID(a):
	'''
	Using hash function to build an unique identifier for each dictionary
	'''
	ID = 1
	for x in ['acti_out','init','optimizer']:
		ID = ID + int(hashlib.sha256(a[x].encode('utf-8')).hexdigest(), 16)
	ID = ID + int(hashlib.sha256(str(a['activation'] + 'b').encode('utf-8')).hexdigest(), 16) 
	ID = ID + int(hashlib.sha256(str(a['dropout']).encode('utf-8')).hexdigest(), 16) 
	ID = ID + int(hashlib.sha256(str(len(a['architecture'])).encode('utf-8')).hexdigest(), 16)
	for y in a['architecture']:
		ID = ID + int(hashlib.sha256(str(y).encode('utf-8')).hexdigest(), 16) 
	ID = ID + int(a['batchnorm'])
	return (ID % 10**10)

def getRandomParams():
	'''
	Returns a dictionary containing all the parameters to train the Keras DNN. Is then fed to getModel::getModel()
	'''
	p_dropout = [0.,0.1,0.2,0.5]
	p_batchnorm = [True, True, False]			# Repetition of values to increase probability
	p_activation = ['relu','relu','relu','sigmoid','softplus','elu','elu']
	#~ p_activation = ['relu','relu','relu','sigmoid','softplus','elu','elu',PRelu(),PRelu(),ELU(),LeakyReLU()]	 
	p_acti_out = ['sigmoid']
	p_init = ['uniform','glorot_uniform','glorot_normal','lecun_uniform','he_uniform','he_normal']
	p_loss = ['binary_crossentropy']
	p_optimizer = ['adagrad','adadelta','adam','adamax','nadam','sgd']
	p_metric = [['binary_accuracy']]
	p_architecture = [ 	[50,20,1],
						[50,25,12,1],
						[60,20,1],
						[60,30,15,1],
						[100,50,25,1],
						[100,100,100,1],
						[200,100,50,1],
						[200,100,50,25,1],
						[300,300,300,1],
						[300,150,75,1],
						[300,200,100,50,1] ]
						
	mydic = {}
	mydic['architecture'] = random.choice(p_architecture)
	mydic['dropout'] = random.choice(p_dropout)
	mydic['batchnorm'] = random.choice(p_batchnorm)
	mydic['activation'] = random.choice(p_activation)
	mydic['acti_out'] = random.choice(p_acti_out)
	mydic['init'] = random.choice(p_init)
	mydic['loss'] = random.choice(p_loss)
	mydic['optimizer'] = random.choice(p_optimizer)
	mydic['metrics'] = random.choice(p_metric)
	
	return mydic

def getParticleSet(fname):
	arr = np.load(fname)
	X = arr[:,0:-2]				### Last two columns are timestamp and particle id
	Y = arr[:,-1]
	X = _normalise(X)
	r = np.concatenate(( X, Y.reshape(( Y.shape[0], 1 )) ) , axis=1)
	del arr, X, Y
	return r

def _normalise(arr):
	for i in range(arr.shape[1]):
		arr[:,i] = (arr[:,i] - np.mean(arr[:,i])) / np.std(arr[:,i])	
	return arr

def save_history(hist, hist_filename):
	import pandas as pd

	df_history = pd.DataFrame(np.asarray(hist.history["loss"]), columns=['loss'])
	df_history['acc'] = pd.Series(np.asarray(hist.history["acc"]), index=df_history.index)
	df_history['val_loss'] = pd.Series(np.asarray(hist.history["val_loss"]), index=df_history.index)
	df_history['val_acc'] = pd.Series(np.asarray(hist.history["val_acc"]), index=df_history.index)
	df_history.to_hdf(hist_filename, key='history', mode='w')	
	
def run():
	
	t0 = time.time()
	
	balanced = True
	
	if len(sys.argv) > 1:
		if sys.argv[1] in ['0','unbalanced']: balanced=False 
		
	if balanced:
		train_e = getParticleSet('/home/drozd/analysis/fraction1/data_train_elecs.npy')
		train_p = getParticleSet('/home/drozd/analysis/fraction1/data_train_prots.npy')
		val_e = getParticleSet('/home/drozd/analysis/fraction1/data_validate_elecs.npy') 
		val_p = getParticleSet('/home/drozd/analysis/fraction1/data_validate_prots.npy') 
	else:
		train_e = getParticleSet('/home/drozd/analysis/data_train_elecs.npy')
		train_p = getParticleSet('/home/drozd/analysis/data_train_prots.npy')
		val_e = getParticleSet('/home/drozd/analysis/data_validate_elecs.npy') 
		val_p = getParticleSet('/home/drozd/analysis/data_validate_prots.npy') 
	
	train = np.concatenate(( train_e, train_p ))
	np.random.shuffle(train)
	X_train = train[:,0:-1]
	Y_train = train[:,-1]
	del train_e,train_p, train
	
	val = np.concatenate(( val_e, val_p ))
	np.random.shuffle(val)
	X_val = val[:,0:-1]
	Y_val = val[:,-1]
	del val_e, val_p, val
	
	model = Sequential()
	model.add(Dense(50,
					input_shape=(X_train.shape[1],),
					kernel_initializer='uniform',
					activation='relu'))
	model.add(Dense(25,kernel_initializer='uniform',activation='relu'))
	model.add(Dense(12,kernel_initializer='uniform',activation='relu'))
	model.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	callbacks = []
	
	history = model.fit(X_train,Y_train,batch_size=100,epochs=25,verbose=2,callbacks=callbacks,validation_data=(X_val,Y_val))
	
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
					
	print("Precision:", prec_95)
	print("Recall:", recall_95)
	print("Run time: ", time.strftime('%H:%M:%S', time.gmtime(time.time() - t0))	)
	
	if not os.path.isdir('results'): os.mkdir('results')
	
	if balanced: minidir='./results/balanced'
	else: minidir='./results/unbalanced'
	if not os.path.isdir(minidir): os.mkdir(minidir)
	
	
	with open(minidir + '/results.pick','wb') as f:
		pickle.dump([l_precision,l_recall,l_thresholds],f,protocol=2)
	np.save(minidir + '/predictions.npy',predictions_proba)
	np.save(minidir + '/Y_Val.npy',Y_val)
	with open(minidir + '/purity_completeness.txt','w') as g:
		g.write("Precision: "+str(prec_95)+'\n')
		g.write("Recall: "+str(recall_95)+'\n')
	save_history(history,minidir+'/history.hdf')
	
	del X_train, X_val, Y_train, Y_val, history, predictions_binary, predictions_proba
	
		
		
	
if __name__ == '__main__' :
	
	run()

