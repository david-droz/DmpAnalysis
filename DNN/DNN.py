'''

DNN.py

Trains a Keras deep neural network on the DAMPE electron-proton separation problem.

- Runs five times, each time on a random set of parameters
- DNN parameters are chosen randomly with getRandomParams(), which contains lists of possible parameters. Set of parameters saved in ./models
- DNN model is created dynamically from getModel.py
- Model is saved after each epoch in ./models
- Results are saved in ./results/*ID*/   where ID is a hash number corresponding to the set of parameters
- Main results are printed to screen:  Precision (purity) and Recall (completeness)

Looks for datasets in ../dataset_train.npy ; ../dataset_validate.npy
	Can change that by suppling them as arguments to the program

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
						[300,200,100,50,1],
						[300,300,300,300,1],
						[50,40,30,20,10,1],
						[100,80,60,40,20,1],
						[200,170,140,110,90,60,30,1],
						[50,50,50,50,50,50],
						[100,100,100,100,100,100] ]
						
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


def _normalise(arr):
	for i in range(arr.shape[1]):
		arr[:,i] = (arr[:,i] - np.mean(arr[:,i])) / np.std(arr[:,i])	
	return arr

def save_history(hist, hist_filename):
	import pandas as pd

	df_history = pd.DataFrame(np.asarray(hist.history["loss"]), columns=['loss'])
	for key in hist.history.keys():
		df_history[key] = pd.Series(np.asarray(hist.history[key]), index=df_history.index)
	df_history.to_hdf(hist_filename, key='history', mode='w')	
	
def run():
	
	t0 = time.time()
	
	train_e = np.load('/home/drozd/analysis/newData/data_train_elecs_under_1.npy')
	train_p = np.load('/home/drozd/analysis/newData/data_train_prots_under_1.npy')
	train = np.concatenate(( train_e, train_p ))
	np.random.shuffle(train)
	
	X_train = train[:,0:-2] / (train[:,0:-2]).max(axis=0)
	Y_train = train[:,-1]
	del train_e, train_p, train 
	
	val_e = np.load('/home/drozd/analysis/newData/data_validate_elecs_under_1.npy') 
	val_p = np.load('/home/drozd/analysis/newData/data_validate_prots_under_1.npy')[0:val_e.shape[0],:]
	val = np.concatenate(( val_e, val_p ))
	np.random.shuffle(val)
	del val_e, val_p
	X_val = val[:,0:-2]  / (val[:,0:-2] ).max(axis=0)
	Y_val = val[:,-1]
	del val

	existence = True
	while existence:
		params = getRandomParams()
		ID = ParamsID(params)										# Get an unique ID associated to the parameters
		if not os.path.isfile('results/' + str(ID) + '/purity_completeness.txt'):		# Check if set of params has already been tested. Don't write file yet because what's below can get interrupted
			existence=False
	
	model = getModel.get_model(params,X_train.shape[1])
	
	if not os.path.isdir('models'): os.mkdir('models')
	with open('models/params_' + str(ID) + '.pick','wb') as f:	# Save the parameters into a file determined by unique ID
		pickle.dump(params,f,protocol=2)
		
	chck = ModelCheckpoint("models/weights_"+str(ID)+"__{epoch:02d}-{val_loss:.2f}.hdf5",period=10)
	earl = EarlyStopping(monitor='loss',min_delta=0.0001,patience=8)			# Alternative: train epoch per epoch, evaluate something at every epoch.
	rdlronplt = ReduceLROnPlateau(monitor='loss',patience=3,min_lr=0.0001)
	callbacks = [chck,earl,rdlronplt]
	
	history = model.fit(X_train,Y_train,batch_size=200,epochs=200,verbose=2,callbacks=callbacks,validation_data=(X_val,Y_val))
	
	predictions_binary = np.around(model.predict(X_val))		# Array of 0 and 1
	predictions_proba = model.predict(X_val)				# Array of numbers [0,1]
	
	purity = precision_score(Y_val,predictions_binary)			# Precision:  true positive / (true + false positive). Purity (how many good events in my prediction?)
	completeness = recall_score(Y_val,predictions_binary)		# Recall: true positive / (true positive + false negative). Completeness (how many good events did I find?)
	F1score = f1_score(Y_val,predictions_binary)				# Average of precision and recall
	
	l_precision, l_recall, l_thresholds = precision_recall_curve(Y_val,predictions_proba)
	
	# 1 - precision = 1 - (TP/(TP + FP)) = (TP + FP)/(TP + FP) - (TP / (TP+FP)) = FP/(TP+FP) = FPR
	
	prec_95 = 0
	recall_95 = 0
	fscore_best = 0
	fscore_best_index = 0
	
	for i in range(len(l_precision)):
		fscore_temp = 2 * l_precision[i] * l_recall[i] / (l_precision[i]+l_recall[i])
		if fscore_temp > fscore_best:
			fscore_best = fscore_temp
			fscore_best_index = i
	
	prec_95 = l_precision[fscore_best_index]
	recall_95 = l_recall[fscore_best_index]
	
	if prec_95 < 0.6 or recall_95 < 0.1 :
		prec_95 = purity
		recall_95 = completeness
					
					
	print("Precision:", prec_95)
	print("Recall:", recall_95)
	print("Iteration run time: ", time.strftime('%H:%M:%S', time.gmtime(time.time() - t0))	)
	
	if not os.path.isdir('results'): os.mkdir('results')
	if not os.path.isdir('results/' + str(ID)) : os.mkdir('results/' + str(ID))
	with open('results/' + str(ID) + '/results.pick','wb') as f:
		pickle.dump([l_precision,l_recall,l_thresholds],f,protocol=2)
	np.save('results/' + str(ID) + '/predictions.npy',predictions_proba)
	np.save('results/Y_Val.npy',Y_val)
	
	save_history(history,'results/'+str(ID)+'/history.hdf')
	
	elecs_p, prots_p = getClassifierScore(Y_val,predictions_proba)
	binList = [x/50 for x in range(0,51)]
	fig4 = plt.figure()
	plt.hist(elecs_p,bins=binList,label='e',alpha=0.7,histtype='step',color='green')
	plt.hist(prots_p,bins=binList,label='p',alpha=0.7,histtype='step',color='red')
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.title('Balanced validation set')
	plt.legend(loc='upper center')
	plt.yscale('log')
	plt.savefig('images/' + str(ID))
	plt.close(fig4)
	
	n_elecs_top = elecs_p[ elecs_p > 0.8 ].shape[0]
	n_prots_top = prots_p[ prots_p > 0.8 ].shape[0]
	contamination = n_prots_top / (n_elecs_top + n_prots_top)
	
	with open('results/' + str(ID) + '/purity_completeness.txt','w') as g:
		g.write("Precision: "+str(prec_95)+'\n')
		g.write("Recall: "+str(recall_95)+'\n')
		g.write("Contamination: "+str(contamination)+'\n')
	
	del X_train, X_val, Y_train, Y_val, history, predictions_binary, predictions_proba
	
		
		
	
if __name__ == '__main__' :
	
	for x in range(5):
		print('------------ ', x, ' ----------------')
		run()

