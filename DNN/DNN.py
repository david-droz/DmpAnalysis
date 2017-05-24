import numpy as np
import time
import cPickle as pickle
import sys
import os
import random
import hashlib

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler, ReduceLROnPlateau
from keras.constraints import maxnorm
from keras.layers.advanced_activations import PReLU, ELU, LeakyReLU

from scipy.stats import randint as sp_randint

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, average_precision_score, precision_recall_curve, recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from keras.wrappers.scikit_learn import KerasClassifier

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
	p_acti_out = ['sigmoid','sigmoid','sigmoid','relu']
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

def XY_split(fname):
	arr = np.load(fname)
	X = arr[:,0:-1]
	Y = arr[:,-1]
	return X,Y
def load_training(fname='dataset_train.npy'): return XY_split(fname)
def load_validation(fname='dataset_validate.npy'): return XY_split(fname)
def load_test(fname='dataset_test.npy'): return XY_split(fname)
	
	
def run():
	
	X_train, Y_train = load_training()
	X_val, Y_val = load_validation()
	
	params = getRandomParams()
	model = getModel.get_model(params,X_train.shape[1])
	
	if not os.path.isdir('models'): os.mkdir('models')
	
	ID = ParamsID(params)										# Get an unique ID associated to the parameters
	with open('models/params_' + str(ID) + '.pick','w') as f:	# Save the parameters into a file determined by unique ID
		pickle.dump(params,f)
		
	chck = ModelCheckpoint("models/weights_"+str(ID)+"__{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5")
	earl = EarlyStopping(monitor='loss',min_delta=0.0001,patience=5)			# Alternative: train epoch per epoch, evaluate something at every epoch.
	rdlronplt = ReduceLROnPlateau(monitor='loss',patience=4,min_lr=0.001)
	callbacks = [chck,earl,rdlronplt]
	
	history = model.fit(X_train,Y_train,batch_size=200,epochs=200,verbose=2,callbacks=callbacks,validation_data=(X_val,Y_val))
	
	predictions_binary = model.predict(X_val)			# Array of 0 and 1
	predictions_proba = model.predict_proba(X_val)		# Array of numbers [0,1]
	
	purity = precision_score(Y_val,predictions_binary)			# Precision:  true positive / (true + false positive). Purity (how many good events in my prediction?)
	completeness = recall_score(Y_val,predictions_binary)		# Recall: true positive / (true positive + false negative). Completeness (how many good events did I find?)
	F1score = f1_score(Y_val,predictions_binary)				# Average of precision and recall
	
	l_precision, l_recall, l_thresholds = precision_recall_curve(Y_val,predictions_proba)
	
	prec_95 = [x for x in l_precision if x>0.95]
	# Keep working here.  Goal is to find the best false positive rate (or other metric?) for a precision above 95%.
	# Would (1 - recall) make sense?   I guess: write the equations to make sure of that.
	
if __name__ == '__main__' :
	

