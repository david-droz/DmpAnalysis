import numpy as np
import time
import cPickle as pickle
import sys
import os

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler
from keras.optimizers import SGD
from keras.constraints import maxnorm

from scipy.stats import randint as sp_randint

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, average_precision_score, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from keras.wrappers.scikit_learn import KerasClassifier

##############################################


def get_model():
	'''
	
	'''
	
	
def load_data(testfraction = 0.001):
	'''
	
	'''
	e_data = np.load('/home/drozd/DNN/dataset_elec.npy')
	e_data = e_data[np.any(e_data,axis=1)]  
	p_data = np.load('/home/drozd/DNN/dataset_prot.npy')
	p_data = p_data[np.any(p_data,axis=1)]  
	if e_data.shape[0] > p_data.shape[0]:
		datasize = int(0.85*p_data.shape[0])
	else:
		datasize = int(0.85*e_data.shape[0])
	dataset = np.concatenate(( e_data[0:datasize,:] , p_data[0:datasize,:] ))
	e_val = e_data[datasize:-1,:]
	p_val = p_data[datasize:-1,:]
	del e_data
	del p_data
	nrofe = e_val.shape[0]
	nrofp = p_val.shape[0]
	
	if os.path.isfile('/home/drozd/DNN/single/test/validation_set.npy'):
		test = np.load('/home/drozd/DNN/single/test/validation_set.npy')
	else:
		if nrofp > (1./testfraction)*nrofe :
			test = np.concatenate((  e_val[0:-1,:]     ,   p_val[0:int((1./testfraction) * nrofe),:]    ))
		else:
			lim_e = int((testfraction)*nrofp) 
			test = np.concatenate(( e_val[0:lim_e,:]   ,  p_val[0:int((1./testfraction) * lim_e),:]  ))
		np.random.shuffle(test)
		np.save('/home/drozd/BDT/test/validation_set.npy',test)
	
	del e_val
	del p_val
	X_train = dataset[:,0:-1]
	X_val = test[:,0:-1]
	Y_train = dataset[:,-1]
	Y_val = test[:,-1]
	del dataset
	del test
	
	
	return X_train, Y_train, X_val, Y_val
	
def main():
	
	
	
	
	
	
	
if __name__ == '__main__' :
