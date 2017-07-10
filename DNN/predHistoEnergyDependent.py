'''

Train a model, and then plot the histogram of classifier score on several energy bins

'''

from __future__ import division, print_function, absolute_import

import numpy as np
import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler, ReduceLROnPlateau

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, average_precision_score, precision_recall_curve, recall_score
from sklearn.metrics import f1_score


##############################################


def _normalise(arr):
	for i in range(arr.shape[1]):
		arr[:,i] = (arr[:,i] - np.mean(arr[:,i])) / np.std(arr[:,i])	
	return arr

def getClassifierScore(truth,pred):
	elecs = pred[truth.astype(bool)]
	prots = pred[~truth.astype(bool)]
			
	return elecs, prots

def getParticleSet(fname, Emin, Emax):
	arr = np.load(fname)
	arr = arr[ arr[:,30] > Emin ]
	arr = arr[ arr[:,30] < Emax ]
	
	X = arr[:,0:-2]	
	Y = arr[:,-1]
	X = _normalise(X)
	r = np.concatenate(( X, Y.reshape(( Y.shape[0], 1 )) ) , axis=1)
	del arr, X, Y
	return r

	
def run(Emin, Emax, BDT=False):
	
	if BDT:
		figureName = 'pred_BDT_'+str(int(Emin/1000))+'-'+str(int(Emax/1000))
	else:
		figureName = 'pred_DNN_'+str(int(Emin/1000))+'-'+str(int(Emax/1000))
		
	if os.path.isfile(figureName+'.png'): return
	
	train_e = getParticleSet('/home/drozd/analysis/fraction1/data_train_elecs.npy', Emin, Emax)
	train_p = getParticleSet('/home/drozd/analysis/fraction1/data_train_prots.npy', Emin, Emax)
	train = np.concatenate(( train_e, train_p ))
	np.random.shuffle(train)
	X_train = train[:,0:-1]
	Y_train = train[:,-1]
	del train_e,train_p, train

	val_e = getParticleSet('/home/drozd/analysis/fraction1/data_validate_elecs_1.npy', Emin, Emax)
	val_p = getParticleSet('/home/drozd/analysis/fraction1/data_validate_prots_1.npy', Emin, Emax)
	val = np.concatenate(( val_e, val_p ))
	np.random.shuffle(val)
	
	X_val = val[:,0:-1]
	Y_val = val[:,-1]
	
	del val_e, val_p, val
	
	print(str(int(Emin/1000))+'-'+str(int(Emax/1000))+ ': Training on ', X_train.shape[0], ' events')
	
	if BDT:
		model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,max_depth=3, min_samples_leaf=0.0001)
		model.fit(X_train, Y_train)
		
		predictions = model.predict_proba(X_val)[:,1]
	else:
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
		#~ earl = EarlyStopping(monitor='loss',min_delta=0.0001,patience=5)
		callbacks = [rdlronplt]
		history = model.fit(X_train,Y_train,batch_size=150,epochs=75,verbose=0,callbacks=callbacks,validation_data=(X_val,Y_val))
		
		predictions = model.predict(X_val)

	
	elecs_p, prots_p = getClassifierScore(Y_val,predictions)	
	
	Nbins_plt = 50
	binList = [x/Nbins_plt for x in range(0,Nbins_plt+1)]		
	fig = plt.figure()
	plt.hist(elecs_p,bins=binList,label='e',alpha=0.7,histtype='step',color='green')
	plt.hist(prots_p,bins=binList,label='p',alpha=0.7,histtype='step',color='red')
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.title( str(int(Emin/1000))+' GeV - '+str(int(Emax/1000))+' GeV' )
	plt.legend(loc='upper center')
	plt.yscale('log')
	plt.savefig(figureName)
	plt.close(fig)
	
	
	
if __name__ == '__main__' :
	
	Nbins = 4
	logbins = np.logspace(5,6.5,Nbins+1)
	
	if len(sys.argv) > 1: BDT = True
	else: BDT = False
	
	for i in range(Nbins):
		run(logbins[i],logbins[i+1],BDT)

