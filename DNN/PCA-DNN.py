'''

Looks at DNN performances with varying number of Principal Components, using PCA

'''

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

from scipy import stats


def getParticleSet(fname):
	arr = np.load(fname)
	X = arr[:,0:-2]				### Last two columns are timestamp and particle id
	Y = arr[:,-1]
	X = _normalise(X)
	r = np.concatenate(( X, Y.reshape(( Y.shape[0], 1 )) ) , axis=1)
	del arr, X, Y
	return r
	
def getClassifierScore(truth,pred):
	elecs = pred[truth.astype(bool)]
	prots = pred[~truth.astype(bool)]
			
	return elecs, prots
	
def getModel(X_train):
	model = Sequential()
	model.add(Dense(300,input_shape=(X_train.shape[1],),kernel_initializer='he_uniform',activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(150,kernel_initializer='he_uniform',activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(70,kernel_initializer='he_uniform',activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(1,kernel_initializer='he_uniform',activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
	return model
	

def _normalise(arr):
	for i in range(arr.shape[1]):
		arr[:,i] = (arr[:,i] - np.mean(arr[:,i])) / np.std(arr[:,i])	
	return arr

def run(preNorm,runOn,n):
	
	if preNorm: outfile = "results/pre_" + runOn + '_' + "%02d" % (n,) + '.pick'
	else: outfile = "results/post_" + runOn + '_' + "%02d" % (n,) + '.pick'
	
	if os.path.isfile(outfile): return
	
	np.random.seed(5)
	p = PCA(n_components=n)
	
	if preNorm:
		
		elecsSet = np.load('/home/drozd/analysis/newData/data_train_elecs.npy')
		protsSet = np.load('/home/drozd/analysis/newData/data_train_prots.npy')
		
		if runOn == 'e':
			p.fit(elecsSet[:,0:-2])
		elif runOn == 'p':
			p.fit(protsSet[:,0:-2])
		else:
			p.fit( np.concatenate(( elecsSet[:,0:-2] , protsSet[:,0:-2]  )) )
		
		del elecsSet, protsSet		
	
	
	train_e = np.load('/home/drozd/analysis/newData/data_train_elecs_under_1.npy')
	train_p = np.load('/home/drozd/analysis/newData/data_train_prots_under_1.npy')
	val_e = np.load('/home/drozd/analysis/newData/data_validate_elecs_under_1.npy') 
	val_p = np.load('/home/drozd/analysis/newData/data_validate_prots_under_1.npy')[0:val_e.shape[0],:]

	train = np.concatenate(( train_e, train_p ))
	np.random.shuffle(train)
	X_train = train[:,0:-2] / (train[:,0:-2]).max(axis=0)
	Y_train = train[:,-1]

	val = np.concatenate(( val_e, val_p ))
	np.random.shuffle(val)
	X_val = val[:,0:-2] / (val[:,0:-2]).max(axis=0)
	Y_val = val[:,-1]
		
	if not preNorm:
		if runOn == 'e':
			p.fit(train_e[:,0:-1])
		elif runOn == 'p':
			p.fit(train_p[:,0:-1])
		else:
			p.fit(X_train)
		
	
	del train_e,train_p, train, val_e, val_p, val
	
	
	X_train = p.transform(X_train)[:,0:n]
	X_val = p.transform(X_val)[:,0:n]
	
	model = getModel(X_train)
	history = model.fit(X_train,Y_train,batch_size=150,epochs=40,verbose=0,callbacks=[],validation_data=(X_val,Y_val))

	predictions_proba = model.predict(X_val)
	predictions_binary = np.around(predictions_proba)
	del X_train, X_val
	
	# Prediction histogram
	elecs_p, prots_p = getClassifierScore(Y_val,predictions_proba)
	binList = [x/50 for x in range(0,51)]
	fig4 = plt.figure()
	plt.hist(elecs_p,bins=binList,label='e',alpha=0.7,histtype='step',color='green')
	plt.hist(prots_p,bins=binList,label='p',alpha=0.7,histtype='step',color='red')
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.title('Balanced validation set')
	plt.legend(loc='best')
	plt.yscale('log')
	if preNorm : plt.savefig('images/pre_'+runOn+'_predHisto_'+ "%02d" % (n,))
	else: plt.savefig('images/post_'+runOn+'_predHisto_'+ "%02d" % (n,))
	plt.close(fig4)
	
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
	
	
	with open(outfile,'wb') as f:
		pickle.dump([n,AUC,mf1,pr,rc,contamination,contamination_95],f,protocol=2)
		
	del Y_val, predictions_proba, predictions_binary
		
	############################################################################################################
	############################################################################################################
	############################################################################################################
	
		
	
if __name__ == '__main__' :
	
	if "alse" in sys.argv[1]:			# Apply PCA before normalisation
		preNorm = False
	else: preNorm = True
	
	if "lec" in sys.argv[2]:			# Compute PCA basis on electrons, on protons, or on both
		runOn = "e"
	elif "rot" in sys.argv[2]:
		runOn = "p"
	else:
		runOn = "all"
		
	if not os.path.isdir('results'):os.mkdir('results')
	
	if not os.path.isdir('images'): os.mkdir('images')
		
	
	for n in range(1,59):
		
		touched = "touch_"+str(int(preNorm)) + runOn + "%02d" % (n,)
		
		if os.path.isfile(touched): continue
		
		with open(touched,'w') as f:
			f.write('a')
		try:
			run(preNorm,runOn,n)
		except:
			os.remove(touched)
			raise
		os.remove(touched)
	#end for
	
	if preNorm: 
		listofPicks = glob.glob('results/pre_'+runOn+'*.pick')
		figBaseName = "pre_runOn_"
	else: 
		listofPicks = glob.glob('results/post_'+runOn+'*.pick')
		figBaseName = "post_runOn_" + runOn + '_'
	
	listofPicks.sort()
	
	nrofvariables = []
	l_AUC = []
	l_f1 = []
	l_pr = []
	l_rc = []
	l_contamination = []
	l_con_95 = []
	for f in listofPicks:
		a,b,c,d,e,f,g = pickle.load(open(f,'rb'))
		nrofvariables.append(a)
		l_AUC.append(b)
		l_f1.append(c)
		l_pr.append(d)
		l_rc.append(e)
		l_contamination.append(f)
		l_con_95.append(g)
	
	fig1 = plt.figure()
	plt.plot(nrofvariables,l_AUC,'o-',label='AUC')
	plt.plot(nrofvariables,l_f1,'o-',label='F1')
	plt.xlabel('Nr of variables')
	plt.ylabel('Score')
	plt.legend(loc='best')
	plt.savefig(figBaseName+'AUC_F1')
	
	fig2 = plt.figure()
	plt.plot(nrofvariables,l_pr,'o-',label='Purity')
	plt.plot(nrofvariables,l_rc,'o-',label='Efficiency')
	plt.xlabel('Nr of variables')
	plt.ylabel('Score')
	plt.legend(loc='best')
	plt.savefig(figBaseName+'PR-RC')
	
	fig3 = plt.figure()
	plt.plot(nrofvariables,l_contamination,'o-',label='cut at 0.9')
	plt.plot(nrofvariables,l_con_95,'o-',label='cut at 0.95')
	plt.xlabel('Nr of variables')
	plt.ylabel('p/(e+p) ratio')
	plt.yscale('log')
	plt.legend(loc='best')
	plt.title('Background fraction')
	plt.savefig(figBaseName+'Bkg')
