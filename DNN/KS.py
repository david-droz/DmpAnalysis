'''

Sorts variables by KS statistics, then performs training on a gradually decreasing set of variables

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

def run():
	
	np.random.seed(5)
	

	train_e = np.load('/home/drozd/analysis/fraction1/data_train_elecs.npy')
	train_p = np.load('/home/drozd/analysis/fraction1/data_train_prots.npy')
	val_e = np.load('/home/drozd/analysis/fraction1/data_validate_elecs_1.npy') 
	val_p = np.load('/home/drozd/analysis/fraction1/data_validate_prots_1.npy') 

	mx = np.concatenate(( train_e[:,0:-2] , train_p[:,0:-2] )).max(axis=0)
	
	arr_elecs = train_e[:,0:-2] / mx
	arr_prots = train_p[:,0:-2] / mx
		
	train = np.concatenate(( train_e, train_p ))
	np.random.shuffle(train)
	X_train = train[:,0:-2] / (train[:,0:-2]).max(axis=0)
	Y_train = train[:,-1]
	del train_e,train_p, train
	
	val = np.concatenate(( val_e, val_p ))
	np.random.shuffle(val)
	X_val = val[:,0:-2] / (val[:,0:-2]).max(axis=0)
	Y_val = val[:,-1]
	del val_e, val_p, val
	
	l_pvalue = []
	l_KS = []
	for i in range(X_train.shape[1]):
		KS_statistic, p_value = stats.ks_2samp(arr_elecs[:,i],arr_prots[:,i])	# Kolmogorov-Smirnov test
		l_pvalue.append(p_value)												# If p-value is high, then the two distributions are likely the same
		l_KS.append(KS_statistic)												# If K-S statistic is high, then the two distributions are likely different.
	del arr_elecs, arr_prots
	
	KS_sorted = l_KS
	KS_sorted.sort()
	KS_sorted_indices = [ l_KS.index(x) for x in KS_sorted]
	with open("KSvalues.txt",'w') as f:
		f.write('VarIndex    KSvalue\n')
		for i in range(len(KS_sorted)):
			f.write(str(KS_sorted_indices[i])+'    '+str(KS_sorted[i])+'\n')
	
	if not os.path.isdir('results'):os.mkdir('results')
	
	if not os.path.isdir('images'): os.mkdir('images')

		
	############################################################################################################
	############################################################################################################
	############################################################################################################
	
	
		
	for n in range(1,X_train.shape[1]+1):
		
		touched='touch_'+str(n)
		
		outfile = "results/" + "%02d" % (n,) + '.pick'
		
		if os.path.isfile(outfile): continue
		
		if os.path.isfile(touched): continue
		
		with open(touched,'w') as fg:
			fg.write('a')
		
		try:
			bestP_values = heapq.nlargest(n,l_pvalue)
			bestP_indices = [ l_pvalue.index(x) for x in bestP_values]		# Those are the n variables we keep in step number n
			
			X_train_new = X_train[:,bestP_indices]
			X_val_new = X_val[:,bestP_indices]
			
			model = getModel(X_train_new)
			
			rdlronplt = ReduceLROnPlateau(monitor='loss',patience=3,min_lr=0.001)
			earl = EarlyStopping(monitor='loss',min_delta=0.0001,patience=5)
			callbacks = [rdlronplt,earl]
			
			history = model.fit(X_train_new,Y_train,batch_size=150,epochs=100,verbose=0,callbacks=callbacks,validation_data=(X_val_new,Y_val))
		
			predictions_proba = model.predict(X_val_new)
			predictions_binary = np.around(predictions_proba)
			del X_train_new, X_val_new
			
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
			plt.savefig('images/predHisto_'+str(n))
			plt.close(fig4)
			
			n_elecs_top = elecs_p[ elecs_p > 0.9 ].shape[0]
			n_prots_top = prots_p[ prots_p > 0.9 ].shape[0]
			contamination = n_prots_top / (n_elecs_top + n_prots_top)
			
			n_elecs_top_95 = elecs_p[ elecs_p > 0.95 ].shape[0]
			n_prots_top_95 = prots_p[ prots_p > 0.95 ].shape[0]
			contamination_95 = n_prots_top_95 / (n_elecs_top_95 + n_prots_top_95)
			
			
			
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
		except IndexError:
			os.remove(touched)
			continue
		except:
			os.remove(touched)
			raise
		
		os.remove(touched)
		del history, predictions_binary, predictions_proba, l_precision, l_recall, l_thresholds, elecs_p, prots_p
	# end for
	
	listofPicks = glob.glob('results/*.pick')
	listofPicks.sort()
	
	del X_train, X_val, Y_train, Y_val
	
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
	plt.savefig('AUC_F1')
	
	fig2 = plt.figure()
	plt.plot(nrofvariables,l_pr,'o-',label='Purity')
	plt.plot(nrofvariables,l_rc,'o-',label='Efficiency')
	plt.xlabel('Nr of variables')
	plt.ylabel('Score')
	plt.legend(loc='best')
	plt.savefig('PR-RC')
	
	fig3 = plt.figure()
	plt.plot(nrofvariables,l_contamination,'o-',label='cut at 0.9')
	plt.plot(nrofvariables,l_con_95,'o-',label='cut at 0.95')
	plt.xlabel('Nr of variables')
	plt.ylabel('p/(e+p) ratio')
	plt.legend(loc='best')
	plt.title('Background fraction')
	plt.savefig('Bkg')
	
	
	
	
	
	
		
		
	
if __name__ == '__main__' :
	
	run()
	
