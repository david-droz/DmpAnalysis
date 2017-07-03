'''

For the same trained model, compare results in balanced vs imbalanced

'''

from __future__ import division, print_function, absolute_import

import numpy as np
import time
import pickle
import sys
import os
import random
import hashlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, average_precision_score, precision_recall_curve, recall_score
from sklearn.metrics import f1_score
from sklearn.externals import joblib


##############################################

def ParamsID(a):
	'''
	Using hash function to build an unique identifier for each dictionary
	'''
	ID = 1
	for x in a.keys():
		ID = ID + int(hashlib.sha256(str(a[x]).encode('utf-8')).hexdigest(), 16)

	return (ID % 10**10)

def XY_split(fname):
	arr = np.load(fname)
	X = arr[:,0:-2]				### Last two columns are timestamp and particle id
	Y = arr[:,-1]
	return X,Y

def _normalise(arr):
	#~ for i in range(arr.shape[1]):
		#~ if np.all(arr[:,i] > 0) :
			#~ arr[:,i] = (arr[:,i] - np.mean(arr[:,i]) + 1.) / np.std(arr[:,i])		# Mean = 1 if all values are strictly positive (from paper)
		#~ else:
			#~ arr[:,i] = (arr[:,i] - np.mean(arr[:,i])) / np.std(arr[:,i])	
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
	
	thr = 1.
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
	
def getSets():
	
	electrons_all = np.concatenate(( np.load('/home/drozd/analysis/fraction1/data_validate_elecs_1.npy') , np.load('/home/drozd/analysis/fraction1/data_test_elecs_1.npy')))
	protons_all = np.concatenate(( np.load('/home/drozd/analysis/fraction1/data_validate_prots_1.npy') , np.load('/home/drozd/analysis/fraction1/data_test_prots_1.npy')))
	
	bigarr = np.concatenate(( electrons_all , protons_all ))
	np.random.shuffle(bigarr)
	
	imbaArr = np.concatenate(( protons_all , electrons_all[ 0:int(protons_all.shape[0]/100) ] ))
	np.random.shuffle(imbaArr)
	
	return bigarr[:,0:-2], bigarr[:,-1], imbaArr[:,0:-2], imbaArr[:,-1]
	

	
def getClassifierScore(truth,pred):
	elecs = pred[truth.astype(bool)]
	prots = pred[~truth.astype(bool)]
			
	return elecs, prots

	
def run():
	
	X_train, Y_train = XY_split('/home/drozd/analysis/dataset_train.npy')
	#~ X_val, Y_val = XY_split('/home/drozd/analysis/fraction1/dataset_validate_1.npy')
	#~ X_val_imba, Y_val_imba = XY_split('/home/drozd/analysis/dataset_validate.npy')
	X_val, Y_val, X_val_imba, Y_val_imba = getSets()
	
	X_train = _normalise(X_train)
	X_val = _normalise(X_val)
	X_val_imba = _normalise(X_val_imba)
	
	
	params = {'lr' : 0.1,
				'n' : 100,
				'max' : 3,
				'leaves' : 0.0001 }
					
	ID = ParamsID(params)
	modelFile = str(ID)+'.pick'
	
	if not os.path.isfile(modelFile):
		model = GradientBoostingClassifier(n_estimators=params['n'], learning_rate=params['lr'],max_depth=params['max'], min_samples_leaf=params['leaves'])
		model.fit(X_train, Y_train)
		joblib.dump(model,modelFile)
	else:
		model = joblib.load(modelFile)
	# --------------------------------
	
	predictions_balanced = model.predict_proba(X_val)[:,1]
	predictions_imba = model.predict_proba(X_val_imba)[:,1]
	predictions_train = model.predict_proba(X_train)[:,1]
	
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
	plt.plot(sk_l_precision_b, sk_l_recall_b,label='balanced, sk')
	plt.plot(sk_l_precision_i, sk_l_recall_i,label='imbalanced, sk')
	plt.plot(sk_l_precision_t, sk_l_recall_t,label='training set')
	plt.plot(man_l_precision_b, man_l_recall_b,'o',label='balanced, hand')
	plt.plot(man_l_precision_i, man_l_recall_i,'o',label='imbalanced, hand')
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
	plt.plot(sk_l_fpr_b, sk_l_tpr_b,label='balanced, sk')
	plt.plot(sk_l_fpr_i, sk_l_tpr_i,label='imbalanced, sk')
	plt.plot(man_l_fpr_b, man_l_tpr_b,'o',label='balanced, hand')
	plt.plot(man_l_fpr_i, man_l_tpr_i,'o',label='imbalanced, hand')
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
	
	
	elecs_t, prots_t = getClassifierScore(Y_train,predictions_train)
	fig3 = plt.figure()
	plt.hist(elecs_t,50,label='e',alpha=0.5,histtype='step',color='green')
	plt.hist(prots_t,50,label='p',alpha=0.5,histtype='step',color='red')
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.title('Training set')
	plt.legend(loc='best')
	plt.yscale('log')
	plt.savefig('predHisto_train')
	
	fig3b = plt.figure()
	plt.hist(elecs_t,50,label='e',alpha=0.5,histtype='step',color='green',normed=True)
	plt.hist(prots_t,50,label='p',alpha=0.5,histtype='step',color='red',normed=True)
	plt.xlabel('Classifier score')
	plt.ylabel('Fraction of events')
	plt.title('Training set - normalised')
	plt.legend(loc='best')
	plt.yscale('log')
	plt.savefig('predHisto_train_n')	
	del elecs_t, prots_t, Y_train, predictions_train
	
	elecs_b, prots_b = getClassifierScore(Y_val,predictions_balanced)
	fig4 = plt.figure()
	plt.hist(elecs_b,50,label='e',alpha=0.5,histtype='step',color='green')
	plt.hist(prots_b,50,label='p',alpha=0.5,histtype='step',color='red')
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.title('Balanced validation set')
	plt.legend(loc='best')
	plt.yscale('log')
	plt.savefig('predHisto_bal')
	
	fig4b = plt.figure()
	plt.hist(elecs_b,50,label='e',alpha=0.5,histtype='step',color='green',normed=True)
	plt.hist(prots_b,50,label='p',alpha=0.5,histtype='step',color='red',normed=True)
	plt.xlabel('Classifier score')
	plt.ylabel('Fraction of events')
	plt.title('Balanced validation set - normalised')
	plt.legend(loc='best')
	plt.yscale('log')
	plt.savefig('predHisto_bal_n')	
	del elecs_b, prots_b, Y_val, predictions_balanced
	
	elecs_i, prots_i = getClassifierScore(Y_val_imba,predictions_imba)
	fig5 = plt.figure()
	plt.hist(elecs_i,50,label='e',alpha=0.5,histtype='step',color='green')
	plt.hist(prots_i,50,label='p',alpha=0.5,histtype='step',color='red')
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.legend(loc='best')
	plt.title('Imbalanced validation set')
	plt.yscale('log')
	plt.savefig('predHisto_imba')
	
	fig5b = plt.figure()
	plt.hist(elecs_i,50,label='e',alpha=0.5,histtype='step',color='green',normed=True)
	plt.hist(prots_i,50,label='p',alpha=0.5,histtype='step',color='red',normed=True)
	plt.xlabel('Classifier score')
	plt.ylabel('Fraction of events')
	plt.title('Imbalanced validation set - normalised')
	plt.legend(loc='best')
	plt.yscale('log')
	plt.savefig('predHisto_imba_n')	
	
	
	
	electrons_all = np.concatenate(( np.load('/home/drozd/analysis/fraction1/data_validate_elecs_1.npy') , np.load('/home/drozd/analysis/fraction1/data_test_elecs_1.npy')))
	protons_all = np.concatenate(( np.load('/home/drozd/analysis/fraction1/data_validate_prots_1.npy') , np.load('/home/drozd/analysis/fraction1/data_test_prots_1.npy')))
	e_score, e_garbage = getClassifierScore( electrons_all[:,-1] ,  model.predict_proba(_normalise(electrons_all[:,0:-2]))[:,1] )
	p_garbage, p_score = getClassifierScore( protons_all[:,-1] ,  model.predict_proba(_normalise(protons_all[:,0:-2]))[:,1] )
		
	fig6 = plt.figure()
	plt.hist( e_score, 50, label='e',alpha=0.5,histtype='step',color='green',normed=False)
	plt.hist( p_score, 50, label='p',alpha=0.5,histtype='step',color='red',normed=False)
	plt.xlabel('Classifier score')
	plt.ylabel('Fraction of events')
	plt.title('Balanced set - loaded separately')
	plt.legend(loc='best')
	plt.yscale('log')
	plt.savefig('predHisto_ba_loadSeparate')	
	
	fig6b = plt.figure()
	plt.hist( e_score, 50, label='e',alpha=0.5,histtype='step',color='green',normed=True)
	plt.hist( p_score, 50, label='p',alpha=0.5,histtype='step',color='red',normed=True)
	plt.xlabel('Classifier score')
	plt.ylabel('Fraction of events')
	plt.title('Balanced set - loaded separately - normed')
	plt.legend(loc='best')
	plt.yscale('log')
	plt.savefig('predHisto_ba_loadSeparate_n')	




if __name__ == '__main__' :
	
	run()

