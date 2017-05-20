import numpy as np
import time
import cPickle as pickle
import sys
import os
import random
import hashlib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, average_precision_score, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV

def ParamsID(a):
	'''
	Using hash function to build an unique identifier for each dictionary
	'''
	ID = 1
	for x in a.keys():
		ID = ID + int(hashlib.sha256(str(a[x]).encode('utf-8')).hexdigest(), 16)

	return (ID % 10**10)

def getRandomParams():
	'''
	Returns a dictionary containing all the parameters to train the Scikit BDT.
	'''
	p_lr = [1.0,0.1,0.01,0.001,0.0001]
	p_n = [25,50,100,150,200,250,300,400,500]
	p_max = [2,3,4,5,6,3]
	p_leaves = [1,10,50,100,0.001,0.0001,1,1]
	
						
	mydic = {}
	mydic['lr'] = random.choice(p_lr)
	mydic['n'] = random.choice(p_n)
	mydic['max'] = random.choice(p_max)
	mydic['leaves'] = random.choice(p_leaves)

	return mydic

def XY_split(fname):
	arr = np.load(fname)
	X = arr[:,0:-1]
	Y = arr[:,-1]
	return X,Y
def load_training(fname='dataset_train.npy'): return XY_split(fname)
def load_validation(fname='dataset_validate.npy'): return XY_split(fname)
def load_test(fname='dataset_test.npy'): return XY_split(fname)


def _run(n_estimators=100,lr=1.0,maxdepth=3,testfraction=0.001,leaves=1):
	

	X_train, Y_train = load_training()
	X_val, Y_val = load_validation()
	
	params = getRandomParams()
	
	if not os.path.isdir('models'): os.mkdir('models')
	
	ID = ParamsID(params)										# Get an unique ID associated to the parameters
	with open('models/params_' + str(ID) + '.pick','w') as f:	# Save the parameters into a file determined by unique ID
		pickle.dump(params,f)
	
	clf = GradientBoostingClassifier(n_estimators=params['n'], learning_rate=params['lr'],max_depth=params['max'], min_samples_leaf=params['leaves'])
	clf.fit(X_train, Y_train)
	
	
	
	
	
	############
	###### Rework below this line
	############
	
	
	score = clf.score(X_val,Y_val)
	print "Score: ", score
	score_train = clf.score(X_train,Y_train)

	
	outstr = '/home/drozd/BDT/models/model_' + str(n_estimators) + '_' + str(lr)+'_'+str(maxdepth)+'_'+str(testfraction)+'_'+str(leaves)+'.pkl'
	joblib.dump(clf, outstr)
	
	prediction = clf.predict_proba(X_val)[:,1]			# Array of actual probabilities. Should be the one for electrons
														# Otherwise, use [:,0]
	
	prediction_binary = clf.predict(X_val)
	
	#~ fpr, tpr, thresholds = roc_curve(Y_val, prediction, pos_label=2)
	fpr, tpr, thresholds = roc_curve(Y_val, prediction)
	auc_score = roc_auc_score(Y_val, prediction)
	print "AUC: ", auc_score
	precision = precision_score(Y_val,prediction_binary)
	print "Precision: ", precision 
	
	precision_PRC, recall_PRC, thresholds_PRC = precision_recall_curve(Y_val, prediction)
	AU_PRC = average_precision_score(Y_val, prediction)
	
	#~ print "Min TPR: ", tpr[0]
	#~ print "Max TPR: ", tpr[-1]
	#~ print "Min FPR: ", fpr[0]
	#~ print "Max FPR: ", fpr[-1]
	
	outstr = '/home/drozd/BDT/results/results_' + str(n_estimators) + '_' + str(lr)+'_'+str(maxdepth)+'_'+str(testfraction)+'_'+str(leaves)+'.pickle'
	with open(outstr,'w') as f:
		pickle.dump([score,score_train,prediction,prediction_binary,precision,fpr,tpr,thresholds,auc_score],f)
	
	return score, fpr, tpr, thresholds, auc_score
	
	


if __name__ == '__main__' :
	
	t0 = time.time()
	
	try:
		n_estimators = int(sys.argv[1])
	except:
		print "Sys.argv[1] not recognised!"
		n_estimators = 100
	try:
		lr = float(sys.argv[2])
	except:
		print "Sys.argv[2] not recognised"
		lr = 1.
		
	try:
		maxdepth = int(sys.argv[3])
	except:
		print "Sys.argv[3] not recognised"
		maxdepth = 3
	
	try:
		feuille = int(sys.argv[4])
	except:
		print "Sys.argv[4] not recognised"
		feuille = 1
	
	for f in ['/home/drozd/BDT/test','/home/drozd/BDT/results','/home/drozd/BDT/models']:	
		if not os.path.isdir(f):
			os.mkdir(f)
	
	
	try:
		accuracy, fpr, tpr, thresholds, auc_score = _run(n_estimators=n_estimators,lr=lr,maxdepth=maxdepth,leaves=feuille)
	except KeyboardInterrupt:
		print "Interrupted"
		print "Total running time: ", time.strftime('%H:%M:%S', time.gmtime(time.time() - t0))	
		sys.exit()
	except Exception:
		raise

	#~ print "Testing accuracy: ", accuracy	
	print "Total running time: ", time.strftime('%H:%M:%S', time.gmtime(time.time() - t0))	
	
	
