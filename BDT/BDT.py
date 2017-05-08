import numpy as np
import time
import cPickle as pickle
import sys
import os

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, average_precision_score, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV


def _run(n_estimators=100,lr=1.0,maxdepth=3,testfraction=0.001,leaves=1):
	

	e_data = np.load('/home/drozd/BDT/dataset_elec.npy')
	e_data = e_data[np.any(e_data,axis=1)]  
	p_data = np.load('/home/drozd/BDT/dataset_prot.npy')
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
	
	if os.path.isfile('/home/drozd/BDT/test/validation_set.npy'):
		test = np.load('/home/drozd/BDT/test/validation_set.npy')
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
	
	
	clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=lr,max_depth=maxdepth, min_samples_leaf=leaves)
	clf.fit(X_train, Y_train)
	
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
	
	
