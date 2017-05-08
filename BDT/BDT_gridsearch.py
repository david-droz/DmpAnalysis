import numpy as np
import time
import cPickle as pickle
import sys
import os

from scipy.stats import randint as sp_randint

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, average_precision_score, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV


def _run():
	
	
	testfraction = 0.001
	
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
	
	
	
	#  ------ Grid search ------ 
	my_val_fold = []
	for i in range(len(X_train)):
		my_val_fold.append(-1)
	for i in range(len(X_val)):
		my_val_fold.append(0)
	
	param_dist = {"max_depth": [1,2,3,4,5,6],
				"n_estimators":[10,30,50,100,300,500,1000,3000,5000],
				"min_samples_leaf": [1,10,100,1000,10000],
				"learning_rate" : [1,0.3,0.1,0.03,0.01,0.001,0.0001]}
	
	n_iter_search = 10
	random_search = RandomizedSearchCV(GradientBoostingClassifier(), param_distributions=param_dist,
										n_iter=n_iter_search,scoring='average_precision',n_jobs=-1,
										cv = PredefinedSplit(test_fold=my_val_fold),verbose=1)
	random_search.fit(np.concatenate((X_train, X_val), axis=0), np.concatenate((Y_train, Y_val), axis=0))
	
	print "Best parameters set found on development set:"
	print " "
	print random_search.best_params_
	
	clf = random_search.best_estimator_
	
	
	# ------ Model evaluation ------ 
	score = clf.score(X_val,Y_val)
	print "Score: ", score
	score_train = clf.score(X_train,Y_train)

	i = 0
	outstr = '/home/drozd/BDT/models/gridsearch' + '_' + str(i) + '.pkl'
	while os.path.isfile(outstr):
		i = i + 1
		outstr = '/home/drozd/BDT/models/gridsearch' + '_' + str(i) + '.pkl'
	
	joblib.dump(random_search, outstr)
	
	prediction = clf.predict_proba(X_val)[:,1]			# Array of actual probabilities. Should be the one for electrons
														# Otherwise, use [:,0]
	
	prediction_binary = clf.predict(X_val)
	
	fpr, tpr, thresholds = roc_curve(Y_val, prediction)
	auc_score = roc_auc_score(Y_val, prediction)
	print "AUC: ", auc_score
	precision = precision_score(Y_val,prediction_binary)
	print "Precision: ", precision 
	
	precision_PRC, recall_PRC, thresholds_PRC = precision_recall_curve(Y_val, prediction)
	AU_PRC = average_precision_score(Y_val, prediction)
	
	outstr = '/home/drozd/BDT/results/gridsearch_' + str(i) + '.pickle'
	with open(outstr,'w') as f:
		pickle.dump([score,score_train,prediction,prediction_binary,precision,
					fpr,tpr,thresholds,auc_score,precision_PRC, recall_PRC, thresholds_PRC,AU_PRC],f)
	
	
	


if __name__ == '__main__' :
	
	t0 = time.time()
	
	
	try:
		_run()
	except KeyboardInterrupt:
		print "Interrupted"
		print "Total running time: ", time.strftime('%H:%M:%S', time.gmtime(time.time() - t0))	
		sys.exit()
	except Exception:
		raise

	#~ print "Testing accuracy: ", accuracy	
	print "Total running time: ", time.strftime('%H:%M:%S', time.gmtime(time.time() - t0))	
	
	
