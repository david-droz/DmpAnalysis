from __future__ import print_function, division, absolute_import
# Python3 compatibility needed

import numpy as np
import time
import pickle
import sys
import os
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Scikit-learn pre-defined metrics
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, average_precision_score, precision_recall_curve, recall_score
from sklearn.metrics import f1_score

# Keras deep neural networks
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler, ReduceLROnPlateau
from keras.constraints import maxnorm
from keras.layers.advanced_activations import PReLU, ELU, LeakyReLU

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
	model.add(Dense(75,kernel_initializer='he_uniform',activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(1,kernel_initializer='he_uniform',activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
	return model
	

def run():
	
	TRAIN_E_PATH = '/home/drozd/analysis/ntuples/MC-skim-fullBGO-NUD-HET-30Aug17/data_train_elecs_under_1.npy'
	TRAIN_P_PATH = '/home/drozd/analysis/ntuples/MC-skim-fullBGO-NUD-HET-30Aug17/data_train_prots_under_1.npy'
	VAL_E_PATH = '/home/drozd/analysis/ntuples/MC-skim-fullBGO-NUD-HET-30Aug17/data_validate_elecs_under_1.npy'
	VAL_P_PATH = '/home/drozd/analysis/ntuples/MC-skim-fullBGO-NUD-HET-30Aug17/data_validate_prots_under_1.npy'
	
	# Load training data and group it together
	train_e = np.load(TRAIN_E_PATH)
	train_p = np.load(TRAIN_P_PATH)
	train = np.concatenate(( train_e, train_p ))
	np.random.shuffle(train)
	
	X_train = train[:,0:-2]				# Last two variables are timestamp and particle ID, ignored for the "X" dataset
	Y_train = train[:,-1]				# Array of labels (particle ID, 0 for protons and 1 for electrons)
	del train_e,train_p, train
	
	val_e = np.load(VAL_E_PATH)
	val_p = np.load(VAL_P_PATH)[0:val_e.shape[0],:]
	val = np.concatenate(( val_e,val_p ))
	X_val = val[:,0:-2]
	Y_val = val[:,-1]
	del val_e,val_p,val

	# Normalisation!  The source of so many problems.
	X_max = X_train.max(axis=0)
	X_train = X_train / X_max
	X_val = X_val / X_max
	
	model = getModel(X_train)
	
	rdlronplt = ReduceLROnPlateau(monitor='loss',patience=3,min_lr=0.0001)			# Optimisation technique: if the loss does not decrease, reduce the learning rate
	callbacks = [rdlronplt]
	
	### Actual training is here!
	history = model.fit(X_train,Y_train,batch_size=100,epochs=100,verbose=2,callbacks=callbacks,validation_data=(X_val,Y_val))
	#####
	
	del X_train, Y_train
	
	model.save('trainedDNN.h5')			# Save the model for later use. Load with keras.models.load_model(filepath)
	model.save_weights('./my_nn_weights.h5',overwrite=True)
	with open('./my_nn_arch.json','w') as fout:
		fout.write(model.to_json())
	np.save("X_max.npy",X_max)
	
	# "Predictions": The neural networks now tries to guess particle identity on unseen data.
	predictions_proba = model.predict(X_val)				# Array of numbers between 0 and 1.
	predictions_binary = np.around(predictions_proba)		# Array of 0 and 1. Equivalent to a cut at 0.5
	del X_val
	
	# Prediction histogram, i.e. the graph that I keep showing
	elecs_p, prots_p = getClassifierScore(Y_val,predictions_proba)
	
	# Plot the prediction histogram and save it as a png file
	binList = [x/50 for x in range(0,51)]
	fig1 = plt.figure()
	plt.hist(elecs_p,bins=binList,label='e',alpha=1.,histtype='step',color='green')
	plt.hist(prots_p,bins=binList,label='p',alpha=1.,histtype='step',color='red',ls='dashed')
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.legend(loc='upper center')
	plt.grid(True)
	plt.ylim((0.9,1e+6))
	plt.yscale('log')
	plt.savefig('predHisto_MCorbit')
	plt.close(fig1)
	
	##############################
	##############################
	
	BT_E_PATH = '/home/drozd/analysis/ntuples/BT/BT_Electron_250G.npy'
	BT_P_PATH = '/home/drozd/analysis/ntuples/BT/BT_Proton_400G.npy'
	BTMC_E_PATH = '/home/drozd/analysis/ntuples/BT/MC_Electron_250G.npy'
	BTMC_P_PATH = '/home/drozd/analysis/ntuples/BT/MC_Proton_400G.npy'
	
	BT = np.concatenate(( np.load(BT_E_PATH), np.load(BT_P_PATH) ))
	X_BT = BT[:,0:-2] / X_max
	Y_BT = BT[:,-1]
	
	BTMC = np.concatenate(( np.load(BTMC_E_PATH), np.load(BTMC_P_PATH) ))
	X_BTMC = BTMC[:,0:-2] / X_max
	Y_BTMC = BTMC[:,-1]
	
	pred_BT = model.predict(X_BT)
	pred_BTMC = model.predict(X_BTMC)
	
	
	pred_e_BT, pred_p_BT = getClassifierScore(Y_BT,pred_BT)
	pred_e_BTMC, pred_p_BTMC = getClassifierScore(Y_BTMC,pred_BTMC)
	
	fig2 = plt.figure()
	plt.hist(pred_e_BT,bins=binList,label='e',alpha=1.,histtype='step',color='green')
	plt.hist(pred_p_BT,bins=binList,label='p',alpha=1.,histtype='step',color='red',ls='dashed')
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.legend(loc='upper center')
	plt.grid(True)
	plt.ylim((0.9,1e+6))
	plt.yscale('log')
	plt.savefig('predHisto_BT')
	plt.close(fig2)
	
	fig3 = plt.figure()
	plt.hist(pred_e_BTMC,bins=binList,label='e',alpha=1.,histtype='step',color='green')
	plt.hist(pred_p_BTMC,bins=binList,label='p',alpha=1.,histtype='step',color='red',ls='dashed')
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.legend(loc='upper center')
	plt.grid(True)
	plt.ylim((0.9,1e+6))
	plt.yscale('log')
	plt.savefig('predHisto_BTMC')
	plt.close(fig3)
	
	fig4 = plt.figure()
	plt.hist(pred_BT,bins=binList,label='BT data',alpha=1.,histtype='step')
	plt.hist(pred_BTMC,bins=binList,label='BT MC',alpha=1.,histtype='step')
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.legend(loc='upper center')
	plt.grid(True)
	plt.ylim((0.9,1e+6))
	plt.yscale('log')
	plt.savefig('predHisto_BTvsMC')
	plt.close(fig4)
	
	fig4b = plt.figure()
	plt.hist(pred_e_BT,bins=binList,label='e, BT data',alpha=1.,histtype='step')
	plt.hist(pred_e_BTMC,bins=binList,label='e, BT MC',alpha=1.,histtype='step')
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.legend(loc='upper center')
	plt.grid(True)
	plt.ylim((0.9,1e+6))
	plt.yscale('log')
	plt.savefig('predHisto_BTvsMC_e')
	plt.close(fig4b)
	
	fig4c = plt.figure()
	plt.hist(pred_p_BT,bins=binList,label='p, BT data',alpha=1.,histtype='step')
	plt.hist(pred_p_BTMC,bins=binList,label='p, BT MC',alpha=1.,histtype='step')
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.legend(loc='upper center')
	plt.grid(True)
	plt.ylim((0.9,1e+6))
	plt.yscale('log')
	plt.savefig('predHisto_BTvsMC_p')
	plt.close(fig4c)
	
	
	
			
	############################################################################################################
	############################################################################################################
	############################################################################################################
	
		
	
if __name__ == '__main__' :
	
	run()
	
	
