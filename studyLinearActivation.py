'''

Full study of a DNN method:

- Train a 4-layers Neural net
- After training, remove the sigmoid activation function from the last layer
- Evaluate performances on various energy bins
- Evaluate performances on beamtest data


'''

from __future__ import print_function, division, absolute_import

import numpy as np
import time
import pickle
import sys
import os
import glob
from uncertainties import ufloat

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score, precision_score, average_precision_score, precision_recall_curve, recall_score
from sklearn.metrics import f1_score

# Keras deep neural networks
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler, ReduceLROnPlateau

def getXTR(arr):
	
	FLast = arr[:,13]/arr[:,44]				# Fraction of energy in last layer
	sumRMS = np.zeros((arr.shape[0],))
	for i in range(14,28):
		#~ sumRMS += np.sqrt(arr[:,i])
		sumRMS += arr[:,i]
		
	return FLast * sumRMS**2 / 8e+6
	
def getXTRL(arr):
	
	energies = arr[:,0:14]
	FLast = energies[np.arange(energies.shape[0]),energies.shape[1] - 1 - (energies[:,::-1]!=0).argmax(1)]	# Fraction of energy in last non-zero layer
				# Black magic by Stack Overflow : https://stackoverflow.com/questions/39959435/set-last-non-zero-element-of-each-row-to-zero-numpy
				
	FLast = FLast / arr[:,44]
	
	sumRMS = np.zeros((arr.shape[0],))
	for i in range(14,28):
		#~ sumRMS += np.sqrt(arr[:,i])
		sumRMS += arr[:,i]
		
	return FLast * sumRMS**2 / 8e+6
	
	
def getCutBased(f,X_val,truth):
	
	elecs = X_val[ truth.astype(bool) ]
	prots = X_val[ ~truth.astype(bool) ]
	
	pred_e = f(elecs)
	pred_p = f(prots)
	
	return pred_e, pred_p
	
	
def getcountsXTRL(pred_e,pred_p,threshold):
	
	tp = pred_e[ pred_e <= threshold].shape[0]
	fn = pred_e[ pred_e > threshold].shape[0]
	fp = pred_p[ pred_p <= threshold].shape[0]
	tn = pred_p[ pred_p > threshold].shape[0]
	
	return tp, fp, tn, fn

def getcountsFast(truth,pred,threshold):
	pred_e = pred[truth.astype(bool)]
	pred_p = pred[~truth.astype(bool)]
	tp = pred_e[ pred_e >= threshold].shape[0]
	fn = pred_e[ pred_e < threshold].shape[0]
	fp = pred_p[ pred_p >= threshold].shape[0]
	tn = pred_p[ pred_p < threshold].shape[0]
	
	del pred_e, pred_p
	return tp, fp, tn, fn


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
	
def getLabels():
	'''
	Names of all the variables
	'''
	lab = []
	ebgo = 'BGO_E_layer_'
	for i in range(14):
		lab.append(ebgo + str(i))
	erms = 'BGO_E_RMS_layer_'
	for i in range(14):
		lab.append(erms + str(i))
	ehit = 'BGO_E_HITS_layer_'
	for i in range(14):
		lab.append(ehit + str(i))
	lab.append('BGO_RMS_longitudinal')
	lab.append('BGO_RMS_radial')
	#~ lab.append('BGO_E_total_corrected')
	lab.append('BGO_E_total')
	lab.append('BGO_total_hits')
	lab.append('BGO_theta_angle')
	
	return lab	
	
	############################################################################################################
	############################################################################################################
	############################################################################################################
	
TRAIN_E_PATH = '/home/drozd/analysis/ntuples/MC-skim-fullBGO-NUD-HET-01Nov17/data_train_elecs_under_1.npy'
TRAIN_P_PATH = '/home/drozd/analysis/ntuples/MC-skim-fullBGO-NUD-HET-01Nov17/data_train_prots_under_1.npy'
VAL_E_PATH = '/home/drozd/analysis/ntuples/MC-skim-fullBGO-NUD-HET-01Nov17/data_validate_elecs_under_1.npy'
VAL_P_PATH = '/home/drozd/analysis/ntuples/MC-skim-fullBGO-NUD-HET-01Nov17/data_validate_prots_under_1.npy'

def train(n_epochs=200):
	
	train_e = np.load(TRAIN_E_PATH)
	train_p = np.load(TRAIN_P_PATH)
	train = np.concatenate(( train_e, train_p ))
	np.random.shuffle(train)
	
	X_train = train[:,0:47]
	Y_train = train[:,-1]
	E_train = train[:,44]
	del train_e,train_p, train
	
	val_e = np.load(VAL_E_PATH)
	val_p = np.load(VAL_P_PATH)
	val = np.concatenate(( val_e, val_p ))
	X_val = val[:,0:47]
	E_val = val[:,44]
	Y_val = val[:,-1]
	evtWeight = val[:,-2]
	del val
	
	X_max = X_train.max(axis=0)
	X_train = X_train/X_max
	X_val = X_val / X_max
	np.save('X_max.npy',X_max)
	
	modelName = 'trainedDNN_'+str(n_epochs)+'.h5'
	historyName = 'trainHistory_'+str(n_epochs)+'.pick'
	
	model = getModel(X_train)
	
	rdlronplt = ReduceLROnPlateau(monitor='loss',patience=3,min_lr=0.0001)	
	callbacks = [rdlronplt]
	
	#####
	history = model.fit(X_train,Y_train,batch_size=100,epochs=n_epochs,verbose=2,callbacks=callbacks,validation_data=(X_val,Y_val))
	#####
	
	model.save_weights('weights_'+str(n_epochs)+'.h5')
	model.save('model_sigmoid_'+str(n_epochs)+'.h5')

	pickle.dump(history.history,open(historyName,'wb'),protocol=2)
	histo = history.history
	
	fig1 = plt.figure()
	plt.plot(histo['loss'],label='loss')
	plt.plot(histo['val_loss'],label='val_loss')
	plt.legend(loc='best')
	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.title('train history')
	plt.savefig('history')
	plt.close(fig1)
	
	predictions = model.predict(X_val)
	
	elecs_p, prots_p = getClassifierScore(Y_val,predictions)
	weights_e = evtWeight[ Y_val.astype(bool) ]
	weights_p = evtWeight[ ~Y_val.astype(bool)]
	
	fig2 = plt.figure()
	binList = [x/50 for x in range(0,51)]
	plt.hist(elecs_p,bins=binList,label='e',alpha=1.,histtype='step',color='green',weights=weights_e)
	plt.hist(prots_p,bins=binList,label='p',alpha=1.,histtype='step',color='red',weights=weights_p)
	#~ plt.hist(elecs_p,bins=binList,label='e unweighted',alpha=1.,histtype='step',color='green',ls='dashed')
	#~ plt.hist(prots_p,bins=binList,label='p unweighted',alpha=1.,histtype='step',color='red',ls='dashed')
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.legend(loc='upper center')
	plt.grid(True)
	plt.ylim((0.9,1e+7))
	plt.yscale('log')
	plt.savefig('classScore_sigmoid_allbins')
	plt.close(fig2)
	
	model3 = Sequential()
	model3.add(Dense(300,input_shape=(X_train.shape[1],),kernel_initializer='he_uniform',activation='relu'))
	model3.add(Dropout(0.1))
	model3.add(Dense(150,kernel_initializer='he_uniform',activation='relu'))
	model3.add(Dropout(0.1))
	model3.add(Dense(75,kernel_initializer='he_uniform',activation='relu'))
	model3.add(Dropout(0.1))
	model3.add(Dense(1,kernel_initializer='he_uniform'))
	model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
	
	for i,x in enumerate(model.layers):
		
		weights = x.get_weights()
		model3.layers[i].set_weights(weights)
		
	elecs_p, prots_p = getClassifierScore(Y_val,model3.predict(X_val))
	
	fig4 = plt.figure()
	binList = [x/50 for x in range(0,51)]
	
	e_redux = elecs_p[elecs_p < 300]
	p_redux = prots_p[prots_p > -500]

	w_e_redux = weights_e.reshape((weights_e.shape[0],1))[elecs_p < 300]
	w_p_redux = weights_p.reshape((weights_p.shape[0],1))[prots_p > -500]
	
	binList = [i for i in range(p_redux.min(),e_redux.max())]

	plt.hist(e_redux,bins=binList,label='e',alpha=1.,histtype='step',color='green',weights=w_e_redux)
	plt.hist(p_redux,bins=binList,label='p',alpha=1.,histtype='step',color='red',weights=w_p_redux)
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.legend(loc='upper center')
	plt.grid(True)
	plt.ylim((0.9,1e+7))
	plt.xlim((-300,300))
	plt.yscale('log')
	plt.savefig('classScore_linear_allbins')
	plt.close(fig4)
	
	model3.save(modelName)

	
def evaluation(e_min,e_max,modelname):	
	val_e = np.load(VAL_E_PATH)
	val_p = np.load(VAL_P_PATH)
	
	val_e = val_e[ val_e[:,44] < e_max ]
	val_e = val_e[ val_e[:,44] >= e_min ]
	
	val_p = val_p[ val_p[:,44] < e_max ]
	val_p = val_p[ val_p[:,44] >= e_min ]
	
	val_p = val_p[0:val_e.shape[0],:]
	val = np.concatenate(( val_e, val_p ))
	
	X_max = np.load('X_max.npy')
	
	X_val = val[:,0:47]/X_max
	E_reco = val[:,44]
	E_truth = val[:,67]
	Y_val = val[:,-1]
	evtWeight = val[:,-2]
	
	model = load_model(modelname)
	model_sig = load_model(modelname.replace('trainedDNN','model_sigmoid'))
	
	predictions = model.predict(X_val)
	predictions_sigmoid = model_sig.predict(X_val)
	
	elecs_p, prots_p = getClassifierScore(Y_val,predictions)
	weights_e = evtWeight[ Y_val.astype(bool) ]
	weights_p = evtWeight[ ~Y_val.astype(bool)]
	
	prots_p_redux = prots_p[prots_p > -500]
	elecs_p_redux = elecs_p[elecs_p < 300]

	weights_e = weights_e.reshape((weights_e.shape[0],1))[elecs_p < 300]
	weights_p = weights_p.reshape((weights_p.shape[0],1))[prots_p > -500]
	
	binList = [i for i in range(prots_p_redux.min(),elecs_p_redux.max())]
	fig1 = plt.figure()
	plt.hist(elecs_p_redux,bins=binList,label='e',alpha=1.,histtype='step',color='green',weights=weights_e)
	plt.hist(prots_p_redux,bins=binList,label='p',alpha=1.,histtype='step',color='red',weights=weights_p)
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.legend(loc='upper center')
	plt.grid(True)

	if e_min/1000 < 70:
		plt.xlim((-200,100))
		plt.ylim((0.9,0.7e+6))
	elif e_min/1000 < 500:
		plt.xlim((-100,50))
		plt.ylim((0.1,3e+4))
	elif e_min/1000 < 3000:
		plt.xlim((-100,50))
		plt.ylim((0.01,3e+2))
	elif e_min/1000 > 3000:
		plt.xlim((-100,70))
		plt.ylim((0.005,8e+1))
	plt.yscale('log')
	plt.title('DNN-linear \n'+str(int(e_min/1000))+'-'+str(int(e_max/1000))+' GeV')
	plt.savefig('predHisto/predHisto_'+str(int(e_min/1000))+'-'+str(int(e_max/1000)))
	plt.close(fig1)
	
	
	# Next steps: make a ROC curve for every bin, compare with XTR (or XTRL)
	# One plot per bin for the comparison
	# Then, one plot to show the ROC at all bins
	
	#~ XTR_e,XTR_p = getCutBased(getXTR,X_val,Y_val)
	XTRL_e, XTRL_p = getCutBased(getXTRL,X_val,Y_val)
	
	l_bkg = []
	l_eff = []
	l_bkg_xtr = []
	l_eff_xtr = []
	l_bkg_xtrl = []
	l_eff_xtrl = []
	npoints = 5000
	
	fig1 = plt.figure()
	
	eff_90 = 1
	eff_90_xtr = 1
	eff_90_xtrl = 1
	bkg_90 = 0
	bkg_90_xtr = 0
	bkg_90_xtrl = 0
	
	for i in range(npoints):
		thr = (-100) + i * (((+300) - (-100))/npoints)
		#~ thr_xtr = i * (np.max(XTR_e)/npoints)
		thr_sigmoid = i*(1./npoints)
		thr_xtrl = i * (np.max(XTRL_e)/npoints)
		
		tp,fp,tn,fn = getcountsFast(Y_val,predictions,thr)
		#~ tp_xtr,fp_xtr,tn_xtr,fn_xtr = getcountsXTRL(XTR_e,XTR_p,thr_xtr)
		tp_xtr,fp_xtr,tn_xtr,fn_xtr = getcountsFast(Y_val,predictions_sigmoid,thr_sigmoid)
		tp_xtrl,fp_xtrl,tn_xtrl,fn_xtrl = getcountsXTRL(XTRL_e,XTRL_p,thr_xtrl)
		
		tp,fp,tn,fn = [ufloat(x,np.sqrt(x)) for x in [tp,fp,tn,fn ]]
		tp_xtr,fp_xtr,tn_xtr,fn_xtr = [ufloat(x,np.sqrt(x)) for x in [tp_xtr,fp_xtr,tn_xtr,fn_xtr]]
		tp_xtrl,fp_xtrl,tn_xtrl,fn_xtrl = [ufloat(x,np.sqrt(x)) for x in [tp_xtrl,fp_xtrl,tn_xtrl,fn_xtrl]]
		
		try:
			bk = fp / (tp+fp)
		except ZeroDivisionError :
			bk = ufloat(1,0)
		try:
			bk_xtr = fp_xtr/(tp_xtr+fp_xtr)
		except ZeroDivisionError :
			bk_xtr = ufloat(1,0)
		try:
			bk_xtrl = fp_xtrl/(tp_xtrl+fp_xtrl)
		except ZeroDivisionError :
			bk_xtrl = ufloat(1,0)
		
		eff = tp / (tp + fn)
		eff_xtr = tp_xtr / (tp_xtr + fn_xtr)
		eff_xtrl = tp_xtrl / (tp_xtrl + fn_xtrl)
		
		if eff > 0.95 and eff < eff_90:
			eff_90 = eff
			bkg_90 = bk			
		if eff_xtr > 0.95 and eff_xtr < eff_90_xtr:
			eff_90_xtr = eff_xtr
			bkg_90_xtr = bk_xtr
		if eff_xtrl > 0.95 and eff_xtrl < eff_90_xtrl:
			eff_90_xtrl = eff_xtrl
			bkg_90_xtrl = bk_xtrl
			
		l_bkg.append( bk.n )
		l_bkg_xtr.append( bk_xtr.n )
		l_eff.append( eff.n )
		l_eff_xtr.append( eff_xtr.n )
		l_bkg_xtrl.append( bk_xtrl.n )
		l_eff_xtrl.append( eff_xtrl.n )
	
	print('-----',str(int(e_min/1000)),' - ',str(int(e_max/1000)), ' GeV -----')
	print("Linear, background at efficiency ", eff_90, " : ", bkg_90)
	print("Sigmoid, background at efficiency ", eff_90_xtr, " : ", bkg_90_xtr)
	print("XTRL, background at efficiency ", eff_90_xtrl, " : ", bkg_90_xtrl)
	
		
	with open('pickles/energy_roc_'+str(int(e_min/1000))+'.pick','wb') as f:
		pickle.dump([l_bkg,l_eff,e_min,e_max],f,protocol=2)
		
	plt.plot([x for x in l_eff],l_bkg,label='DNN-linear')
	plt.plot([x for x in l_eff_xtr],l_bkg_xtr,label='DNN-sigmoid')
	plt.plot([x for x in l_eff_xtrl],l_bkg_xtrl,label='XTRL')
	plt.xlabel('Efficiency')
	plt.ylabel('Background fraction')
	plt.title(str(int(e_min/1000))+'-'+str(int(e_max/1000))+' GeV')
	#plt.xscale('log')
	plt.xlim((0.85,1.01))
	plt.yscale('log')
	plt.legend(loc='best')
	plt.savefig('ROC/roc_'+str(int(e_min/1000)))
	plt.close(fig1)
		
def rocEnergies(energies,N_bins):
	
	fig1 = plt.figure()
	
	for i in range(N_bins):
		with open('pickles/energy_roc_'+str(int(energies[i]/1000))+'.pick','rb') as f:
			bk,ef,emin,emax = pickle.load(f)
		label = str(int(energies[i]/1000))+'-'+str(int(energies[i+1]/1000))+'GeV'
		plt.plot([x for x in ef],bk,label=label)
	plt.xlabel('Efficiency')
	plt.ylabel('Background fraction')
	#plt.xscale('log')
	plt.xlim((0.85,1.01))
	plt.yscale('log')
	plt.legend(loc='best')
	plt.savefig('roc_DNN_linear')
	
def beamTest(modelname):
	
	from matplotlib.backends.backend_pdf import PdfPages
	
	X_max = np.load('X_max.npy')
	model = load_model(modelname)
	
	BT_E_PATH = '/home/drozd/analysis/ntuples/BT/BT_Electron_250G.npy'
	BT_P_PATH = '/home/drozd/analysis/ntuples/BT/BT_Proton_400G.npy'
	BTMC_E_PATH = '/home/drozd/analysis/ntuples/BT/MC_Electron_250G.npy'
	BTMC_P_PATH = '/home/drozd/analysis/ntuples/BT/MC_Proton_400G.npy'
	
	pp_e = PdfPages('beamtest/vars_e.pdf')
	pp_p = PdfPages('beamtest/vars_p.pdf')
	
	arr_bt_e = np.load(BT_E_PATH)
	arr_bt_p = np.load(BT_P_PATH)
	arr_mc_e = np.load(BTMC_E_PATH)
	arr_mc_p = np.load(BTMC_P_PATH)
	
	labs = getLabels()
	for i in range(47):
		
		fig_t_e = plt.figure()
		plt.hist(arr_bt_e[:,i],50,normed=True,histtype='step',label='BT')
		plt.hist(arr_mc_e[:,i],50,normed=True,histtype='step',label='MC')
		plt.legend(loc='best')
		plt.yscale('log')
		plt.title('Electron 250 GeV \n' + labs[i] )
		plt.savefig(pp_e, format='pdf')
		if i in [44,45,0,13,15,16,14]:
			plt.savefig('beamtest/vars/electron_'+labs[i])
		plt.close(fig_t_e)
		
		fig_t_p = plt.figure()
		plt.hist(arr_bt_p[:,i],50,normed=True,histtype='step',label='BT')
		plt.hist(arr_mc_p[:,i],50,normed=True,histtype='step',label='MC')
		plt.legend(loc='best')
		plt.yscale('log')
		plt.title('Proton 400 GeV \n' + labs[i] )
		plt.savefig(pp_p, format='pdf')
		if i in [44,45,0,13,15,16,14]:
			plt.savefig('beamtest/vars/proton_'+labs[i])
		plt.close(fig_t_p)
	pp_e.close()
	pp_p.close()
	
	
	
	BT = np.concatenate(( arr_bt_e, arr_bt_p ))
	X_BT = BT[:,0:47] / X_max
	Y_BT = BT[:,-1]
	
	BTMC = np.concatenate(( arr_mc_e, arr_mc_p ))
	X_BTMC = BTMC[:,0:47] / X_max
	Y_BTMC = BTMC[:,-1]
	
	pred_BT = model.predict(X_BT)
	pred_BTMC = model.predict(X_BTMC)
	
	pred_e_BT, pred_p_BT = getClassifierScore(Y_BT,pred_BT)
	pred_e_BTMC, pred_p_BTMC = getClassifierScore(Y_BTMC,pred_BTMC)
	
	fig2 = plt.figure()
	pred_p_BT_redux = pred_p_BT[pred_p_BT > -110]
	binList = [ -100 + i*(30 - (-100))/50 for i in range(51) ]
	plt.hist(pred_e_BT,bins=binList,label='Electron 250 GeV',alpha=1.,histtype='step',color='green')
	plt.hist(pred_p_BT_redux,bins=binList,label='Proton 400 GeV',alpha=1.,histtype='step',color='red',ls='dashed')
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.title('Beamtest data')
	plt.legend(loc='upper center')
	plt.grid(True)
	plt.ylim((0.9,5e+5))
	plt.xlim((-100,30))
	plt.yscale('log')
	plt.savefig('beamtest/predHisto_BT')
	plt.close(fig2)
	
	fig3 = plt.figure()
	binList = [ -20 + i*(20 - (-20))/50 for i in range(51) ]
	plt.hist(pred_e_BTMC,bins=binList,label='Electron 250 GeV',alpha=1.,histtype='step',color='green')
	plt.hist(pred_p_BTMC,bins=binList,label='Proton 400 GeV',alpha=1.,histtype='step',color='red',ls='dashed')
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.title('Beamtest Monte-Carlo')
	plt.legend(loc='upper center')
	plt.grid(True)
	plt.ylim((0.8,1e+5))
	plt.xlim((-20,20))
	plt.yscale('log')
	plt.savefig('beamtest/predHisto_BTMC')
	plt.close(fig3)
	
	fig4 = plt.figure()
	pred_BT_redux = pred_BT[pred_BT > -60]
	binList = [ -50 + i*(20 - (-50))/50 for i in range(51) ]
	plt.hist(pred_BT_redux,bins=binList,label='BT data',alpha=1.,histtype='step',normed=True)
	plt.hist(pred_BTMC,bins=binList,label='BT MC',alpha=1.,histtype='step',normed=True)
	plt.xlabel('Classifier score')
	plt.ylabel('Fraction of events')
	plt.title('Beamtest, electron 250GeV, proton 400GeV')
	plt.legend(loc='upper right')
	plt.grid(True)
	plt.xlim((-50,50))
	#~ plt.ylim((0.9,1e+6))
	plt.yscale('log')
	plt.savefig('beamtest/predHisto_BTvsMC')
	plt.close(fig4)
	
	fig4b = plt.figure()
	binList = [i for i in range( min([pred_e_BT.min(),pred_e_BTMC.min()]), max([pred_e_BT.max(),pred_e_BTMC.max()]))]
	plt.hist(pred_e_BT,bins=binList,label='BT data',alpha=1.,histtype='step',normed=True)
	plt.hist(pred_e_BTMC,bins=binList,label='BT MC',alpha=1.,histtype='step',normed=True)
	plt.xlabel('Classifier score')
	plt.ylabel('Fraction of events')
	plt.title('Beamtest, electron 250 GeV')
	plt.legend(loc='upper center')
	plt.grid(True)
	#~ plt.ylim((0.9,1e+6))
	plt.yscale('log')
	plt.savefig('beamtest/predHisto_BTvsMC_e')
	plt.close(fig4b)
	
	fig4c = plt.figure()
	binList = [ -100 + i*(20 - (-100))/50 for i in range(51) ]
	plt.hist(pred_p_BT,bins=binList,label='p, BT data',alpha=1.,histtype='step',normed=True)
	plt.hist(pred_p_BTMC,bins=binList,label='p, BT MC',alpha=1.,histtype='step',normed=True)
	plt.xlabel('Classifier score')
	plt.ylabel('Fraction of events')
	plt.title('Beamtest, proton 400 GeV')
	plt.legend(loc='upper center')
	plt.grid(True)
	plt.xlim((-100,20))
	#~ plt.ylim((0.9,1e+6))
	plt.yscale('log')
	plt.savefig('beamtest/predHisto_BTvsMC_p')
	plt.close(fig4c)
	############################################################################################################
	############################################################################################################
	############################################################################################################
	
	
def plotXTR():
	
	train_e = np.load(TRAIN_E_PATH)
	train_p = np.load(TRAIN_P_PATH)
	
	XTR_e = getXTR(train_e)
	XTR_p = getXTR(train_p)
	XTRL_e = getXTRL(train_e)
	XTRL_p = getXTRL(train_p)
	
	XTR_p = XTR_p[XTR_p < 1.1*XTR_e.max()]
	XTRL_p = XTRL_p[XTRL_p < 1.1*XTRL_e.max()]
	
	binList = [i for i in range(0,XTR_p.max())]
	
	fig1 = plt.figure()
	plt.hist(XTR_e,bins=binList,histtype='step',label='e')
	plt.hist(XTR_p,bins=binList,histtype='step',label='p')
	plt.title('XTR')
	plt.xlabel('XTR')
	plt.yscale('log')
	plt.legend(loc='upper right')
	plt.savefig('xtr/xtr')
	plt.close(fig1)
	
	binList = [i for i in range(0,XTRL_p.max())]
	fig2 = plt.figure()
	plt.hist(XTRL_e,bins=binList,histtype='step',label='e')
	plt.hist(XTRL_p,bins=binList,histtype='step',label='p')
	plt.title('XTRL')
	plt.xlabel('XTRL')
	plt.yscale('log')
	plt.legend(loc='upper right')
	plt.savefig('xtr/xtrl')
	plt.close(fig2)
		
		
		
		
	############################################################################################################
	############################################################################################################
		
	
if __name__ == '__main__' :
	
	n_epochs = 80
	
	for x in ['predHisto','ROC','pickles','beamtest','beamtest/vars','xtr']:
		if not os.path.isdir(x):
			os.mkdir(x)
			
	plotXTR()
	
	if not os.path.isfile('trainedDNN_'+str(n_epochs)+'.h5'):
		train(n_epochs)
		
	N_bins = 7
	energies = np.geomspace(10 * 1e+3,10 * 1e+6,N_bins+1)
	for i in range(N_bins):
		evaluation(energies[i],energies[i+1],'trainedDNN_'+str(n_epochs)+'.h5')
		
	rocEnergies(energies,N_bins)
	
	beamTest('trainedDNN_'+str(n_epochs)+'.h5')

	
	
