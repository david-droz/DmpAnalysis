'''

Full study of a DNN method:

- Train a 4-layers Neural net
- After training, remove the sigmoid activation function from the last layer
- Evaluate performances on various energy bins
- Evaluate performances on beamtest data

Pretty ugly code, growing spaghetti. Would need rework.
	Also would need commenting

'''

from __future__ import print_function, division, absolute_import

import numpy as np
import time
import pickle
import sys
import os
import glob
from uncertainties import ufloat
from argparse import ArgumentParser

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score, precision_score, average_precision_score, precision_recall_curve, recall_score
from sklearn.metrics import f1_score

# Keras deep neural networks
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler, ReduceLROnPlateau


####
# BEGIN: CONVENIENCE FUNCTIONS
####

def _now():
	return time.strftime("%Y-%m-%d %H:%M:%S")

def loadData(path):
	def _findIt(particle,t):			# Useless for the moment. 
		a = glob.glob(path+'*.npy')		# Intended use: if the datasets are not called the same way anymore
		for f in a:
			if particle in f and t in f:
				return f
	
	tr_e = np.load(path+'train_electrons.npy')
	tr_p = np.load(path+'train_protons.npy')
	t_e = np.load(path+'test_electrons.npy')
	t_p = np.load(path+'test_protons.npy')
	
	return tr_e, tr_p, t_e, t_p
	

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
	
def plotXTR(test_e,test_p):
	'''
	Simple function to plot XTR/XTRL/Zeta
	'''
	
	XTRL_e = test_e[:,-2]
	XTRL_p = test_p[:,-2]
	
	
	binList = [i for i in range(0,160)]
	
	fig1 = plt.figure()
	plt.hist(XTRL_e,bins=binList,histtype='step',label='e')
	plt.hist(XTRL_p,bins=binList,histtype='step',label='p')
	plt.title(r'$\zeta$')
	plt.xlim((0,150))
	plt.xlabel(r'$\zeta$')
	plt.yscale('log')
	plt.legend(loc='upper right')
	plt.savefig('xtr/xtr')
	plt.close(fig1)
	
	del XTRL_e, XTRL_p		# In case of low memory


def histWithErrors(data,bins,label,color,normed=False):
	y,bin_edges,other = plt.hist(data,bins=bins,label=label,histtype='step',color=color,normed=normed)
	bin_centers = [ (bin_edges[i]+bin_edges[i+1])/2. for i in range(len(y))]
	if not normed:
		plt.errorbar(bin_centers,y,yerr=np.sqrt(y),fmt='none',ecolor=color)

####
# BEGIN: MAIN FUNCTIONS
####

def runTraining(n,batch_size,X_train,Y_train,X_val,Y_val,modelName,evtWeight):
	
	modelName2 = os.path.splitext(modelName)[0] + 'sigmoid' + os.path.splitext(modelName)[1]		# Name of DNN model with output activation
	historyName = os.path.splitext(modelName)[0] + 'history.pick'									# Name of train history file
	
	model = getModel(X_train)
	
	rdlronplt = ReduceLROnPlateau(monitor='loss',patience=3,min_lr=0.0001)	
	callbacks = [rdlronplt]
	
	#####
	history = model.fit(X_train,Y_train,batch_size=batch_size,epochs=n,verbose=0,callbacks=callbacks,validation_data=(X_val,Y_val))
	#####
	
	model.save(modelName2)
	
	histo = history.history
	pickle.dump(histo,open(historyName,'wb'),protocol=2)
	
	
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
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.legend(loc='upper center')
	plt.grid(True)
	plt.ylim((0.09,1e+5))
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
	
	e_redux = elecs_p[elecs_p < 250]
	p_redux = prots_p[prots_p > -250]

	w_e_redux = weights_e.reshape((weights_e.shape[0],1))[elecs_p < 250]
	w_p_redux = weights_p.reshape((weights_p.shape[0],1))[prots_p > -250]
	
	#~ binList = [i for i in range(p_redux.min(),e_redux.max(),3)]
	binList = np.histogram(np.hstack((e_redux,p_redux)),100)[1]

	plt.hist(e_redux,bins=binList,label='e',alpha=1.,histtype='step',color='green',weights=w_e_redux)
	plt.hist(p_redux,bins=binList,label='p',alpha=1.,histtype='step',color='red',weights=w_p_redux)
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.legend(loc='upper right')
	plt.grid(True)
	plt.ylim((0.09,1e+5))
	plt.xlim((-150,150))
	plt.yscale('log')
	plt.savefig('classScore_linear_allbins')
	plt.close(fig4)
	
	model3.save(modelName)
	

def buildPredArray(modelName,X_val,val):
	
	model_sigmoid = load_model(os.path.splitext(modelName)[0] + 'sigmoid' + os.path.splitext(modelName)[1])
	model_linear = load_model(modelName)
	
	nevents = val.shape[0]
	
	E_reco = val[:,44]
	E_truth = val[:,67]
	Y_val = val[:,-1]
	XTRL = val[:,-2]
	evtWeight = val[:,-3]
	evtID = val[:,-4]
	
	predictions_lin = model_linear.predict(X_val)
	predictions_sig = model_sigmoid.predict(X_val)
	
	# Begin stupid
	evtID = evtID.reshape((nevents,1))
	Y_val = Y_val.reshape((nevents,1))
	E_reco = E_reco.reshape((nevents,1))
	E_truth = E_truth.reshape((nevents,1))
	XTRL = XTRL.reshape((nevents,1))
	predictions_lin = predictions_lin.reshape((nevents,1))
	predictions_sig = predictions_sig.reshape((nevents,1))
	
	out_array = np.concatenate(( evtID,Y_val,E_reco,E_truth,XTRL,predictions_lin,predictions_sig ),axis=1)
	# End stupid
	
	np.save("prediction_array.npy",out_array)
	del out_array,evtID,Y_val,E_reco,E_truth,XTRL,predictions_lin,predictions_sig

def evaluation(e_min,e_max,val,X_val,Y_val,E_reco,E_truth,XTRL,modelname):	
	
	model = load_model(modelName)
	model_sig = load_model(os.path.splitext(modelName)[0] + 'sigmoid' + os.path.splitext(modelName)[1])
	
	predictions = model.predict(X_val)
	predictions_sigmoid = model_sig.predict(X_val)
	
	elecs_p, prots_p = getClassifierScore(Y_val,predictions)
	
	prots_p_redux = prots_p[prots_p > -250]
	elecs_p_redux = elecs_p[elecs_p < 250]
	
	binList = np.histogram(np.hstack((elecs_p_redux,prots_p_redux)),100)[1]
	fig1 = plt.figure()
	plt.hist(elecs_p_redux,bins=binList,label='e',alpha=1.,histtype='step',color='green')
	plt.hist(prots_p_redux,bins=binList,label='p',alpha=1.,histtype='step',color='red')
	plt.xlabel('Classifier score')
	plt.ylabel('Number of events')
	plt.legend(loc='upper left')
	plt.grid(True)
	plt.xlim((-100,100))
	plt.ylim((0.9,5e+4))
	plt.yscale('log')
	plt.title('DNN-linear \n'+str(int(e_min/1000))+'-'+str(int(e_max/1000))+' GeV')
	plt.savefig('predHisto/predHisto_'+str(int(e_min/1000))+'-'+str(int(e_max/1000)))
	plt.close(fig1)
	
	
	XTRL_e = XTRL[ Y_val.astype(bool) ]
	XTRL_p = XTRL[ ~Y_val.astype(bool) ]
	
	
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
	
	print('-----',str(int(e_min/1000)),' - ',str(int(e_max/1000)), ' GeV -----')
	
	for i in range(npoints):
		thr = np.min(predictions) + i * ((np.max(predictions) - np.min(predictions))/npoints)
		thr_sigmoid = i*(1./npoints)
		thr_xtrl = i * (np.max(XTRL_e)/npoints)
		
		tp,fp,tn,fn = getcountsFast(Y_val,predictions,thr)
		tp_xtr,fp_xtr,tn_xtr,fn_xtr = getcountsFast(Y_val,predictions_sigmoid,thr_sigmoid)
		tp_xtrl,fp_xtrl,tn_xtrl,fn_xtrl = getcountsXTRL(XTRL_e,XTRL_p,thr_xtrl)
		
		tp,fp,tn,fn = [ufloat(x,np.sqrt(x)) for x in [tp,fp,tn,fn ]]
		tp_xtr,fp_xtr,tn_xtr,fn_xtr = [ufloat(x,np.sqrt(x)) for x in [tp_xtr,fp_xtr,tn_xtr,fn_xtr]]
		tp_xtrl,fp_xtrl,tn_xtrl,fn_xtrl = [ufloat(x,np.sqrt(x)) for x in [tp_xtrl,fp_xtrl,tn_xtrl,fn_xtrl]]
		
		try:
			bk = fp / (tp+fp)
		except ZeroDivisionError as ex :
			bk = ufloat(1,0)
		try:
			bk_xtr = fp_xtr/(tp_xtr+fp_xtr)
		except ZeroDivisionError as ex :
			bk_xtr = ufloat(1,0)
		try:
			bk_xtrl = fp_xtrl/(tp_xtrl+fp_xtrl)
		except ZeroDivisionError as ex :
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
	
	return roc_auc_score(Y_val,predictions)

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

def beamTest(modelname,indexList,X_max,doFraction):
	
	from matplotlib.backends.backend_pdf import PdfPages
	
	model = load_model(modelname)
	
	SET_PATH = '/home/drozd/analysis/ntuples/MC-skim-RMS-07Feb18/'
	BT_E_PATH = SET_PATH + 'BT_data_e.npy'
	BT_P_PATH = SET_PATH + 'BT_data_p.npy'
	BTMC_E_PATH = SET_PATH + 'BT_MC_e.npy'
	BTMC_P_PATH = SET_PATH + 'BT_MC_p.npy'
	BT_E_100_PATH = SET_PATH + 'BT_data_e_100.npy'
	BTMC_E_100_PATH = SET_PATH + 'BT_MC_e_100.npy'
	
	pp_e = PdfPages('beamtest/vars_e.pdf')
	pp_e_100 = PdfPages('beamtest/vars_e_100.pdf')
	pp_p = PdfPages('beamtest/vars_p.pdf')
	
	
	arr_mc_e = np.load(BTMC_E_PATH)
	arr_mc_p = np.load(BTMC_P_PATH)
	arr_bt_e = np.load(BT_E_PATH)
	arr_bt_p = np.load(BT_P_PATH)[0:arr_mc_p.shape[0]]
	
	
	arr_bt_e_100 = np.load(BT_E_100_PATH)
	arr_mc_e_100 = np.load(BTMC_E_100_PATH)[:,0:arr_bt_e_100.shape[0]]
	
	if doFraction:			# Divide the first 14 layers by the total energy so we have fractions instead of absolute scale
		for i in range(14):
			arr_mc_e[:,i] = arr_mc_e[:,i]/arr_mc_e[:,44]
			arr_bt_e[:,i] = arr_bt_e[:,i]/arr_bt_e[:,44]
			arr_mc_p[:,i] = arr_mc_p[:,i]/arr_mc_p[:,44]
			arr_bt_p[:,i] = arr_bt_p[:,i]/arr_bt_p[:,44]
			arr_mc_e_100[:,i] = arr_mc_e_100[:,i]/arr_mc_e_100[:,44]
			arr_bt_e_100[:,i] = arr_bt_e_100[:,i]/arr_bt_e_100[:,44]
	
	labs = getLabels()
	for i in range(47):
		
		for arr_bt, arr_mc, pp, name in [ [arr_bt_e,arr_mc_e,pp_e,"Electron 250"],[arr_bt_p,arr_mc_p,pp_p,"Proton 400"],[arr_bt_e_100,arr_mc_e_100,pp_e_100,"Electron 100"]]:
		
			fig_bt = plt.figure()
			binList = np.histogram( np.hstack((arr_bt[:,i],arr_mc[:,i])),50)[1]
			plt.hist(arr_bt[:,i],bins=binList,normed=True,histtype='step',label='BT')
			plt.hist(arr_mc[:,i],bins=binList,normed=True,histtype='step',label='MC')
			plt.legend(loc='best')
			plt.yscale('log')
			plt.title(name+' GeV \n' + labs[i] )
			plt.savefig(pp, format='pdf')
			plt.close(fig_bt)
	pp_e.close()
	pp_p.close()
	pp_e_100.close()
	
	BT = np.concatenate(( arr_bt_e, arr_bt_p ))
	X_BT = BT[:,indexList] / X_max
	Y_BT = BT[:,-1]
	
	BTMC = np.concatenate(( arr_mc_e, arr_mc_p ))
	X_BTMC = BTMC[:,indexList] / X_max
	Y_BTMC = BTMC[:,-1]
	
	
	pred_BT = model.predict(X_BT)
	pred_BTMC = model.predict(X_BTMC)
	
	pred_e_BT, pred_p_BT = getClassifierScore(Y_BT,pred_BT)
	pred_e_BTMC, pred_p_BTMC = getClassifierScore(Y_BTMC,pred_BTMC)
	
	pred_e_100_BT = model.predict( arr_bt_e_100[:,indexList]/X_max )
	pred_e_100_BTMC = model.predict( arr_mc_e_100[:,indexList]/X_max )
	
	for s in ['linear','log']:
	
		fig2 = plt.figure()
		pred_p_BT_redux = pred_p_BT[pred_p_BT > -110]
		binList = [ -100 + i*(30 - (-100))/50 for i in range(51) ]
		#~ histWithErrors(pred_e_BT,binList,'Electron 250 GeV','green')
		#~ histWithErrors(pred_p_BT,binList,'Proton 400 GeV','red')
		plt.hist(pred_e_BT,bins=binList,label='Electron 250 GeV',alpha=1.,histtype='step',color='green')
		plt.hist(pred_p_BT_redux,bins=binList,label='Proton 400 GeV',alpha=1.,histtype='step',color='red',ls='dashed')
		plt.xlabel('Classifier score')
		plt.ylabel('Number of events')
		plt.title('Beamtest data')
		plt.legend(loc='upper center')
		plt.grid(True)
		plt.gca().set_axisbelow(True)
		plt.gca().grid(color='gray', linestyle='dashed')
		if s == 'log':
			plt.ylim((0.9,5e+5))
		plt.xlim((-100,30))
		plt.yscale(s)
		plt.savefig('beamtest/predHisto_BT_'+s)
		plt.close(fig2)
		
		fig3 = plt.figure()
		binList = [ -20 + i*(20 - (-20))/50 for i in range(51) ]
		#~ histWithErrors(pred_e_BTMC,binList,'Electron 250 GeV','green')
		#~ histWithErrors(pred_p_BTMC,binList,'Proton 400 GeV','red')
		plt.hist(pred_e_BTMC,bins=binList,label='Electron 250 GeV',alpha=1.,histtype='step',color='green')
		plt.hist(pred_p_BTMC,bins=binList,label='Proton 400 GeV',alpha=1.,histtype='step',color='red',ls='dashed')
		plt.xlabel('Classifier score')
		plt.ylabel('Number of events')
		plt.title('Beamtest Monte-Carlo')
		plt.legend(loc='upper center')
		plt.grid(True)
		plt.gca().set_axisbelow(True)
		plt.gca().grid(color='gray', linestyle='dashed')
		if s == 'log':
			plt.ylim((0.8,1e+5))
		plt.xlim((-20,20))
		plt.yscale(s)
		plt.savefig('beamtest/predHisto_BTMC_'+s)
		plt.close(fig3)
		
		fig4 = plt.figure()
		pred_BT_redux = pred_BT[pred_BT > -60]
		binList = [ -50 + i*(20 - (-50))/50 for i in range(51) ]
		histWithErrors(pred_BT_redux,binList,'BT data','C0')
		histWithErrors(pred_BTMC,binList,'BT MC','C1')
		#~ plt.hist(pred_BT_redux,bins=binList,label='BT data',alpha=1.,histtype='step',normed=False)
		#~ plt.hist(pred_BTMC,bins=binList,label='BT MC',alpha=1.,histtype='step',normed=False)
		plt.xlabel('Classifier score')
		plt.ylabel('Events')
		plt.title('Beamtest, electron 250GeV, proton 400GeV')
		plt.legend(loc='upper right')
		plt.grid(True)
		plt.gca().set_axisbelow(True)
		plt.gca().grid(color='gray', linestyle='dashed')
		plt.xlim((-50,50))
		#~ plt.ylim((0.9,1e+6))
		plt.yscale(s)
		plt.savefig('beamtest/predHisto_BTvsMC_'+s)
		plt.close(fig4)
		
		fig4b = plt.figure()
		binList = [i for i in range( int(min([pred_e_BT.min(),pred_e_BTMC.min()])), int(max([pred_e_BT.max(),pred_e_BTMC.max()])))]
		histWithErrors(pred_e_BT,binList,'BT data','C0')
		histWithErrors(pred_e_BTMC,binList,'BT MC','C1')
		#~ plt.hist(pred_e_BT,bins=binList,label='BT data',alpha=1.,histtype='step',normed=False)
		#~ plt.hist(pred_e_BTMC,bins=binList,label='BT MC',alpha=1.,histtype='step',normed=False)
		plt.xlabel('Classifier score')
		plt.ylabel('Events')
		plt.title('Beamtest, electron 250 GeV')
		plt.legend(loc='upper left')
		plt.grid(True)
		plt.gca().set_axisbelow(True)
		plt.gca().grid(color='gray', linestyle='dashed')
		#~ plt.ylim((0.9,1e+6))
		plt.yscale(s)
		plt.savefig('beamtest/predHisto_BTvsMC_e_'+s)
		plt.close(fig4b)
		
		fig4c = plt.figure()
		binList = [ -100 + i*(20 - (-100))/50 for i in range(51) ]
		histWithErrors(pred_p_BT,binList,'BT data','C0')
		histWithErrors(pred_p_BTMC,binList,'BT MC','C1')
		#~ plt.hist(pred_p_BT,bins=binList,label='p, BT data',alpha=1.,histtype='step',normed=False)
		#~ plt.hist(pred_p_BTMC,bins=binList,label='p, BT MC',alpha=1.,histtype='step',normed=False)
		plt.xlabel('Classifier score')
		plt.ylabel('Events')
		plt.title('Beamtest, proton 400 GeV')
		plt.legend(loc='upper left')
		plt.grid(True)
		plt.gca().set_axisbelow(True)
		plt.gca().grid(color='gray', linestyle='dashed')
		plt.xlim((-100,20))
		#~ plt.ylim((0.9,1e+6))
		plt.yscale(s)
		plt.savefig('beamtest/predHisto_BTvsMC_p_'+s)
		plt.close(fig4c)
		
		fig4d = plt.figure()
		#~ e_100_redux = pred_e_100_BT[np.absolute(pred_e_100_BT) < 25]
		#~ e_100_MC_red = pred_e_100_BTMC[np.absolute(pred_e_100_BTMC) < 25]
		#~ binList = [i for i in range( int(min([pred_e_100_BT.min(),pred_e_100_BTMC.min()])), int(max([pred_e_100_BT.max(),pred_e_100_BTMC.max()])))]
		binList = [i for i in range(-20,30,2)]
		histWithErrors(pred_e_100_BT,binList,'BT data','C0')
		histWithErrors(pred_e_100_BTMC,binList,'BT MC','C1')
		plt.xlabel('Classifier score')
		plt.ylabel('Events')
		plt.title('Beamtest, electron 100 GeV')
		plt.legend(loc='upper left')
		plt.grid(True)
		plt.gca().set_axisbelow(True)
		plt.gca().grid(color='gray', linestyle='dashed')
		#~ plt.ylim((0.9,1e+6))
		plt.yscale(s)
		plt.savefig('beamtest/predHisto_BTvsMC_e_100_'+s)
		plt.close(fig4d)
	############################################################################################################
	############################################################################################################
	############################################################################################################





if __name__ == '__main__' :
	
	parser = ArgumentParser()
	parser.add_argument("--nepochs","-n", dest='nepoch', default=100, type=int, help="Number of epochs for training")
	parser.add_argument("--batch_size","-bs", dest='batches', default=50, type=int, help="Batch size for training")
	parser.add_argument("--nohits", action='store_true', dest='nohits', default=False, help="Don't include BGO hits")
	parser.add_argument("--fraction", action='store_true', dest='fraction', default=False, help="Fraction of energy instead of absolute")
	parser.add_argument("--bins","-nb", dest="nbins", default=6, type=int, help="Number of energy bins")
	parser.add_argument("--emin","-e", dest="emin", default=40*1e+3, type=float, help="Minimum BGO energy (MeV)")
	parser.add_argument("--emax","-E", dest="emax", default=10*1e+6, type=float, help="Maximum BGO energy (MeV)")
	parser.add_argument("--setpath","-p", dest="path", default='/home/drozd/analysis/ntuples/MC-skim-RMS-07Feb18/', help="Path to numpy files")
	parser.add_argument("--autoencoder","-a", dest="autoencoder", default=False, help="Use a variational autoencoder pre-training")
	opts = parser.parse_args()
	
	for x in ['predHisto','ROC','pickles','beamtest','beamtest/vars','xtr','models']:
		if not os.path.isdir(x):
			os.mkdir(x)
	
	modelName = "models/model_"+str(opts.nepoch)+'.h5'
	
	if opts.nohits:		# Build a list of indices that exclude the BGO Hits from the numpy arrays
		indexList = [i for i in range(28)] + [i for i in range(42,45)] + [46]
		modelName = modelName.replace('.h5','_noHits.h5')
	else:
		indexList = [i for i in range(47)]
		
	if opts.fraction:
		modelName = modelName.replace('h5','_fraction.h5')
	
	# Load the four datasets in memory. Avoid slow down by constant read from hard drive
	print(_now(), ': Loading data')
	train_electrons, train_protons, test_electrons, test_protons = loadData(opts.path)
	
	train = np.concatenate((train_electrons,train_protons))
	np.random.shuffle(train)
	X_train = train[:,indexList]
	Y_train = train[:,-1]
	
	
	test = np.concatenate((test_electrons,test_protons))
	X_test = test[:,indexList]
	Y_test = test[:,-1]
	weight_test = test[:,-3]
	if opts.fraction:			# Divide the first 14 layers by the total energy so we have fractions instead of absolute scale
		for i in range(14):
			X_train[:,i] = X_train[:,i]/train[:,44]
			X_test[:,i] = X_test[:,i]/test[:,44]
	X_max = X_train.max(axis=0)			# Normalisation: Shift variables to the [0:1] range
	np.save('models/X_max.npy',X_max)
	X_train = X_train / X_max	
	X_test = X_test / X_max
	####
			
	plotXTR(test_electrons,test_protons)
	
	if not os.path.isfile(modelName):
		print(_now(), ': Starting training')
		runTraining(opts.nepoch,opts.batches,X_train,Y_train,X_test,Y_test,modelName,weight_test)
		
	try:
		buildPredArray(modelName,X_test,test)
	except Exception as ex:
		print("!! Error in buildPredArray:", ex)
	
	if not os.path.isfile('pickles/auc.pickle'):
		print(_now(), ': Starting energy evaluation')
		energies = np.geomspace(opts.emin,opts.emax,opts.nbins+1)
		auc = []
		for i in range(opts.nbins):
			val_bin = test[ test[:,44] > energies[i] ]
			val_bin = val_bin[ val_bin[:,44] <= energies[i+1] ]
			X_val_bin = val_bin[:,indexList]
			Y_val_bin = val_bin[:,-1]
			E_reco_bin = val_bin[:,44]
			E_truth_bin = val_bin[:,67]
			XTRL = val_bin[:,-2]
			
			if opts.fraction:
				for j in range(14): X_val_bin[:,j] = X_val_bin[:,j]/E_reco_bin
			X_val_bin = X_val_bin/X_max
			
			auc_score = evaluation(energies[i],energies[i+1],val_bin,X_val_bin,Y_val_bin,E_reco_bin,E_truth_bin,XTRL,modelName)
			auc.append( [ (energies[i]+energies[i+1])/2. , auc_score] )
			print('-----')
		rocEnergies(energies,opts.nbins)
		with open('pickles/auc.pickle','wb') as f:
			pickle.dump(auc,f,protocol=2)
	
	
	print(_now(), ': Starting beamTest analysis')
	beamTest(modelName,indexList,X_max,opts.fraction)
	print(_now(), ': Done with everything')
