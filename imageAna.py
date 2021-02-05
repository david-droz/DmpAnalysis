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
from matplotlib.colors import LogNorm

from sklearn.metrics import roc_curve, roc_auc_score, precision_score, average_precision_score, precision_recall_curve, recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Keras deep neural networks
from keras.models import load_model
#~ from keras.models import Sequential, Model
#~ from keras.layers.core import Dense, Dropout
#~ from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
#~ from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler, ReduceLROnPlateau
#~ from keras import regularizers
#~ from keras.optimizers import Adam

from datetime import datetime

def main():
	
	if not os.path.isdir('eval'): os.mkdir('eval')
	
	PATH = '/home/drozd/analysis/runs/run_29Jan19_image/inFiles/'
	
	arr_p = np.load(PATH+'bigArr_32b_p.npy')
	arr_e = np.load(PATH+'bigArr_32b_e.npy')
	
	energy_p = np.load(PATH+'energy_p.npy')
	energy_e = np.load(PATH+'energy_e.npy')
	
	xtrl_p = np.load(PATH+'values_p.npy')[:,61]
	xtrl_e = np.load(PATH+'values_e.npy')[:,61]
	
	print("Data loaded",datetime.now())
	
	arr_p = arr_p.reshape(arr_p.shape[0],arr_p.shape[1],arr_p.shape[2],1) 
	arr_e = arr_e.reshape(arr_e.shape[0],arr_e.shape[1],arr_e.shape[2],1) 
	
	model = load_model('out/model.h5')
	
	print("Model loaded",datetime.now())
	
	if "2image" in os.getcwd():
		arr_p_XZ = arr_p[:,[2*i for i in range(7)],:,:]
		arr_p_YZ = arr_p[:,[2*i+1 for i in range(7)],:,:]
		arr_e_XZ = arr_e[:,[2*i for i in range(7)],:,:]
		arr_e_YZ = arr_e[:,[2*i+1 for i in range(7)],:,:]
		
		pred_p = model.predict( [arr_p_XZ,arr_p_YZ] )
		pred_e = model.predict( [arr_e_XZ,arr_e_YZ] )
	else:
		pred_p = model.predict(arr_p)
		pred_e = model.predict(arr_e)
	
	print("Predictions done",datetime.now())
	
	np.save('pred_e.npy',pred_e)
	np.save('pred_p.npy',pred_p)
	
	#####
	# LOOP ON ENERGY BINS
	#####
	
	Nbins = 8
	energyBins = np.logspace( np.log10(500*1000), np.log10(5000*1000), Nbins+1 )
	
	for i in range(Nbins):
		strbin = str(int(energyBins[i]/1000.))+'-'+str(int(energyBins[i+1]/1000.))
		print(strbin,datetime.now())
		try:
			bin_pred_e = pred_e[ np.logical_and(energy_e >= energyBins[i],energy_e < energyBins[i+1]) ]
			bin_pred_p = pred_p[ np.logical_and(energy_p >= energyBins[i],energy_p < energyBins[i+1]) ]
			bin_xtrl_e = xtrl_e[ np.logical_and(energy_e >= energyBins[i],energy_e < energyBins[i+1]) ]
			bin_xtrl_p = xtrl_p[ np.logical_and(energy_p >= energyBins[i],energy_p < energyBins[i+1]) ]
		except IndexError:
			print(pred_e.shape,pred_p.shape,energy_e.shape,energy_p.shape,xtrl_e.shape,xtrl_p.shape)
			raise
		
		fig1 = plt.figure()
		binList = np.linspace(-15,15,100)
		_ = plt.hist(bin_pred_e,binList,label='e',color='green',histtype='stepfilled',alpha=0.7)
		_ = plt.hist(bin_pred_p,binList,label='p',color='red',histtype='stepfilled',alpha=0.7)
		plt.ylim(ymin=0.9)
		plt.xlabel('Classifier score')
		plt.ylabel('Number of events')
		plt.title( strbin.replace('-','GeV - ') + ' GeV' )
		plt.legend(loc='upper center')
		plt.savefig('eval/classScore_'+strbin)
		plt.grid(True)
		plt.yscale('log')
		plt.savefig('eval/classScore_log_'+strbin)
		plt.close(fig1)
		
		
		bin2D = [np.linspace(-10,30,100),np.linspace(0,40,100)]
		fig1b = plt.figure()
		plt.hist2d( bin_pred_e.flatten(),bin_xtrl_e.flatten(), bin2D, cmap=plt.cm.jet,norm=LogNorm(),cmin=1)
		plt.xlabel('CNN')
		plt.ylabel(r'$\zeta$')
		plt.title( 'MC electrons \n'+strbin.replace('-','GeV - ') + ' GeV' )
		plt.savefig('eval/hist2d_e_{}.png'.format(strbin))
		plt.close(fig1b)
		
		bin2D = [np.linspace(-30,10,100),np.linspace(0,60,100)]
		fig1b = plt.figure()
		plt.hist2d( bin_pred_p.flatten(),bin_xtrl_p.flatten(), bin2D, cmap=plt.cm.jet,norm=LogNorm(),cmin=1)
		plt.xlabel('CNN')
		plt.ylabel(r'$\zeta$')
		plt.title( 'MC protons \n'+strbin.replace('-','GeV - ') + ' GeV' )
		plt.savefig('eval/hist2d_p_{}.png'.format(strbin))
		plt.close(fig1b)
		
		###
		# ROC
		###
		
		l_bkg = []
		l_eff = []
		npoints = 1000
		eff_90 = 1
		bkg_90 = 0
		
		for i in range(npoints):
			thr = (-100) + i * (((+300) - (-100))/npoints)
			
			tp = bin_pred_e[ bin_pred_e >= thr].shape[0]
			fn = bin_pred_e[ bin_pred_e < thr].shape[0]
			fp = bin_pred_p[ bin_pred_p >= thr].shape[0]
			tn = bin_pred_p[ bin_pred_p < thr].shape[0]
			
			tp,fp,tn,fn = [ufloat(x,np.sqrt(x)) for x in [tp,fp,tn,fn ]]
			
			try:
				bk = fp / (tn+fp)
			except ZeroDivisionError :
				bk = ufloat(1,0)
			eff = tp / (tp + fn)
			
			if eff > 0.95 and eff < eff_90:
				eff_90 = eff
				bkg_90 = bk
				
			l_bkg.append( (bk.n , bk.s) )
			l_eff.append( (eff.n,eff.s) )
			
		with open('eval/results_'+strbin+'.pickle','wb') as f:
			pickle.dump( [l_bkg,l_eff,bkg_90,eff_90],f,protocol=2)
			
		fig2 = plt.figure()
		plt.plot([x[0] for x in l_eff],[x[0] for x in l_bkg],label='CNN')
		plt.xlabel('Electron efficiency')
		plt.ylabel('Proton efficiency')
		plt.title(strbin.replace('-',' - ')+' GeV')
		plt.xlim((0.85,1.01))
		plt.yscale('log')
		#~ plt.legend(loc='best')
		plt.savefig('eval/roc_'+strbin)
		plt.close(fig2)





if __name__ == '__main__':
	
	main()
