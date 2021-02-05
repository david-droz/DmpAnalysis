'''

Compare variables using three methods:

1. Sum of squared bin difference
2. Kolmogorov-Smirnov test
3. Area of histogram difference


Visual selection:
Electrons:  DNN - 0.1 * XTRL + 1 >= 0
Protons: others


'''

from __future__ import print_function, division, absolute_import

import numpy as np
import time
import pickle
import sys
import os
import psutil


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import load_model
from scipy.stats import ks_2samp

from matplotlib.backends.backend_pdf import PdfPages

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
	lab.append('BGO_E_total')
	lab.append('BGO_total_hits')
	lab.append('BGO_theta_angle')
	
	for i in range(2):
		lab.append('PSD_E_layer_' + str(i))
	for i in range(2):
		lab.append('PSD_hits_layer_' + str(i))
		
	lab.append('STK_NClusters')
	lab.append('STK_NTracks')
	for i in range(4):
		lab.append('STK_E_' + str(i))
	for i in range(4):
		lab.append('STK_E_RMS_' + str(i))
	for i in range(4):
		lab.append('NUD_channel_'+str(i))
		
	return lab	

def KSvars(MC,FM):
	'''
	Kolmogorov Smirnov test
	If p-value is low, then MC and FM likely come from different distributions
	'''
	KSscores = []
	pvalues = []
	for i in range(len(getLabels())) :
		a,b = ks_2samp( MC[:,i], FM[:,i] )
		KSscores.append(a)
		pvalues.append(b)
	return KSscores,pvalues
	
def histDistance(MC,FM,plotName):
	'''
	Bin both MC and FM, normalise such that max=1, then make the bin-by-bin squared difference and sum it. 
	'''
	
	lab = getLabels()
	dist = []
	pp = PdfPages(plotName)
	
	for i,l in enumerate(lab):
		hist_FM, bins_FM = np.histogram(FM[:,i],'fd')
		hist_MC, bins_MC = np.histogram(MC[:,i],bins=bins_FM)
		
		hist_FM = hist_FM / hist_FM.max()
		hist_MC = hist_MC / hist_MC.max()
		
		#~ fig1 = plt.figure()
		#~ width = np.diff(bins_FM)
		#~ center = (bins_FM[:-1] + bins_FM[1:]) / 2.
		#~ plt.bar(center,hist_FM,align='center',width=width,color="none",edgecolor='C0',label='Flight')
		#~ plt.bar(center,hist_MC,align='center',width=width,color="none",edgecolor='C1',label='MC')
		#~ plt.set_xticks(bins_FM)
		#~ plt.title(l)
		#~ plt.legend(loc='best')
		#~ plt.yscale('log')
		#~ plt.savefig(pp,format='pdf')
		#~ plt.close(fig1)
		fig1 = plt.figure()
		wfm = [1./hist_FM.max() for i in range(len(FM[:,i]))]
		wmc = [1./hist_MC.max() for i in range(len(MC[:,i]))]
		plt.hist(FM[:,i],bins=bins_FM,histtype='step',label='Flight',weights=wfm)
		plt.hist(MC[:,i],bins=bins_FM,histtype='step',label='MC',weights=wmc)
		plt.title(l)
		plt.legend(loc='best')
		plt.yscale('log')
		plt.savefig(pp,format='pdf')
		plt.close(fig1)
		
		
		diff = np.power(hist_FM - hist_MC,2)
		diff = np.sum(diff) / bins_FM.shape[0]
		
		dist.append(diff)
	pp.close()	
	return dist
		
def histArea(MC,FM):
	'''
	Bin both MC and FM, subtract the two histograms, get the area (bin sum?) of resulting histogram
	'''
	def _makeIntegral(MCvar,FMvar):
		hist_FM, bins_FM = np.histogram(FMvar,'fd')
		hist_MC, bins_MC = np.histogram(MCvar,bins=bins_FM)
		
		hist_FM = hist_FM / hist_FM.max()
		hist_MC = hist_MC / hist_MC.max()
		
		diff = np.abs( hist_FM - hist_MC )
		integral = np.sum(  ( bins_FM[1:] - bins_FM[:-1] )*diff )
		return integral
	
	
	lab = getLabels()
	dist = []
	dist_norm = []
	#~ pp = PdfPages(plotName)
	
	for i,l in enumerate(lab):
		
		dist.append( _makeIntegral( MC[:,i], FM[:,i] ) )
		dist_norm.append( _makeIntegral( MC[:,i]/MC[:,i].max() , FM[:,i]/FM[:,i].max() ) )
		
	return dist, dist_norm



	

def main():
	
	indices = [i for i in range(28)] + [42,43,44,46,47,48] + [i for i in range(51,60) ]
	lab = getLabels()
	
	model = load_model('/home/drozd/analysis/runs/run_17Jul18_otherVars/PSD_STK_noHits/out/model_linear.h5')
	X_max = np.load('/home/drozd/analysis/runs/run_17Jul18_otherVars/PSD_STK_noHits/out/X_max.npy')
	
	PATH = '/home/drozd/analysis/runs/run_17Jul18_otherVars/'
	MC = np.concatenate(( np.load(PATH+'allElectrons_test.npy'), np.load(PATH+'allProtons_test.npy') ))
	PATH = '/home/drozd/analysis/runs/run_25Jun18_multiSel/flight/'	
	flight = np.concatenate(( np.load(PATH+'allFlight_2016.npy') , np.load(PATH+'allFlight_2017.npy')  ))
	
	# 400 - 800 GeV
	MC = MC[ np.logical_and( MC[:,44] >= (400*1000) , MC[:,44] < (600*1000) ) ]
	flight = flight[ np.logical_and( flight[:,44] >= (400*1000) , flight[:,44] < (600*1000) ) ]
	
	pred_flight = model.predict( flight[:,indices]/X_max )
	
	MC_e = MC[ MC[:,-1].astype(bool) ]
	MC_p = MC[ ~MC[:,-1].astype(bool) ]
	del MC
	
	XTRLf = flight[:,-2]
	pred_flight = pred_flight.reshape(XTRLf.shape)
	
	cond = pred_flight - (0.1* XTRLf) + 1 
	
	flight_e = flight[ cond > 0 ]
	flight_p = flight[ cond < 0 ]
	pred_e = pred_flight[ cond > 0 ]
	pred_p = pred_flight[ cond < 0 ]
	
	blarg, bins = np.histogram(pred_flight[ np.absolute(pred_flight) < 50],'fd')
	del flight, cond, XTRLf, pred_flight, blarg
	
	fig0 = plt.figure()
	plt.hist(pred_e,bins=bins,histtype='step',color='green',label='e?')
	plt.hist(pred_p,bins=bins,histtype='step',color='red',label='p?')
	plt.yscale('log')
	plt.legend(loc='best')
	plt.savefig('test_predHisto')
	plt.close(fig0)
	del pred_e,pred_p
	
	
	KS_e, pvalue_e = KSvars(MC_e,flight_e)
	KS_p, pvalue_p = KSvars(MC_p,flight_p)
	
	hDist_e = histDistance(MC_e,flight_e,'vars_hDist_e.pdf')
	hDist_p = histDistance(MC_p,flight_p,'vars_hDist_p.pdf')
	
	areaDist_e, areaDistNorm_e = histArea(MC_e,flight_e)
	areaDist_p, areaDistNorm_p = histArea(MC_p,flight_p)
	
	xVals = [i for i in range(len(getLabels() )) ]
	
	fig1 = plt.figure()
	plt.bar([x - 0.2 for x in xVals],KS_e,label='e',alpha=0.6)
	plt.bar([x + 0.2 for x in xVals],KS_p,label='p',alpha=0.6)
	plt.xlabel('Variable index')
	plt.ylabel('KS score')
	plt.title('Kolmogorov-Smirnov test')
	plt.legend(loc='best')
	plt.savefig('KS')
	plt.close(fig1)
	
	fig2 = plt.figure()
	plt.bar(xVals,pvalue_e,label='e')
	plt.bar(xVals,pvalue_p,label='p')
	plt.xlabel('Variable index')
	plt.ylabel('KS p-value')
	plt.yscale('log')
	plt.title('Kolmogorov-Smirnov test (p-value)')
	plt.legend(loc='best')
	plt.savefig('KS_pvalue')
	plt.close(fig2)
	
	fig3 = plt.figure()
	plt.bar(xVals,hDist_e,label='e')
	plt.bar(xVals,hDist_p,label='p')
	plt.xlabel('Variable index')
	plt.ylabel('histogram distance')
	plt.legend(loc='best')
	plt.savefig('histoDist')
	plt.close(fig3)
	
	fig4 = plt.figure()
	plt.bar(xVals,areaDist_e,label='e')
	plt.bar(xVals,areaDist_p,label='p')
	plt.xlabel('Variable index')
	plt.ylabel('histogram difference area')
	plt.legend(loc='best')
	plt.savefig('histoArea')
	plt.close(fig4)
	
	fig5 = plt.figure()
	plt.bar(xVals,areaDistNorm_e,label='e')
	plt.bar(xVals,areaDistNorm_p,label='p')
	plt.xlabel('Variable index')
	plt.ylabel('histogram difference area (normalised)')
	plt.legend(loc='best')
	plt.savefig('histoAreaNorm')
	plt.close(fig5)
	



def studyChange():
	'''
	Change one variable at a time, look at how it affects the DNN output
	Do it multiple times per variable to have a mean/std 
	'''
	
	indices = [i for i in range(28)] + [42,43,44,46,47,48] + [i for i in range(51,60) ]
	lab = getLabels()
	
	model = load_model('/home/drozd/analysis/runs/run_17Jul18_otherVars/PSD_STK_noHits/out/model_linear.h5')
	X_max = np.load('/home/drozd/analysis/runs/run_17Jul18_otherVars/PSD_STK_noHits/out/X_max.npy')
	
	PATH = '/home/drozd/analysis/runs/run_17Jul18_otherVars/'
	MC = np.concatenate(( np.load(PATH+'allElectrons_test.npy'), np.load(PATH+'allProtons_test.npy') ))
	np.random.shuffle(MC)
	
	
	if not os.path.isfile('resultArray.npy'):
		resultArray = np.zeros(( 500, len(indices) ))
		
		for i in range(500) :		# Select 500 random events
			
			myEvent = MC[i,indices]
			myEvent_n = myEvent / X_max
			basePred = model.predict( myEvent_n.reshape( (1, myEvent.shape[0] ) ) )[0][0]
			
			for j in range(len(indices)) :
				otherEvent = np.copy(myEvent)
				otherEvent[j] *= 2			# Multiply variable j by 2
				otherEvent /= X_max
				newPred = model.predict( otherEvent.reshape( (1,myEvent.shape[0]) ) )[0][0] 
				
				resultArray[i,j] = (newPred - basePred)/basePred
				del otherEvent
				
		np.save('resultArray.npy',resultArray)		
	else:
		resultArray = np.load('resultArray.npy')	
			
	means = np.mean(resultArray,axis=0)
	std = np.std(resultArray,axis=0)
	
	fig1 = plt.figure()
	plt.errorbar( indices, means, yerr=std, fmt='.' )
	plt.xlabel('Variable index (name)')
	plt.ylabel('Relative change')
	plt.title('Change in DNN score when a single variable is doubled \n mean +/- std over 500 events')
	plt.savefig('varInfluence')
	
	fig2 = plt.figure()
	for i in range(30):
		plt.plot(indices,resultArray[i],'.',color='C0')
	plt.xlabel('Variable index (name)')
	plt.ylabel('Relative change')
	plt.title('Change in DNN score when a single variable is doubled \n 30 events')
	plt.savefig('varInfluence_allPoints')
		
	
	
	
if __name__ == '__main__' :
	
	#~ main()
	
	studyChange()	
