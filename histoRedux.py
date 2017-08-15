'''

Compares the histograms of all variables for the machine learning, for electrons and protons
Substracts the two histograms, then computes the area of the result

'''

from __future__ import print_function, division, absolute_import


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def getLabels():
	
	lab = []
	
	ebgo = 'BGO_E_layer_'
	for i in range(14):
		lab.append(ebgo + str(i))
	erms = 'BGO_E_RMS_layer_'
	for i in range(14):
		lab.append(erms + str(i))
	lab.append('BGO_RMS_longitudinal')
	lab.append('BGO_RMS_radial')
	lab.append('BGO_E_total_corrected')
	lab.append('BGO_total_hits')
	
	for i in range(2):
		lab.append('PSD_E_layer_' + str(i))
	for i in range(2):
		lab.append('PSD_hits_layer_' + str(i))
	for k in ['1a','1b','2a','2b']:
		lab.append('PSD_E_RMS_layer_' + k)
		
	lab.append('STK_NClusters')
	lab.append('STK_NTracks')
	for i in range(8):
		lab.append('STK_E_' + str(i))
	for i in range(8):
		lab.append('STK_E_RMS_' + str(i))
		
	lab.append('timestamp')
	lab.append('label')
	
	return lab

def run():
	val_e = np.load('/home/drozd/analysis/newData/data_validate_elecs_under_1.npy') 
	val_p = np.load('/home/drozd/analysis/newData/data_validate_prots_under_1.npy')[0:val_e.shape[0],:]
	
	nrofvars = val_e.shape[1] - 2
	
	lab = getLabels()
	
	for i in range(nrofvars):
		val_e[:,i] = val_e[:,i] / val_e[:,i].max(axis=0)
		val_p[:,i] = val_p[:,i] / val_p[:,i].max(axis=0)
	
	binList = [x/100. for x in range(0,101)]
	
	if not os.path.isdir('images'): os.mkdir('images')
	
	l_area = []
	
	for n in range(nrofvars):
		
		fig1 = plt.figure()
		e_n, e_bins, e_patches = plt.hist(val_e[:,n],bins=binList,histtype='step',color='green',label='e')
		p_n, p_bins, p_patches = plt.hist(val_p[:,n],bins=binList,histtype='step',color='red',label='p')
		
		e_bins_c = [(e_bins[i]+e_bins[i+1])/2. for i in range(len(e_n))]
		p_bins_c = [(p_bins[i]+p_bins[i+1])/2. for i in range(len(p_n))]
		
		sub = [abs(e_n[i] - p_n[i]) for i in range(len(e_n))]
		
		
		plt.plot(e_bins_c,sub,color='blue',label='sub')
		plt.legend(loc='best')
		plt.yscale('log')
		plt.ylabel('Nr of events')
		plt.title(lab[n])
		
		plt.savefig('images/'+"%02d" % (n,))
		plt.close(fig1)
		
		area = np.trapz(sub,x=e_bins_c)
		
		l_area.append(area)
		
		if area > 3500:
			print(n)
		
	fig2 = plt.figure()
	plt.plot([n for n in range(nrofvars)],l_area,'o')
	plt.xlabel('Variable number')
	plt.ylabel('Area of histogram difference')
	plt.grid(True)
	plt.savefig('result')
	
	
	
	
	
	
if __name__ == '__main__':
	
	run()
