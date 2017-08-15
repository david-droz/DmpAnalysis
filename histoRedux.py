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



def run():
	val_e = np.load('/home/drozd/analysis/newData/data_validate_elecs_under_1.npy') 
	val_p = np.load('/home/drozd/analysis/newData/data_validate_prots_under_1.npy')[0:val_e.shape[0],:]
	
	nrofvars = val_e.shape[1] - 2
	
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
		
		
		plt.hist(sub,bins=binList,histtype='step',color='blue',label='sub')
		plt.legend(loc='best')
		plt.yscale('log')
		
		plt.savefig('images/'+"%02d" % (n,))
		
		area = np.trapz(sub,x=e_bins_c)
		
		l_area.append(area)
		
	fig2 = plt.figure()
	plt.plot([n for n in range(nrofvars)],l_area,'o')
	plt.grid(True)
	plt.savefig('result')
	
	
	
	
	
	
if __name__ == '__main__':
	
	run()
