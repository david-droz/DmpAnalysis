'''

makeHistograms.py

Plot the (normalised) histograms of all variables in the machine learning datasets. 

	> python makeHistograms.py arrayofelectrons.npy arrayofprotons.npy


'''

import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

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
	lab.append('BGO_E_total_corrected')
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
		
	lab.append('timestamp')
	lab.append('label')
	
	return lab

def _normalise(arr):
	for i in range(arr.shape[1]):
		if np.all(arr[:,i] > 0) :
			arr[:,i] = (arr[:,i] - np.mean(arr[:,i]) + 1.) / np.std(arr[:,i])		# Mean = 1 if all values are strictly positive (from paper)
		else:
			arr[:,i] = (arr[:,i] - np.mean(arr[:,i])) / np.std(arr[:,i])	
	return arr

if __name__ == '__main__':
	
	arr_elecs = np.load(sys.argv[1])[:,0:-2]
	arr_prots = np.load(sys.argv[2])[:,0:-2]
	
	arr_elecs_n = arr_elecs / arr_elecs.max(axis=0)
	arr_prots_n = arr_prots / arr_prots.max(axis=0)
		
	lab = getLabels()
	
	pp_n = PdfPages('allVars_norm.pdf')
	pp = PdfPages('allVars.pdf')
	
	for i in range(arr_elecs.shape[1]):
		
		f1 = plt.figure()
		plt.hist(arr_elecs[:,i], 50, normed=1,histtype='step', label='e')
		plt.hist(arr_prots[:,i], 50, normed=1,histtype='step', label='p')
		plt.legend(loc='best')
		plt.yscale('log')
		plt.title(lab[i])
		plt.savefig(pp, format='pdf')
		plt.close(f1)
		
		f2 = plt.figure()
		plt.hist(arr_elecs_n[:,i], 50, normed=1,histtype='step', label='e')
		plt.hist(arr_prots_n[:,i], 50, normed=1,histtype='step', label='p')
		plt.legend(loc='best')
		plt.yscale('log')
		plt.title(lab[i])
		plt.savefig(pp_n, format='pdf')
		plt.close(f2)
		
	pp.close()
