'''

makeHistograms.py

Plot the (normalised) histograms of all variables in the machine learning datasets. 

	> python makeHistograms.py arrayofelectrons.npy arrayofprotons.npy


'''

import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.backends.backend_pdf import PdfPages


def _normalise(arr):
	for i in xrange(arr.shape[1]):
		if np.all(arr[:,i] > 0) :
			arr[:,i] = (arr[:,i] - np.mean(arr[:,i]) + 1.) / np.std(arr[:,i])		# Mean = 1 if all values are strictly positive (from paper)
		else:
			arr[:,i] = (arr[:,i] - np.mean(arr[:,i])) / np.std(arr[:,i])	
	return arr

if __name__ == '__main__':
	
	arr_elecs = _normalise(np.load(sys.argv[1])[:,0:-2])
	arr_prots = _normalise(np.load(sys.argv[2])[:,0:-2])
	
	pp = PdfPages('multipage.pdf')
	
	for i in xrange(arr_elecs.shape[1]):
		
		plt.hist(arr_elecs[:,i], 50, normed=1,histtype='step', label='e')
		plt.hist(arr_prots[:,i], 50, normed=1,histtype='step', label='p')
		plt.legend(loc='best')
		#~ plt.xscale("log")
		plt.yscale('log')
		plt.title('var '+str(i))
		plt.savefig(pp, format='pdf')
		
	
	pp.close()
