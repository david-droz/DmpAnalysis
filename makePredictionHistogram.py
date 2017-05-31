'''

Make the histogram of ML predictions

'''

import sys
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

	ID = sys.argv[1]
	
	predictions = np.load('./results/' + str(ID) + '/predictions.npy')
	truth = np.load('./results/Y_Val.npy')
	
	
	# Histogram of the predict_proba. Have to identify protons and electrons
