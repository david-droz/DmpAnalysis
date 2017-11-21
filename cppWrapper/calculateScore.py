'''

Python function to compute the DNN score of a given DAMPE event

Input must be a Python list that contains the following variables, in that order:
	
	| Variable index	|	Variable name					| 	DmpSoftware function call
	------------------------------------------------------------------------------------
	|		0 - 13		|	Energy in BGO layer i			|	DmpChain->pEvtBgoRec()->GetELayer(i)
	|		14 - 27		|	RMS2 in individual BGO layers	|			---    		  ->GetRMS2()[i]
	|		28 - 41		|	Hits in individual BGO layers	|			---    		  ->GetLayerHits()[i]
	|		42			|	Longitudinal RMS				|			---    		  ->GetRMS_l()
	|		43			| 	Radial RMS						|			---    		  ->GetRMS_r()
	|		44			|	Total BGO energy				|			---    		  ->GetTotalEnergy()
	|		45			|	Total BGO hits					|			---    		  ->GetTotalHits() 
	|		46			|	XZ slope (angle calculation)	|			---    		  ->GetSlopeXZ()
	|		47			|	YZ slope (angle calculation)	|			---    		  ->GetSlopeYZ()

'''


from __future__ import division

import math
import numpy as np
import ctypes
from keras.models import load_model

model = load_model('trainedDNN.h5')
X_max = np.load('X_max.npy')

def calculateScore(event):
	
	array = np.array(event)
	
	XZ = array[-2]
	YZ = array[-1]
	tgZ = math.atan(np.sqrt( (XZ*XZ) + (YZ*YZ) ) )
	
	array = array[0:-1]
	array[-1] = tgZ*180./math.pi
	
	array = array.reshape((1,47))
	
	pred = model.predict(array / X_max)
	
	return pred


def testMethod(event):
	'''
	Test function. Expects a list of floats
	'''
	
	#~ print event
	
	array = np.array(event)
	
	#~ print array
	
	XZ = array[-2]
	YZ = array[-1]
	tgZ = math.atan(np.sqrt( (XZ*XZ) + (YZ*YZ) ) )
	
	array = array[0:-1]
	array[-1] = tgZ*180./math.pi
	
	print array
	
	return array.max()
