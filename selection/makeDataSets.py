'''

Run after selection.py

'''

import sys
import numpy as np
from tree_tools import np2root
import random
import glob 
import struct
import ast







# Pick training/validation/testing set:
# Make a list of all available indexes for both electrons and protons
# Randomly choose indexes:
#		for i in xrange(training_size):
#			training_indexes.append(  random_elements_from_list_of_indexes  )
# Can use method "pop" to be sure to not repeat indexes
#
# This allows to select a number N of electrons and protons for all sets
# For the validation and testing sets, can use oversampling (bootstrapping ?) for protons


# Also don't forget to implement Stephan's method to turn that into a ROOT file


#~ def getNrEvents(filelist):
	#~ a = 0
	#~ for f in filelist:
		#~ arr = np.load(f)
		#~ a = a + arr.shape[0]
		#~ del arr
	#~ return a
	
def getNrEvents(filelist):
	'''
	http://stackoverflow.com/questions/43917512/python-merge-many-big-numpy-arrays-with-unknown-shape-that-would-not-fit-in-m
	'''
	npy_magic = b"\x93NUMPY"
	npy_v1_header = struct.Struct(
	        "<"   # little-endian encoding
	        "6s"  # 6 byte magic string
	        "B"   # 1 byte major number
	        "B"   # 1 byte minor number
	        "H"   # 2 byte header length
	        # ... header string follows
	)
	npy_v2_header = struct.Struct(
	        "<"   # little-endian encoding
	        "6s"  # 6 byte magic string
	        "B"   # 1 byte major number
	        "B"   # 1 byte minor number
	        "L"   # 4 byte header length
	        # ... header string follows
	)
	def read_npy_file_header(filename):
	    with open(filename, 'rb') as fp:
	        buf = fp.read(npy_v1_header.size)
	        magic, major, minor, hdr_size = npy_v1_header.unpack(buf)
	        if magic != npy_magic:
	            raise IOError("Not an npy file")
	        if major not in (0,1):
	            raise IOError("Unknown npy file version")
	        if major == 2:
	            fp.seek(0)
	            buf = fp.read(npy_v2_header.size)
	            magic, major, minor, hdr_size = npy_v2_header.unpack(buf)
	        return ast.literal_eval(fp.read(hdr_size).decode('ascii'))
	a = 0
	for f in filelist:
		a = a + read_npy_file_header(f)
	return a


def getSetIndexes(nrofe,nrofp,trainingFraction,validationFraction,validationMixture,testMixture):
	'''
	Returns list of event indexes for training, validating, testing and for elctrons, protons
	
	i.e. if the list returned is [4,9,15], it means I use events number 4,9, and 15.
	'''
	# Training
	available_E = range(nrofe)
	available_P = range(nrofp)
	selectedE_train = []
	selectedP_train = []
	for i in xrange(trainingfraction * nrofe):	
		j = random.randint(0,len(available_E)-1)
		k = random.randint(0,len(available_P)-1)
		selectedE_train.append( available_E.pop(j) )
		selectedP_train.append( available_P.pop(k) )
	# Validation
	# Has to be unbalanced
	selectedE_validate = []
	selectedP_validate = []
	for i in xrange(validationFraction * nrofe):
		j = random.randint(0,len(available_E)-1)
		selectedE_validate.append( available_E.pop(j) )
	if (validationMixture * len(selectedE_validate) ) > ( 0.7*len(available_P)) :		# Not enough protons, sampling with replacement
		# Oversampling with replacement
		for i in xrange(validationMixture * len(selectedE_validate)):
			j = random.randint(0,len(available_P)-1)
			selectedP_validate.append( available_P[j] )
	else:
		for i in xrange(validationMixture * len(selectedE_validate)):
			j = random.randint(0,len(available_P)-1)
			selectedP_validate.append( available_P.pop(j) )		
	# Testing - Also has to be unbalanced
	selectedE_test = [  x for x in available_E  ]
	selectedP_test = []
	if (testMixture * len(selectedE_test)) > len(available_P) :   # Not enough protons
		for i in xrange(testMixture * len(selectedE_test)):
			j = random.randint(0,len(available_P)-1)
			selectedP_test.append( available_P[j] )
	else:			
		for i in xrange(testMixture * len(selectedE_test)):
			j = random.randint(0,len(available_P)-1)
			selectedP_test.append( available_P.pop(j) )

	return selectedE_train, selectedP_train, selectedE_validate, selectedP_validate, selectedE_test, selectedP_test


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
	lab.append('timestamp')
	lab.append('label')
	
	return lab

if __name__ == '__main__':
	
	electronFiles = glob.glob('tmp/elec*.npy')
	protonFiles = glob.glob('tmp/prot*.npy')
	
	nrofe = getNrEvents(electronFiles)
	nrofp = getNrEvents(protonFiles)
	
	with np.load(electronFiles[0]) as arr:
		nrofvars = arr.shape[1]
	
	labels = getLabels()
	
	
	# Training
	trainingFraction = XXXXXXXXXXXXXXXXXXXXXXXXX	# Fraction of electrons that go into training
	validationFraction = XXXXXXXXXXXXXXXXX			# Fraction of electrons that go into validation
	validationMixture = XXXXXXXXXXXXXXXX			# Ratio protons/electrons for validation. Number greater than 1.
	testMixture = XXXXXXXXXXXXXXXXXXX				#  Ratio protons/electrons for testing
	
	selectedE_train, selectedP_train, selectedE_validate, selectedP_validate, selectedE_test, selectedP_test = getSetIndexes(nrofe,nrofp,trainingFraction,validationFraction,validationMixture,testMixture)
	
	#~ mmp_e = np.memmap('dataset_elec.npy',dtype='float64',mode='w+',shape=(nrofe,nrofvars))
	#~ mmp_p = np.memmap('dataset_prot.npy',dtype='float64',mode='w+',shape=(nrofp,nrofvars))
	#~ 
	#~ for i,f in enumerate(electronFiles):
		#~ mmp_e[i,:] = np.load(f)
	#~ for i,f in enumerate(protonFiles):
		#~ mmp_p[i,:] = np.load(f)
	
	## Forget about memmap, I cannot get it to work.
	
	
	# Now: load and merge all .npy files
		# To do that: either use the efficient numpy reading as suggested on StackOverflow, and then memmap
		# Or convert to Pandas DataFrame
		# Or use the large memory available on some ATLAS nodes
	
	# Then: split into the three datasets. Do it like:  small_dataset = big_dataset[ listofindices ,:]
	
	# Then: Save as both numpy files and root file
