'''

makeDataSets.py  v0.1

Merges together the output of selection.py and creates training/validation/testing sets. Takes a lot of memory!

Usage:
	> python makeDataSets.py --train_split 0.6 --validation_split 0.2 --validation_mixture 100 --test_mixture 100


'''

import sys
import numpy as np
from tree_tools import np2root
import random
import glob 
import struct
import ast
import argparse

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
	for f in [electronFiles,protonFiles]: f.sort()
	
	nrofe = getNrEvents(electronFiles)
	nrofp = getNrEvents(protonFiles)
	
	with np.load(electronFiles[0]) as arr:
		nrofvars = arr.shape[1]
	labels = getLabels()
	
	parser = argparse.ArgumentParser()
	parser.add_argument("-ts", "--train_split", help="Fraction of total electrons that go into training [0;1]",default=0.6,type=float)
	parser.add_argument("-vs", "--validation_split", help="Fraction of total electrons that go into validation [0;1-train_split]",default=0.2,type=float)
	parser.add_argument("-vm", "--validation_mixture", help="Ratio of protons to electrons in validation set ( > 1)",default=100,type=int)
	parser.add_argument("-tm", "--test_mixture", help="Ratio of protons to electrons in testing set ( > 1)", default=100,type=int)
		
	args = parser.parse_args()
	
	if args.train_split + args.validation_split >= 1.:
		raise Exception("Train split and validation split make up more than 100% of data")
	
	# Fractions and mixtures
	trainingFraction = args.train_split				# Fraction of electrons that go into training
	validationFraction = args.validation_split		# Fraction of electrons that go into validation
	validationMixture = args.validation_mixture		# Ratio protons/electrons for validation. Number greater than 1.
	testMixture = args.test_mixture				#  Ratio protons/electrons for testing
	
	selectedE_train, selectedP_train, selectedE_validate, selectedP_validate, selectedE_test, selectedP_test = getSetIndexes(nrofe,nrofp,trainingFraction,validationFraction,validationMixture,testMixture)
	
	# Merging and splitting - electrons
	arr_e = np.load(electronFiles[0])
	i = 0
	for f in electronFiles:
		if i==0:
			i=i+1
			continue
		arr_e = np.concatenate( (arr_e , np.load(f) ) )
	np.random.shuffle(arr_e)
	set_e_train = arr_e[ selectedE_train, :]
	set_e_validate = arr_e[ selectedE_validate, :]
	set_e_test = arr_e[ selectedE_test, :]
	del arr_e
	np.save('data_training_elecs.npy',set_e_train)
	np.save('data_validate_elecs.npy',set_e_validate)
	np.save('data_testing_elecs.npy',set_e_test)
	
	# Protons
	arr_p = np.load(protonFiles[0])
	i = 0
	for f in protonFiles:
		if i==0:
			i=i+1
			continue
		arr_p = np.concatenate( (arr_p , np.load(f) ) )
	np.random.shuffle(arr_p)
	set_p_train = arr_p[ selectedP_train, :]
	set_p_validate = arr_p[ selectedP_validate, :]
	set_p_test = arr_p[ selectedP_test, :]
	del arr_p
	np.save('data_training_prots.npy',set_p_train)
	np.save('data_validate_prots.npy',set_p_validate)
	np.save('data_testing_prots.npy',set_p_test)
	
	train_set = np.concatenate( (set_e_train, set_p_train ) )
	np.random.shuffle(train_set)
	np.save('dataset_training.npy',train_set)
	np2root(train_set,getLabels(),outname='dataset_training.root')
	del train_set
	
	validate_set = np.concatenate( (set_e_validate, set_p_validate ) )
	np.random.shuffle(validate_set)
	np.save('dataset_validate.npy',validate_set)
	np2root(validate_set,getLabels(),outname='dataset_validate.root')
	del validate_set
	
	test_set = np.concatenate( (set_e_test, set_p_test ) )
	np.random.shuffle(test_set)
	np.save('dataset_testing.npy',test_set)
	np2root(test_set,getLabels(),outname='dataset_testing.root')
	del test_set
	
	
