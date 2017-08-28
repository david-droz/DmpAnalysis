'''

makeDataSets.py  v0.4

Merges together the output of selection.py and creates training/validation/testing sets. Takes a lot of memory!

Usage:
	> python makeDataSets.py --train_split 0.6 --validation_split 0.2 --validation_mixture 100 --test_mixture 100 (--oversampling)


'''

import sys
import numpy as np
from tree_tools import np2root
import random
import glob 
import struct
import ast
import argparse
import os
import time
import cPickle as pickle


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
		#print read_npy_file_header(f)
		a = a + read_npy_file_header(f)['shape'][0]
	return a


def getSetIndexes(nrofe,nrofp,trainingFraction,validationFraction,validationMixture,testMixture,oversampling):
	'''
	Returns list of event indexes for training, validating, testing and for elctrons, protons
	
	i.e. if the list returned is [4,9,15], it means I use events number 4,9, and 15.
	'''
	
	# https://stackoverflow.com/questions/10048069/what-is-the-most-pythonic-way-to-pop-a-random-element-from-a-list
	
	pickFile = 'indices_'+str(nrofe)+str(nrofp)+str(trainingFraction)+str(validationFraction)+str(validationMixture)+str(testMixture)+'.pick'
	if oversampling:
		pickFile = pickFile.replace('.pick','over.pick')
	else:
		pickFile = pickFile.replace('.pick','under.pick')
	
	
	if os.path.isfile(pickFile):
		with open(pickFile,'rb') as f:
			a = pickle.load(f)
		return a[0], a[1], a[2], a[3], a[4], a[5]
	
	# ---- Training ----
	
	available_E = range(nrofe)
	available_P = range(nrofp)
	selectedE_train = []
	selectedP_train = []
	
	random.shuffle(available_E)							# Pick randomly
	random.shuffle(available_P)
	
	for i in xrange(int(1.2e+6)):								# 1M events
		selectedE_train.append( available_E.pop() )			# Use "pop" : don't want to reuse events
		selectedP_train.append( available_P.pop() )			# Same number of events for training

	# ---- Validation ----
	# Has to be unbalanced
	
	selectedE_validate = []
	selectedP_validate = []		
	
	if oversampling:
		for i in xrange(0.75e+6):								# 750k electrons 
			selectedE_validate.append( available_E.pop() )
		
		if (validationMixture * len(selectedE_validate) ) > ( 0.7*len(available_P)) :		# Not enough protons, sampling with replacement
			# Oversampling with replacement
			for i in xrange(validationMixture * len(selectedE_validate)):
				j = random.randint(0,len(available_P)-1)
				selectedP_validate.append( available_P[j] )
		else:
			for i in xrange(validationMixture * len(selectedE_validate)):
				selectedP_validate.append( available_P.pop() )	
	else:
		#n_elecs = int((len(available_P)/2.)/validationMixture)
		n_elecs = int(0.75e+6)
		for i in xrange(n_elecs):
			selectedE_validate.append( available_E.pop() )
		for i in xrange(len(available_P)/2):
			selectedP_validate.append( available_P.pop() )
				
	# ---- Testing ----
	
	selectedE_test = []
	selectedP_test = []
	
	if oversampling:
		for i in xrange(60000):											# 60k electrons
			selectedE_test.append( available_E.pop() )
			
		if (testMixture * len(selectedE_test)) > len(available_P) :		# Not enough protons
			for i in xrange(testMixture * len(selectedE_test)):
				j = random.randint(0,len(available_P)-1)
				selectedP_test.append( available_P[j] )
		else:			
			for i in xrange(testMixture * len(selectedE_test)):
				selectedP_test.append( available_P.pop() )
	else:
		#n_elecs = (len(available_P))/testMixture
		n_elecs = int(0.75e+6)
		for i in xrange(n_elecs):
			selectedE_test.append( available_E.pop() )
		selectedP_test = available_P
	
	with open(pickFile,'wb') as f:
		pickle.dump([selectedE_train, selectedP_train, selectedE_validate, selectedP_validate, selectedE_test, selectedP_test],f)
	
	return selectedE_train, selectedP_train, selectedE_validate, selectedP_validate, selectedE_test, selectedP_test


def getLabels():
	
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

if __name__ == '__main__':
	
	print "Starting up..."
	
	t0 = time.time()
	
	electronFiles = glob.glob('tmp/*/elec*.npy')
	protonFiles = glob.glob('tmp/*/prot*.npy')
	for f in [electronFiles,protonFiles]: f.sort()
	
	evtFile = 'eventNumbers.pickle'
	if os.path.isfile(evtFile):
		with open(evtFile,'rb') as f:
			nrofe,nrofp = pickle.load(f)
	else:
		nrofe = getNrEvents(electronFiles)
		nrofp = getNrEvents(protonFiles)
		with open(evtFile,'wb') as f:
			pickle.dump([nrofe,nrofp],f)
	
	arr = np.load(electronFiles[0])
	nrofvars = arr.shape[1]
	del arr
	labels = getLabels()
	
	print "Got labels (", str(time.strftime('%H:%M:%S', time.gmtime( time.time() - t0 ))), ')'
	
	parser = argparse.ArgumentParser()
	parser.add_argument("-ts", "--train_split", help="Fraction of total electrons that go into training [0;1]",default=0.6,type=float)
	parser.add_argument("-vs", "--validation_split", help="Fraction of total electrons that go into validation [0;1-train_split]",default=0.2,type=float)
	parser.add_argument("-vm", "--validation_mixture", help="Ratio of protons to electrons in validation set ( > 1)",default=100,type=int)
	parser.add_argument("-tm", "--test_mixture", help="Ratio of protons to electrons in testing set ( > 1)", default=100,type=int)
	parser.add_argument("--onlyprotons",help="Run only for protons",action='store_true',default=False)
	parser.add_argument("--onlyelectrons",help="Run only for electrons",action='store_true',default=False)
	parser.add_argument("--onlymerge",help="Only merge e/p runs",action='store_true',default=False)
	parser.add_argument("--oversampling",help="If not enough protons, sample with replacement",action='store_true',default=False)
		
	args = parser.parse_args()
	
	if args.train_split + args.validation_split >= 1.:
		raise Exception("Train split and validation split make up more than 100% of data")
	
	# Fractions and mixtures
	trainingFraction = args.train_split				# Fraction of electrons that go into training
	validationFraction = args.validation_split		# Fraction of electrons that go into validation
	validationMixture = args.validation_mixture		# Ratio protons/electrons for validation. Number greater than 1.
	testMixture = args.test_mixture				#  Ratio protons/electrons for testing
	
	if not args.onlymerge:
		selectedE_train, selectedP_train, selectedE_validate, selectedP_validate, selectedE_test, selectedP_test = getSetIndexes(nrofe,nrofp,trainingFraction,validationFraction,validationMixture,testMixture,args.oversampling)
		
		print "Got indices (", str(time.strftime('%H:%M:%S', time.gmtime( time.time() - t0 ))), ')'
		
		# Merging and splitting - electrons
		if not args.onlyprotons :
			
			print "--- Electrons ---"
			
			arr_e = np.load(electronFiles[0])
			i = 0
			for f in electronFiles:
				if i==0:
					i=i+1
					continue
				arr_e = np.concatenate( (arr_e , np.load(f) ) )
			print "Built the large array (", str(time.strftime('%H:%M:%S', time.gmtime( time.time() - t0 ))), ')'
			np.random.shuffle(arr_e)
			
			print "Saving train, validate, test arrays (", str(time.strftime('%H:%M:%S', time.gmtime( time.time() - t0 ))), ')'
			
			if args.oversampling:
				outname_e = '_elecs_over_' + str(validationMixture) + '.npy'
			else:
				outname_e = '_elecs_under_' + str(validationMixture) + '.npy'
			
			np.save('data_train'+outname_e , arr_e[ selectedE_train, :])
			np.save('data_validate'+outname_e , arr_e[ selectedE_validate, :])
			np.save('data_test'+outname_e , arr_e[ selectedE_test, :])
			np2root(arr_e[ selectedE_train, :] , getLabels() , outname='dataset_train'+outname_e.replace('npy','root'))
			np2root(arr_e[ selectedE_validate, :] , getLabels() , outname='dataset_validate'+outname_e.replace('npy','root'))
			np2root(arr_e[ selectedE_test, :] , getLabels() , outname='dataset_test'+outname_e.replace('npy','root'))
			
			del arr_e
			print "Done saving (", str(time.strftime('%H:%M:%S', time.gmtime( time.time() - t0 ))), ')'
	
		
		# Protons
		if not args.onlyelectrons :
			
			print "--- Protons ---"
			
			arr_p = np.load(protonFiles[0])
			i = 0
			for f in protonFiles:
				if i==0:
					i=i+1
					continue
				arr_p = np.concatenate( (arr_p , np.load(f) ) )
			print "Built the large array (", str(time.strftime('%H:%M:%S', time.gmtime( time.time() - t0 ))), ')'
			np.random.shuffle(arr_p)
			
			if args.oversampling:
				outname_p = '_prots_over_' + str(validationMixture) + '.npy'
			else:
				outname_p = '_prots_under_' + str(validationMixture) + '.npy'
			
			print "Saving proton train (", str(time.strftime('%H:%M:%S', time.gmtime( time.time() - t0 ))), ')'
			np.save('data_train'+outname_p , arr_p[ selectedP_train, :])
			np2root(arr_p[ selectedP_train, :] , getLabels() , outname='dataset_train'+outname_p.replace('npy','root') )
			
			print "Saving proton validate (", str(time.strftime('%H:%M:%S', time.gmtime( time.time() - t0 ))), ')'
			np.save('data_validate'+outname_p , arr_p[ selectedP_validate, :])
			np2root(arr_p[ selectedP_validate, :] , getLabels(),outname='dataset_validate'+outname_p.replace('npy','root') )

			print "Saving test (", str(time.strftime('%H:%M:%S', time.gmtime( time.time() - t0 ))), ')'
			np.save('data_test'+outname_p , arr_p[ selectedP_test, :])
			np2root(arr_p[ selectedP_test, :] , getLabels() , outname='dataset_test'+outname_p.replace('npy','root') )

			del arr_p
			print "Done saving (", str(time.strftime('%H:%M:%S', time.gmtime( time.time() - t0 ))), ')'
		
		
	
	# Concatenate electrons and protons
	if not args.onlyprotons and not args.onlyelectrons:
		
		if args.oversampling:
			outname_e = '_elecs_over_' + str(validationMixture) + '.npy'
			outname_p = '_prots_over_' + str(validationMixture) + '.npy'
			outname_all = '_over_' + str(validationMixture) + '.npy'
		else:
			outname_e = '_elecs_under_' + str(validationMixture) + '.npy'
			outname_p = '_prots_under_' + str(validationMixture) + '.npy'	
			outname_all = '_under_' + str(validationMixture) + '.npy'	
			
		print "Building train set..."
		train_set = np.concatenate( ( np.load('data_train'+outname_e) , np.load('data_train'+outname_p) ) )
		np.random.shuffle(train_set)
		np.save('dataset_train'+outname_all , train_set)
		np2root(train_set , getLabels() , outname='dataset_train'+outname_all.replace('npy','root'))
		del train_set
		print "Done  (", str(time.strftime('%H:%M:%S', time.gmtime( time.time() - t0 ))), ')'
		
		print "Building validation set..."
		set_e_validate = np.load('data_validate'+outname_e)
		set_p_validate = np.load('data_validate'+outname_p)
		validate_set = np.concatenate( (set_e_validate, set_p_validate ) )
		np.random.shuffle(validate_set)
		np.save('dataset_validate'+outname_all , validate_set)
		np2root(validate_set , getLabels() , outname='dataset_validate'+outname_all.replace('npy','root'))
		del validate_set, set_e_validate, set_p_validate
		print "Done  (", str(time.strftime('%H:%M:%S', time.gmtime( time.time() - t0 ))), ')'
		
		print "Building test set..."
		set_e_test = np.load('data_test'+outname_e)
		set_p_test = np.load('data_test'+outname_p)
		test_set = np.concatenate( (set_e_test, set_p_test ) )
		np.random.shuffle(test_set)
		np.save('dataset_test'+outname_all , test_set)
		np2root(test_set , getLabels() , outname='dataset_test'+outname_all.replace('npy','root') )
		del test_set, set_e_test, set_p_test
		print "Done (", str(time.strftime('%H:%M:%S', time.gmtime( time.time() - t0 ))), ')'
	
