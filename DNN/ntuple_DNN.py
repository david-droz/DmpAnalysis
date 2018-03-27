'''

Code that takes as input one of Xin's analysis ntuples, and should return a root file containing the same TTree but with the DNN score

'''

from __future__ import print_function, division, absolute_import

from ROOT import TTree, TFile
from keras.models import load_model
import numpy as np
import sys
import os
import math


class MLntuple(object):
	
	def __init__(self,model_path='model_100.h5',xmax_path='X_max.npy',infiles=[]):
		
		try:
			self.xmax = np.load(xmax_path)
		except IOError:
			raise IOError("Cannot locate/load normalisation file: ", xmax_path)
		
		try:
			self.model = load_model(model_path)
		except OSError:
			raise IOError("Cannot locate/load model file: ", model_path)
		
		self.infiles = infiles
		self.setOutput()
		self.predictions = {}
	
	def setOutput(self,directory="DNN_ntuples"):
		self.outdir = directory
		
		
	def addFromList(self,l):
		self.infiles += l
	def addFromFile(self,infile):
		with open(infile,r) as f:
			for item in f:
				self.infiles.append(item.replace('\n',''))
	def addFile(self,f):
		if ".root" in f:
			self.infiles.append(f)
		else:
			try:
				cond = os.path.isfile(f)
			except TypeError:
				self.addFromList(f)
			else:
				if cond: self.addFromFile(f)
				else: raise IOError("Cannot find file / Cannot understand type of: ", f)
		
	def _predict(self,arr,f):
		'''
		Dictionary mapping filenames to numpy arrays of predictions
		'''
		self.predictions[os.path.basename(f)] = self.model.predict( arr / self.xmax )
		
	def getPredictions(self):
		return self.predictions	
		
	def run(self):
		if not os.path.isdir(self.outdir): os.mkdir(self.outdir)
		
		for f in self.infiles:
			
			TF = TFile(f,'READ')
			TT = TF.Get("DmlNtup")
			predArray = np.zeros( (int(TT.GetEntries()), self.xmax.shape[0]) )
			
			for n in range(0,TT.GetEntries()):
				pev = TT.GetEntry(n)
				erec = TT.tt_bgoTotalE_GeV * 1000		# DNN trained in MeV
				
				for frac_i in range(0,14):
					#~ predArray[n,frac_i] = getattr(TT,"tt_F"+str(frac_i)) * erec	# Energy fraction goes like tt_F0, tt_F1, ...
					predArray[n,frac_i] = getattr(TT,"tt_F"+str(frac_i))
				for rms_i in range(0,14):
					predArray[n,rms_i+14] = getattr(TT,"tt_Rms"+str(rms_i))
				for hits_i in range(0,14):
					predArray[n,hits_i+28] = ord(getattr(TT,"tt_nBarLayer"+str(hits_i)))
							
				predArray[n,42] = TT.tt_Rmsl
				predArray[n,43] = TT.tt_Rmsr
				predArray[n,44] = erec
				predArray[n,45] = TT.tt_nBgoHits
				
				XZ = TT.tt_bgoRecSlopeX
				YZ = TT.tt_bgoRecSlopeY
				tgZ = math.atan(np.sqrt( (XZ*XZ) + (YZ*YZ) ) )
				
				predArray[n,46] = tgZ*180./math.pi
			# END FOR
			
			self._predict(predArray,f)
			np.save('data.npy',predArray)
			
			TF.Close()
				



if __name__ == '__main__' :
	
	analyser = MLntuple()
	analyser.addFile(sys.argv[1])
	analyser.run()
	
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	
	d = analyser.getPredictions()
	fig1 = plt.figure()
	for f in d.keys():
		arr = d[f]
		arr = arr[ np.absolute(arr) < 40 ]
		plt.hist(arr,50,histtype='step',label=f)
	plt.legend(loc='best')
	plt.savefig('test')
