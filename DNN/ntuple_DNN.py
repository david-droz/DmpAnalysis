'''

Code that takes as input one of Xin's analysis ntuples, and should return a root file containing the same TTree but with the DNN score

# https://root-forum.cern.ch/t/adding-a-branch-to-an-existing-tree/9449
# https://root-forum.cern.ch/t/pyroot-adding-a-branch-to-a-tree/2918/2
# https://root-forum.cern.ch/t/creating-branches-in-python/16677
# https://web.archive.org/web/20150124185243/http://wlav.web.cern.ch/wlav/pyroot/tpytree.html
# https://root.cern.ch/root/roottalk/roottalk01/0363.html

'''

from __future__ import print_function, division, absolute_import

from ROOT import TTree, TFile, TBranch
from keras.models import load_model
import numpy as np
import sys
import os
import math
import glob 
import yaml
from shutil import copyfile
from array import array


class MLntuple(object):
	
	def __init__(self,models_path,infiles=[]):
		
		
		self.modelLoc = sorted(glob.glob(models_path+'/*'))
		self.models = {}
		self.modelXmax = {}
		self.modelIndices = {}
		for k in self.modelLoc:
			self.modelXmax[os.path.basename(k)] = np.load(k+'/X_max.npy')
			self.models[os.path.basename(k)] = load_model(k+'/model.h5')
			self.modelIndices[os.path.basename(k)] = yaml.load(open(k+'/indices.yaml','r'))
		self.modelNames = [os.path.basename(x) for x in self.modelLoc]
		
		self.infiles = infiles
		self.setOutput()
		
		self.predictions = {}
		for k in self.models.keys():
			self.predictions[k] = {}
			for f in self.infiles:
				self.predictions[k][os.path.basename(f)] = []
	
	
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
				
	def run(self,suffix='DNN'):
		if not os.path.isdir(self.outdir): os.mkdir(self.outdir)
		
		for f in self.infiles:
			
			newF = self.outdir + "/" + os.path.basename(f).replace(".root","."+suffix+".root")
			if os.path.isfile(newF): 
				print("File exists: ", newF)
				continue
			copyfile(f,newF)
			
			try:
				TF = TFile(newF,'update')
				TT = TF.Get("DmlNtup")
				
				dicOfArrays = {}
				dicOfBranches = {}
				for k in self.models.keys():
					dicOfArrays[k] = array("f",[0.0])
					dicOfBranches[k] = TT.Branch( "MLscore_"+k,dicOfArrays[k], "MLscore_"+k+'/F')
				
				for n in range(0,TT.GetEntries()):
					predArray = np.zeros( (1,47) ) 
					pev = TT.GetEntry(n)
					erec = TT.tt_bgoTotalE_GeV * 1000		# DNN trained in MeV
					
					for frac_i in range(0,14):
						predArray[0,frac_i] = getattr(TT,"tt_F"+str(frac_i)) * erec	# Energy fraction goes like tt_F0, tt_F1, ...
						#~ predArray[n,frac_i] = getattr(TT,"tt_F"+str(frac_i))
					for rms_i in range(0,14):
						predArray[0,rms_i+14] = getattr(TT,"tt_Rms"+str(rms_i))
					for hits_i in range(0,14):
						try:
							predArray[0,hits_i+28] = ord(getattr(TT,"tt_nBarLayer"+str(hits_i)))
						except AttributeError:
							predArray[0,hits_i+28] = 0
					
					predArray[0,42] = TT.tt_Rmsl
					predArray[0,43] = TT.tt_Rmsr			
					predArray[0,44] = erec
					predArray[0,45] = TT.tt_nBgoHits
					
					XZ = TT.tt_bgoRecSlopeX
					YZ = TT.tt_bgoRecSlopeY
					tgZ = math.atan(np.sqrt( (XZ*XZ) + (YZ*YZ) ) )
					
					predArray[0,46] = tgZ*180./math.pi
					
					#Prediction part
					for k in self.models.keys():
						
						try:
							self.predictions[k][os.path.basename(f)].append( self.models[k].predict( predArray[:,self.modelIndices[k]]/self.modelXmax[k]) )
						except KeyError:
							self.predictions[k][os.path.basename(f)] = [ self.models[k].predict( predArray[:,self.modelIndices[k]]/self.modelXmax[k]) ]
							
						
						
						dicOfArrays[k][0] = self.predictions[k][os.path.basename(f)][n]
						dicOfBranches[k].Fill()
				# END FOR
				TT.Write()
				TF.Close()
			except:
				os.remove(newF)
				raise
			
	def savePredictions(self,outfile):
		with open(outfile,'w') as f:
			yaml.dump(self.predictions,f)
	



if __name__ == '__main__' :
	
	suffix = 'DNN'
	
	analyser = MLntuple(sys.argv[1])
	analyser.addFile(sys.argv[2])
	analyser.run(suffix=suffix)
	
	
	analyser.savePredictions(sys.argv[3])
