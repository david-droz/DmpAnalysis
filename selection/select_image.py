'''
selection.py v0.1

Selects events from ROOT files, builds them as images and write NumPy arrays ready to use by deep neural networks

Usage:
> python selection.py filelist particle
	filelist: ASCII file of ROOT files (pre-skimmed)
	particle: the particle to select. Can be of the following formats:
			e = ['e','elec','electron','11','E','Elec','Electron']
			p = ['p','prot','proton','2212','P','Prot','Proton']
			gamma = ['g','gamma','photon','22','Gamma','Photon']
		if not supplied, script tries to guess particle based on filelist

'''

from __future__ import division

import math
import numpy as np
import ctypes
import sys
import glob
import argparse
import os
import time
from ROOT import gSystem
gSystem.Load("libDmpEvent.so")
import ROOT
import cPickle as pickle
import gc
gc.enable()
import yaml

from selection import containmentCut, cutMaxELayer, maxBarCut, openRootFile, identifyParticle

		
def analysis(files,pid,nr):
	'''
	Select good events from a filelist and saves them as a numpy array
	'''
	folder = './img/'
	temp_basename = os.path.basename(sys.argv[1])
	temp_basename = os.path.splitext(temp_basename)[0]
	folder = folder + temp_basename
	
	if not os.path.isdir(folder): os.mkdir(folder)
	
	if pid == 11:
		outstr = folder + '/elec_' + str(nr) + '.npy'
	elif pid == 2212:
		outstr = folder + '/prot_' + str(nr) + '.npy'
	elif pid == 22:
		outstr = folder + '/gamma_' + str(nr) + '.npy'
	elif pid == 'flight' :
		outstr = folder + '/flight_' + str(nr) + '.npy'
		
	if os.path.isfile(outstr):
		return
	
	dmpch = openRootFile(files)
	nvts = dmpch.GetEntries()
	
	a = []	# Image list
	y = []	# Labels
	for evt in xrange(nvts):
			
		pev = dmpch.GetDmpEvent(evt)
		if not pev.pEvtHeader().GeneratedTrigger(3) : continue
		
		edep = np.zeros((14,21))
		for i in xrange(14):
			for j in xrange(21):
			edep[i,j] = pev.pEvtBgoRec().GetEdep(i,j)
		a.append(edep)
		
			try:
				if pev.pEvtSimuPrimaries().pvpart_pdg == 11 :		# Electron
					y.append(1)
				elif pev.pEvtSimuPrimaries().pvpart_pdg == 22 :		# Photon
					y.append(2)
				else:												# Proton
					y.append(0)
			except:													# Flight
				y.append(0)
		
		
	arr = np.array(a)	
	np.save(outstr.replace('.npy','_X.npy'),arr)
	np.save(outstr.replace('.npy','_Y.npy'),np.array(y))
	del arr, a
	dmpch.Terminate()

if __name__ == "__main__" :
	
	if ".root" in sys.argv[1]:
		filelist=[sys.argv[1]]
	else:
		filelist = []
		with open(sys.argv[1],'r') as f:
			for lines in f:
				if ".root" in lines:
					filelist.append(lines.replace('\n',''))
	
	if len(sys.argv) < 3:
		raise Exception("Not enough arguments")
	
	particle = identifyParticle(sys.argv[2])
	
	if not os.path.isdir('img'):
		os.mkdir('img')
		
	analysis(filelist,particle,int(sys.argv[3]))
	
