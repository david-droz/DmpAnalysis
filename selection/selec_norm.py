'''
Selects events from ROOT files, extracts useful variables, normalises the variables and write NumPy arrays ready to use by deep neural networks

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

def openRootFile(efilelist): 
	'''
	Returns a TChain from a filelist
	'''
	chain = ROOT.DmpChain("CollectionTree")
	for f in efilelist:
		chain.Add(f)
	if not chain.GetEntries():
		raise IOError("0 events in DmpChain - something went wrong")
	return chain

def identifyParticle(part):
	'''
	Particle identification based on either the argument or the file name
	'''
	e = ['e','elec','electron','11','E','Elec','Electron']
	p = ['p','prot','proton','2212','P','Prot','Proton']
	gamma = ['g','gamma','photon','22','Gamma','Photon']
	
	for cat in [e,p,gamma]:
		if part in cat:
			return int(cat[3])
	
	for cat in [e,p,gamma]:
		for x in cat[1:]:
			if x in part:
				return int(cat[3])
	
	raise Exception("Particle not identified - " + part)

def getBGOvalues(pev):
	'''
	Extract values related to BGO and write them as a python list.
	'''
	templist = []
	
	RMS2 = pev.pEvtBgoRec().GetRMS2()
	totalEnergy = pev.pEvtBgoRec().GetElectronEcor()
	sumRMS = 0
	for i in xrange(14): sumRMS += RMS2[j]
	totalHits = pev.pEvtBgoRec().GetTotalHits()
	
	# Energy per layer
	for i in xrange(14): templist.append(  pev.pEvtBgoRec().GetELayer(i) / totalEnergy  )
	
	# RMS2 per layer
	for j in xrange(14): 
		if RMS2[j] < 0 :		# In PMO code, if RMS is not defined then RMS2 = -999. Prefer to move it to 0.
			templist.append( 0 )
		else:
			templist.append( RMS2[j] / sumRMS )
	
	# Hits on every layer		
	hitsPerLayer = pev.pEvtBgoRec().GetLayerHits()
	for k in xrange(14):
		templist.append(hitsPerLayer[k] / totalHits)
	
	return templist

def getPSDvalues(pev):
	'''
	Extracts PSD values and return as a Python list
	'''
	templist = []
	
	totalE = pev.pEvtPsdRec().GetLayerEnergy(0) + pev.pEvtPsdRec().GetLayerEnergy(1)
	totalHits = pev.pEvtPsdRec().GetLayerHits(0) + pev.pEvtPsdRec().GetLayerHits(1)
	
	templist.append(pev.pEvtPsdRec().GetLayerEnergy(0) / totalE)
	templist.append(pev.pEvtPsdRec().GetLayerEnergy(1) / totalE)
	templist.append(pev.pEvtPsdRec().GetLayerHits(0) / totalHits)
	templist.append(pev.pEvtPsdRec().GetLayerHits(1) / totalHits)

	return templist
	
def getSTKvalues(pev):
	'''
	Extracts STK values and return as list
	'''
	templist = []
	nBins = 4				# In DmpSoftware package, STK is not defined per layers. 
							# Here we treat it as a calorimeter
	# Nr of clusters, nr of tracks
	nrofclusters = pev.NStkSiCluster()
	#~ templist.append(nrofclusters)
	#~ templist.append(pev.NStkKalmanTrack())
	
	if nrofclusters == 0:
		for i in xrange(nBins):
			templist.append(0)
			templist.append(0)
		return templist
	
	# Below: compute the RMS of cluster distributions
	l_pos = np.zeros(nrofclusters)
	l_z = np.zeros(nrofclusters)
	l_energy = np.zeros(nrofclusters)
	
	for i in xrange(nrofclusters):
		pos = pev.pStkSiCluster(i).GetH()
		z = pev.pStkSiCluster(i).GetZ()
		energy = pev.pStkSiCluster(i).getEnergy()
		
		l_pos[i] = pos
		l_z[i] = z
		l_energy[i] = energy
	
	minz = np.min(l_z)
	maxz = np.max(l_z)
	bins = np.linspace(minz,maxz,nBins+1)
	
	ene_per_bin = []
	rms_per_bin = []
	
	for i in xrange(nBins):
		
		cog = 0
		ene_tot = 0
		for j in xrange(nrofclusters):
			if l_z[j] < bins[i] or l_z[j] > bins[i+1]:	# Wrong bin
				continue
			ene_tot = ene_tot + l_energy[j]
			cog = cog + l_pos[j]*l_energy[j]
		if ene_tot == 0:
			ene_per_bin.append(0)
			rms_per_bin.append(0)
			continue
		cog = cog/ene_tot
		rms = 0
		ene_per_bin.append(ene_tot)
		for j in xrange(nrofclusters):
			if l_z[j] < bins[i] or l_z[j] > bins[i+1]:	# Wrong bin
				continue
			rms = rms + (l_energy[j] * (l_pos[j] - cog) * (l_pos[j] - cog) )
		rms = math.sqrt(rms/ene_tot)
		rms_per_bin.append(rms)

	for x in ene_per_bin:
		templist.append(x / sum(ene_per_bin))
	for y in rms_per_bin:
		templist.append(y / sum(rms_per_bin))
	
	del l_pos, l_z, l_energy, ene_per_bin, rms_per_bin
	return templist	
	
def getNUDvalues(pev):
	'''
	Extract raw ADC signal from NUD
	'''
	templist = [0 for x in xrange(4)]
	f = pev.pEvtNudRaw().fADC
	for i in xrange(4): 
		templist[i] = f[i]
	total = sum(templist)
	return [x / total for x in templist]

def getValues(pev):
	'''
	List of variables:
		0 - 13 : Fraction of energy in BGO layer i
		14 - 27 : RMS2 of energy deposited in layer i, divided by sumRMS2
		28 - 41 : Fraction of hits in layer i
		----
		42 - 43 : Fraction of energy in PSD layer 1,2
		44 - 45 : Fraction of hits in PSD layer 1,2
		----
		46 - 49 : Fraction of energy in STK clusters, 4 vertical bins
		50 - 53 : RMS of energy in STK clusters, divided by sumRMS, 4 vertical bins
		----
		54 - 57 : Raw NUD signal, divided by total
		----
		58 : timestamp
		59 : Particle ID (0 for proton, 1 for electron, 2 for photon)
	'''
	templist = []

	### BGO
	templist = templist + getBGOvalues(pev)
	
	### PSD
	templist = templist + getPSDvalues(pev)
	
	### STK
	templist = templist + getSTKvalues(pev)
	
	### NUD
	templist = templist + getNUDvalues(pev)
	
	### Timestamp
	sec = pev.pEvtHeader().GetSecond()					# Timestamp is used as an unique particle identifier for data. If need be.
	msec = pev.pEvtHeader().GetMillisecond()
	if msec >= 1. :
		msec = msec / 1000.
	templist.append(sec + msec)
	
	if pev.pEvtSimuPrimaries().pvpart_pdg == 11 :		# Electron
		templist.append(1)
	elif pev.pEvtSimuPrimaries().pvpart_pdg == 22 :		# Photon
		templist.append(2)
	else:												# Proton
		templist.append(0)

	
	return templist
	

		
def analysis(files,pid,nr):
	'''
	Select good events from a filelist and saves them as a numpy array
	'''
	folder = './tmp/'
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
		
	if os.path.isfile(outstr):
		return
	
	dmpch = openRootFile(files)
	nvts = dmpch.GetEntries()
	
	a = []
	for i in xrange(nvts):
		pev = dmpch.GetDmpEvent(i)
		
		#~ if selection(pev,pid):
			#~ templist = getValues(pev)
			#~ a.append(templist)
		#~ else :
			#~ continue
		
		if not pev.pEvtHeader().GeneratedTrigger(3): continue		# High energy trigger, recommended by Valentina	
		
		a.append(getValues(pev))
		
	arr = np.array(a)
	
	np.save(outstr,arr)
	
	del arr, a
	
	dmpch.Terminate()
	return	

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
	
	if not os.path.isdir('tmp'):
		os.mkdir('tmp')
		
	analysis(filelist,particle,int(sys.argv[3]))
	
