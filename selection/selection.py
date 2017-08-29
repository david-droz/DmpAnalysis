'''
selection.py v0.1

Selects events from ROOT files, extracts useful variables and write NumPy arrays ready to use by deep neural networks

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

def selection(pev,particle):
	'''
	Returns True if good event, False otherwise
	'''
	if not pev.pEvtHeader().GeneratedTrigger(3):		# High energy trigger
		return False
	
	if pev.pEvtSimuPrimaries().pvpart_pdg != particle:
		return False
		
	if pev.pEvtBgoRec().GetElectronEcor() < 1e+5:	# 100 GeV
		return False
		
	BHET = sum([ 
				sum([
					pev.pEvtBgoRec().GetEdep(i,j) for j in xrange(22)
					])  
				for i in xrange(14) 
			])
	BHXS = [0. for i in xrange(14)]
	BHER = [0. for i in xrange(14)]
	bhm  = 0.
	SIDE = [False for i in xrange(14)]
	
	for i in xrange(14):				
		im = None				# Find the bar with max energy deposition of a layer and record its number as im
		em = 0.0;
		for j in xrange(22):		# 22 BGO bars
			ebar = pev.pEvtBgoRec().GetEdep(i,j)
			if ebar < em : continue 
			em = ebar                  
			im = j		
			
		if not em: continue
		
		if im in [0,21]:		# Edge bars (first and last BGO bars)
			cog = 27.5 * im		# 27.5 = BARPITCH    What is that? No idea
		else:	
			ene = 0.0			# I have no idea what this is doing. 
			cog = 0.0			# cog = center of gravity?
			for j in [im-1, im, im+1]: 
				ebar = pev.pEvtBgoRec().GetEdep(i,j)
				ene+=ebar
				cog+= 27.5 * j * ebar
			cog/=ene
			
		posrms   = 0.0
		enelayer = 0.0
		for j in xrange(22):
			ebar = pev.pEvtBgoRec().GetEdep(i,j)
			posbar = 27.5 * j 
			enelayer += ebar
			posrms += ebar * (posbar-cog)*(posbar-cog)
		posrms = math.sqrt( posrms / enelayer)
		BHXS[i] = posrms
		BHER[i] = enelayer / BHET
		
		if im in [0,21]:
			SIDE[i] = True
			
	if [SIDE[s] for s in [1,2,3] if SIDE[s] ]: 		# "First layers not side" ... ?
		return False
	if bhm > 0.35: 				# "Max layer cut" ... ?
		return False
	# End Andrii's electron cut
	
	return True	


def getBGOvalues(pev):
	templist = []
	
	RMS2 = pev.pEvtBgoRec().GetRMS2()
	
	for i in xrange(14): templist.append(  pev.pEvtBgoRec().GetELayer(i)  )
	
	for j in xrange(14): 
		if RMS2[j] < 0 :
			templist.append( 0 )
		else:
			templist.append( RMS2[j] )
			
	hitsPerLayer = pev.pEvtBgoRec().GetLayerHits()
	for k in xrange(14):
		templist.append(hitsPerLayer[k])
		
	templist.append( pev.pEvtBgoRec().GetRMS_l() )
	templist.append( pev.pEvtBgoRec().GetRMS_r() )

	templist.append( pev.pEvtBgoRec().GetElectronEcor() )
	templist.append( pev.pEvtBgoRec().GetTotalHits() )
	
	XZ = pev.pEvtBgoRec().GetSlopeXZ()
	YZ = pev.pEvtBgoRec().GetSlopeYZ()
	
	tgZ = math.atan(np.sqrt( (XZ*XZ) + (YZ*YZ) ) )
	templist.append(tgZ*180./math.pi)
	
	return templist

def getPSDvalues(pev):
	'''
	https://dampevm3.unige.ch/doxygen/trunk/Documentation/html/classDmpEvtPsdHits.html
	https://dampevm3.unige.ch/doxygen/trunk/Documentation/html/classDmpEvtPsdRec.html
	
	Returns a list:
		1. Energy on layer 1
		2. Energy on layer 2
		3. Nr of hits on layer 1
		4. Nr of hits on layer 2
	
	'''
	templist = []
	
	templist.append(pev.pEvtPsdRec().GetLayerEnergy(0))
	templist.append(pev.pEvtPsdRec().GetLayerEnergy(1))
	templist.append(pev.pEvtPsdRec().GetLayerHits(0))
	templist.append(pev.pEvtPsdRec().GetLayerHits(1))

	
	return templist
	
def getSTKvalues(pev):
	'''
	https://dampevm3.unige.ch/doxygen/trunk/Documentation/html/classDmpEvent.html
	https://dampevm3.unige.ch/doxygen/trunk/Documentation/html/classDmpStkTrack.html
	
	DmpEvent :  Int_t 	NStkSiCluster ()
	'''
	templist = []
	nBins = 4
	
	nrofclusters = pev.NStkSiCluster()
	templist.append(nrofclusters)
	templist.append(pev.NStkKalmanTrack())
	
	if nrofclusters == 0:
		for i in xrange(nBins):
			templist.append(0)
			templist.append(0)
		return templist
	
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
		templist.append(x)
	for y in rms_per_bin:
		templist.append(y)
	
	del l_pos, l_z, l_energy, ene_per_bin, rms_per_bin
	return templist	
	
def getNUDvalues(pev):
	templist = [0 for x in xrange(4)]
	f = pev.pEvtNudRaw().fADC
	for i in xrange(4): 
		templist[i] = f[i]
	return templist

def getValues(pev):
	'''
	templist:
		0 - 13 : Energy in BGO layer i
		14 - 27 : RMS2 of energy deposited in layer i
		28 - 41 : Number of hits in layer i
		
		42 : longitudinal RMS ( DmpEvtBgoRec::GetRMS_l )
		43 : radial RMS ( DmpEvtBgoRec::GetRMS_r )
		44 : total BGO energy (corrected)
		45 : total BGO hits
		46 : theta angle of BGO trajectory
		----
		47 - 48 : Energy in PSD layer 1,2
		49 - 50 : Nr of hits in PSD layer 1,2
		----
		51 : nr of Si clusters
		52 : nr of tracks
		53 - 56 : energy in STK clusters, 4 vertical bins
		57 - 60 : RMS of energy in STK clusters, 4 vertical bins
		----
		61 - 64 : Raw NUD signal
		----
		65 : timestamp
		66 : Particle ID (0 for proton, 1 for electron, 2 for photon)
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
	sec = pev.pEvtHeader().GetSecond()			
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
	
