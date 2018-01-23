'''
selection.py v0.9

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
import yaml

from ../skim/MC_skimmer import containmentCut, cutMaxELayer, maxBarCut

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
	
def getEventWeight(pev):
	EKin = pev.pEvtSimuPrimaries().pvpart_ekin
	pid = pev.pEvtSimuPrimaries().pvpart_pdg
	
	nProtonHighE   = 781390000
	nProtonLowE    = 492600000
	nProtonVHE     = 2*11526500	
	
	nElectronHighE = 133740000
	nElectronLowE  = 499600000
	
	w = EKin/100000.
	
	if pid == 11 :			# electron
		w = 1/(w*w)
	elif pid == 2212 :		# proton
		w = w**(-1.7)
	
	if EKin < 100 * 1e+3 :
		if pid == 2212 : w *= nProtonHighE/nProtonLowE
		elif pid == 11 : w *= nElectronHighE/nElectronLowE
	elif EKin > 10 * 1e+6 :
		if pid == 2212 : w *= nProtonHighE/nProtonVHE
		
	return w

def getXTRL(pev):
	
	'''
	From a given event, returns the energy ratio and energy RMS in all BGO layers, and XTR/XTRL/zeta/whatever-it-is-called
	'''
	
	NBGOLAYERS  = 14
	NBARSLAYER  = 22
	EDGEBARS    = [0,21]
	BARPITCH    = 27.5
	
	edep = np.zeros((NBGOLAYERS,NBARSLAYER))
	for i in xrange(NBGOLAYERS):
		for j in xrange(NBARSLAYER):
			edep[i,j] = pev.pEvtBgoRec().GetEdep(i,j)
	
	BHET = edep.sum()
	BHXS = [0. for i in xrange(NBGOLAYERS)]
	BHER = [0. for i in xrange(NBGOLAYERS)]
	COG = [0. for i in xrange(NBGOLAYERS)]
	bhm  = 0.
	SIDE = [False for i in xrange(NBGOLAYERS)]
	
	for i in xrange(NBGOLAYERS):
		# Find the bar with max energy deposition of a layer and record its number as im
		im = None
		em = 0.0;
		for j in xrange(NBARSLAYER):
			ebar = edep[i,j]
			if ebar < em : continue 
			em = ebar
			im = j
		
		if not em: continue
		
		if im in EDGEBARS:
			cog = BARPITCH * im   #BHX[i][im]
			
		else:
			ene = 0.
			cog = 0.
			for  j in [im-1, im, im+1]: 
				ebar = edep[i,j]
				ene += ebar
				cog += BARPITCH * j * ebar
			cog /= ene
			
		posrms   = 0.
		enelayer = 0.
		for j in xrange(NBARSLAYER):
			ebar = edep[i,j]
			posbar = BARPITCH * j 
			enelayer += ebar
			posrms += ebar * (posbar-cog)*(posbar-cog)
		posrms = math.sqrt( posrms / enelayer)
		BHXS[i] = posrms
		COG[i] = cog
		BHER[i] = enelayer / BHET
	
	sumRMS = sum(BHXS)
	F = [r for r in reversed(BHER) if r][0]
	XTRL = sumRMS**4.0 * F / 8000000.
	
	del edep
	
	return BHER, BHXS, XTRL

def selection(pev,particle,cutStat):
	'''
	Returns True if good event, False otherwise
	'''
	def incrementKey(dic,key):
		if key in dic.keys():
			dic[key] += 1
		else:
			dic[key] = 1

	if not pev.pEvtHeader().GeneratedTrigger(3): 
		incrementKey(cutStat,'HET')
		return False
		
	if not containmentCut(pev.pEvtBgoRec()):
		incrementKey(cutStat,'Containment')
		return False
	elif not cutMaxELayer(pev.pEvtBgoRec()):
		incrementKey(cutStat,'MaxELayer')
		return False
	elif not maxBarCut(pev):
		incrementKey(cutStat,'maxBar')
		return Falsei
	
	erec = pev.pEvtBgoRec().GetElectronEcor()
	
	if erec < 10 * 1e+3:		# 10 GeV
		incrementKey(cutStat,'10GeV')
		return False
	elif erec > 10 * 1e+6:		# 10 TeV
		incrementKey(cutStat,'10TeV')
		return False
		
	
	
	return True


def getBGOvalues(pev):
	'''
	Extract values related to BGO and write them as a python list.
	'''
	templist = []
	
	#~ RMS2 = pev.pEvtBgoRec().GetRMS2()		# Obsolete. Let's use the manual computation instead.
	ELayer, RMS, zeta = getXTRL(pev)
	
	# Energy per layer
	for i in xrange(14): templist.append( ELayer[i]  )
	
	# RMS2 per layer
	for j in xrange(14): templist.append( RMS[j] )
	
	# Hits on every layer		
	hitsPerLayer = pev.pEvtBgoRec().GetLayerHits()
	for k in xrange(14):
		templist.append(hitsPerLayer[k])
		
	templist.append( pev.pEvtBgoRec().GetRMS_l() )
	templist.append( pev.pEvtBgoRec().GetRMS_r() )

	#~ templist.append( pev.pEvtBgoRec().GetElectronEcor() )
	templist.append( pev.pEvtBgoRec().GetTotalEnergy() )
	templist.append( pev.pEvtBgoRec().GetTotalHits() )
	
	# Angle of reconstructed trajectory
	XZ = pev.pEvtBgoRec().GetSlopeXZ()
	YZ = pev.pEvtBgoRec().GetSlopeYZ()
	
	tgZ = math.atan(np.sqrt( (XZ*XZ) + (YZ*YZ) ) )
	templist.append(tgZ*180./math.pi)
	
	return templist

def getPSDvalues(pev):
	'''
	Extracts PSD values and return as a Python list
	'''
	templist = []
	
	templist.append(pev.pEvtPsdRec().GetLayerEnergy(0))
	templist.append(pev.pEvtPsdRec().GetLayerEnergy(1))
	templist.append(pev.pEvtPsdRec().GetLayerHits(0))
	templist.append(pev.pEvtPsdRec().GetLayerHits(1))

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
	templist.append(nrofclusters)
	templist.append(pev.NStkKalmanTrack())
	
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
		templist.append(x)
	for y in rms_per_bin:
		templist.append(y)
	
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
	return templist

def getValues(pev,i):
	'''
	List of variables:
		0 - 13 : Energy in BGO layer i
		14 - 27 : RMS of energy deposited in layer i
		28 - 41 : Number of hits in layer i
		
		42 : longitudinal RMS ( DmpEvtBgoRec::GetRMS_l )
		43 : radial RMS ( DmpEvtBgoRec::GetRMS_r )
		44 : total BGO energy 
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
		65 : EneLayerMax/Etot. Must be <0.5 for MC skim, <0.35 for data skim
		----
		66 : timestamp
		67 : True energy (EKin). Set to 0 if missing (i.e. flight data)
		68 : Event index
		69 : Event weight (weight according to energy spectrum)
		70 : XTRL
		71 : Particle ID (0 for proton, 1 for electron, 2 for photon)
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
	
	### EneLayerMax/Etot
	minilist = np.zeros((14))
	for i in range(14):
		minilist[i] = pev.pEvtBgoRec().GetELayer(i)
	templist.append( minilist.max()/ minilist.sum() )
	del minilist
	
	### Timestamp
	sec = pev.pEvtHeader().GetSecond()					# Timestamp is used as an unique particle identifier for data. If need be.
	msec = pev.pEvtHeader().GetMillisecond()
	if msec >= 1. :
		msec = msec / 1000.
	templist.append(sec + msec)
	
	### EKin
	try:
		EKin = pev.pEvtSimuPrimaries().pvpart_ekin
	except:
		EKin = 0
	templist.append(EKin)
	
	### Event index
	templist.append(i)
	
	### Event weight
	try:
		w = getEventWeight(pev)
	except:
		w = 1
		raise
	templist.append(w)
	
	### XTRL
	ELayer, RMS, zeta = getXTRL(pev)
	templist.append(zeta)	
	del ELayer, RMS, zeta
	
	### Event label
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
	
	if not os.path.isdir('statistics'): os.mkdir('statistics')
	
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
	selectionStatistics = {}
	fileIndexing = {}
	cutStatistics = {}
	for i in xrange(nvts):
			
		pev = dmpch.GetDmpEvent(i)
		currentFileName = os.path.basename(dmpch.GetFile().GetName())
		if currentFileName not in selectionStatistics.keys():
			selectionStatistics[currentFileName] = [0,0]
		
		##
		sel = selection(pev,pid,cutStatistics)
		##
		
		if not sel:
			selectionStatistics[currentFileName][1] += 1
			continue
		
		selectionStatistics[currentFileName][0] += 1
		
		if currentFileName not in fileIndexing.keys():
			fileIndexing[currentFileName] = [i]
		else:
			fileIndexing[currentFileName].append(i)
		
		a.append(getValues(pev,i))
		
		
		# Next: 
		#		write the three dictionaries to files (e.g. yaml file)
		
	base = os.path.splitext(os.path.basename(sys.argv[1]))[0]
	with open('statistics/'+base+'selStat.yaml','w') as f:
		yaml.dump(selectionStatistics,f)
	with open('statistics/'+base+'fileIndex.yaml','w') as f:
		yaml.dump(fileIndexing,f)
	with open('statistics/'+base+'cutStat.yaml','w') as f:
		yaml.dump(cutStatistics,f)
	
		
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
	
