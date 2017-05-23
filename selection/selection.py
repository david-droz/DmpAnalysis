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
			
	if len([r for r in BHER if r]) < 14: 
		return False
	if [SIDE[s] for s in [1,2,3] if SIDE[s] ]: 		# "First layers not side" ... ?
		return False
	if bhm > 0.35: 				# "Max layer cut" ... ?
		return False
	# End Andrii's electron cut
	templist = []				
	for layer in [1,2,12,13]:	# Numbers of BGO layers, minus one  (starting from 0 to 13)
		templist.append(  pev.pEvtBgoRec().GetELayer(layer)  )
	if not any(templist):	# Making sure that there was at least one hit in one of the BGO layers
		return False
	
	return True	


def getBGOvalues(pev):
	templist = []
	BHXS = [0. for i in xrange(14)]	
	for i in xrange(14):	# Numbers of BGO layers
		
		templist.append(  pev.pEvtBgoRec().GetELayer(i)  )

		im = None				
		em = 0.0;
		for j in xrange(22):						# Find the maximum of energy (em) and its position (im)
			ebar = pev.pEvtBgoRec().GetEdep(i,j)
			if ebar < em : continue 
			em = ebar
			im = j
		if not em: continue
		if im in [0,21]:
			cog = 27.5 * im	
		else:	
			ene = 0.0			
			cog = 0.0
			for j in [im-1, im, im+1]: 				# im : bar that was hit, and then check the two neighbours
				ebar = pev.pEvtBgoRec().GetEdep(i,j)
				ene+=ebar
				cog+= 27.5 * j * ebar
			cog/=ene					# Center of energy distribution.	CoG = Center of Gravity
		posrms   = 0.0
		enelayer = 0.0
		for j in xrange(22):
			ebar = pev.pEvtBgoRec().GetEdep(i,j)
			posbar = 27.5 * j 		# 27.5mm : horizontal position of BGO bar
			enelayer += ebar
			posrms += ebar * (posbar-cog)*(posbar-cog)
		BHXS[i] = math.sqrt( posrms / enelayer)
	
	for item in BHXS:
		templist.append( item )
		
	templist.append( pev.pEvtBgoRec().GetRMS_l() )
	templist.append( pev.pEvtBgoRec().GetRMS_r() )

	templist.append( pev.pEvtBgoRec().GetElectronEcor() )
	templist.append( pev.pEvtBgoRec().GetTotalHits() )
	
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
		5-8. RMS of energy on layer 1a, 1b, 2a, 2b.
	
	'''
	templist = []
	
	templist.append(pev.pEvtPsdRec().GetLayerEnergy(0))
	templist.append(pev.pEvtPsdRec().GetLayerEnergy(1))
	templist.append(pev.pEvtPsdRec().GetLayerHits(0))
	templist.append(pev.pEvtPsdRec().GetLayerHits(1))

	PSD_total_hits = pev.NEvtPsdHits()
	if PSD_total_hits == 0:
		for i in xrange(4): 
			templist.append(rms)
		return templist

	l_pos = np.zeros(PSD_total_hits)
	l_z = np.zeros(PSD_total_hits)
	l_energy = np.zeros(PSD_total_hits)
	
	for i in xrange(PSD_total_hits):
		pos = pev.pEvtPsdHits().GetHitX(i)
		if pos == 0:
			pos = pev.pEvtPsdHits().GetHitY(i)
		z = pev.pEvtPsdHits().GetHitZ(i)
		energy = pev.pEvtPsdHits().fEnergy[i]
		
		l_pos[i] = pos
		l_z[i] = z
		l_energy[i] = energy
		
	minz = np.min(l_z)
	maxz = np.max(l_z)
	bins = np.linspace(minz,maxz,5)		# 4 bins
	
	for i in xrange(4):
		
		cog = 0
		ene_tot = 0
		for j in xrange(PSD_total_hits):
			if l_z[j] < bins[i] or l_z[j] > bins[i+1]:	# Wrong bin
				continue
			ene_tot = ene_tot + l_energy[j]
			cog = cog + l_pos[j]*l_energy[j]
		if ene_tot == 0:
			templist.append(0)
			continue
		cog = cog/ene_tot
		rms = 0
		for j in xrange(PSD_total_hits):
			if l_z[j] < bins[i] or l_z[j] > bins[i+1]:	# Wrong bin
				continue
			rms = rms + (l_energy[j] * (l_pos[j] - cog) * (l_pos[j] - cog) )
		rms = math.sqrt(rms/ene_tot)
	
		templist.append(rms)
	
	del l_z, l_energy, l_pos
	return templist
	
def getSTKvalues(pev):
	'''
	https://dampevm3.unige.ch/doxygen/trunk/Documentation/html/classDmpEvent.html
	https://dampevm3.unige.ch/doxygen/trunk/Documentation/html/classDmpStkTrack.html
	
	DmpEvent :  Int_t 	NStkSiCluster ()
	'''
	templist = []
	
	nrofclusters = pev.NStkSiCluster()
	templist.append(nrofclusters)
	templist.append(pev.NStkKalmanTrack())
	
	if nrofclusters == 0:
		for i in xrange(8):
			templist.append(0)
			templist.append(0)
		return templist
	
	l_pos = np.zeros(nrofclusters)
	l_z = np.zeros(nrofclusters)
	l_energy = np.zeros(nrofclusters)
	l_width = np.zeros(nrofclusters)
	
	for i in xrange(nrofclusters):
		pos = pev.pStkSiCluster(i).GetH()
		z = pev.pStkSiCluster(i).GetZ()
		energy = pev.pStkSiCluster(i).getEnergy()
		width = pev.pStkSiCluster(i).getWidth()
		
		l_pos[i] = pos
		l_z[i] = z
		l_energy[i] = energy
		l_width[i] = width
	
	minz = np.min(l_z)
	maxz = np.max(l_z)
	bins = np.linspace(minz,maxz,9)		# 8 bins
	
	ene_per_bin = []
	rms_per_bin = []
	
	for i in xrange(8):
		
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
	
	del l_pos, l_z, l_energy, l_width, ene_per_bin, rms_per_bin
	return templist	

def getValues(pev):
	'''
	templist:
		0 - 13 : Energy in BGO layer i
		14 - 27 : RMS of energy deposited in layer i
		28 : longitudinal RMS ( DmpEvtBgoRec::GetRMS_l )
		29 : radial RMS ( DmpEvtBgoRec::GetRMS_r )
		30 : total BGO energy (corrected)
		31 : total BGO hits
		----
		32 - 33 : Energy in PSD layer 1,2
		34 - 35 : Nr of hits in PSD layer 1,2
		36 - 39 : RMS of energy deposited in PSD layer 1a,1b,2a,2b
		----
		40 : nr of Si clusters
		41 : nr of tracks
		42 - 49 : energy in STK clusters, 8 vertical bins
		50 - 57 : RMS of energy in STK clusters, 8 vertical bins
		----
		58 : timestamp
		59 : Particle ID (0 for proton, 1 for electron)
	'''
	templist = []

	### BGO
	templist = templist + getBGOvalues(pev)
	
	### PSD
	templist = templist + getPSDvalues(pev)
	
	### STK
	templist = templist + getSTKvalues(pev)
	
	sec = pev.pEvtHeader().GetSecond()			
	msec = pev.pEvtHeader().GetMillisecond()
	if msec >= 1. :
		msec = msec / 1000.
	templist.append(sec + msec)
	
	if pev.pEvtSimuPrimaries().pvpart_pdg == 11 :
		templist.append(1)
	else:
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
		
		if selection(pev,pid):
			templist = getValues(pev)
			a.append(templist)
		else :
			continue
		
	arr = np.array(a)
	
	np.save(outstr,arr)
	
	del arr, a, dmpch
	return
		
def merge():
	print " Concatenating..."
	
	for particle in ['elec','prot']:
		
		listofsubarr = glob.glob('./tmp/*/' + particle + '*.npy')
		
		bigarr = np.load(listofsubarr[0])
		
		for miniarr in listofsubarr[1:]:
			bigarr = np.concatenate((bigarr, np.load(miniarr) ))
		
		outstr = 'dataset_' + particle + '.npy'
		bigarr = bigarr[np.any(bigarr,axis=1)]
		for i in xrange(5):
			np.random.shuffle(bigarr)
		
		np.save(outstr,bigarr)
		if particle=='elec':
			nrofelectrons = bigarr.shape[0]
		else:
			nrofprotons = bigarr.shape[0]
		del bigarr
	print "Done."
	print "Number of protons: ", nrofprotons
	print "Number of electrons: ", nrofelectrons		

if __name__ == "__main__" :
	
	t0 = time.time()

	filelist = []
	with open(sys.argv[1],'r') as f:
		for lines in f:
			filelist.append(lines.replace('\n',''))

	if len(sys.argv) > 2:
		particle = identifyParticle(sys.argv[2])
	else:
		particle = identifyParticle(sys.argv[1])
	
	if particle == 2212:
		nrofchunks = 1000
	elif particle == 11:
		nrofchunks = 350
	chunksize = len(filelist)/nrofchunks
	
	
	if not os.path.isdir('tmp'):
		os.mkdir('tmp')
	
	for i in xrange(nrofchunks):
		print "--- Chunk ", i
		
		if len(filelist) < chunksize: break
		
		chunk = []
		for k in xrange(chunksize):
			chunk.append( filelist.pop(0) )
		
		analysis(chunk,particle,i)
		
	analysis(filelist,particle,nrofchunks)
	
	print 'Run time: ', str(strftime('%H:%M:%S', gmtime( time.time() - t0 )))
