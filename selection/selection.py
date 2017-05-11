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

def openRootFile(efilelist): 
	'''
	Returns a TChain from a filelist
	'''
	chain = ROOT.DmpChain("CollectionTree")
	for f in efilelist:
		chain.Add(f)
	if not chain.GetEntries():
		raise IOError("0 events in DmpChain - something went wrong")
	return echaine

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

def getValues(pev):
	'''
	templist:
		0 - 13 : Energy in BGO layer i
		14 - 27 : RMS of energy deposited in layer i
		28 : longitudinal RMS ( DmpEvtBgoRec::GetRMS_l )
		29 : radial RMS ( DmpEvtBgoRec::GetRMS_r )
		30 : total BGO energy (corrected)
		31 : total hits
		32 : timestamp
		33 : Particle ID (0 for proton, 1 for electron)
	'''
	templist = []
	BHXS = [0. for i in xrange(14)]	
	for i in xrange(14):	# Numbers of BGO layers
		
		####
		templist.append(  pev.pEvtBgoRec().GetELayer(i)  )
		####
		
		im = None				
		em = 0.0;
		for j in xrange(22):
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
		
		####
		templist.append( item )
		####
		
	templist.append( pev.pEvtBgoRec().GetRMS_l() )
	templist.append( pev.pEvtBgoRec().GetRMS_r() )
	
	#~ templist.append( pev.pEvtBgoRec().GetTotalEnergy() )
	templist.append( pev.pEvtBgoRec().GetElectronEcor() )
	templist.append( pev.pEvtBgoRec().GetTotalHits() )
	
	
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
	
	if pid == 11:
		outstr = './tmp/elec_' + str(nr) + '.npy'
	else:
		outstr = './tmp/prot_' + str(nr) + '.npy'
		
	if os.path.isfile(outstr):
		return
	
	dmpch = openRootFile(files)
	nvts = dmpch.GetEntries()
	
	a = []
	for i in xrange(nvts):
		pev = dmpch.GetDmpEvent(i)
		
		if selection(pev,pid):
			templist = getValues(pev)
		else :
			continue
			
			a.append(templist)

	arr = np.array(a)
	
	np.save(outstr,arr)
	
	#~ return arr.shape[0]
	return
		
def merge():
	print " Concatenating..."
	
	for particle in ['elec','prot']:
		
		listofsubarr = glob.glob('./tmp/' + particle + '*.npy')
		
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
	
	nrofchunks = 300
	chunksize = len(filelist)/nrofchunks
	
	if len(sys.argv) > 2:
		particle = identifyParticle(sys.argv[2])
	else:
		particle = identifyParticle(sys.argv[1])
	
	if not os.path.isdir('tmp'):
		os.mkdir('tmp')
	
	for i in xrange(nrofchunks):
		print "--- Chunk ", i
		
		if len(filelist) < chunksize: break
		
		chunk = []
		for k in xrange(chunksize):
			chunk.append( filelist.pop(0) )
		
		analysis(chunk,particle,i)
