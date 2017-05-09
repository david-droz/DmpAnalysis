

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

	print "Building TChain ..."            
	echaine = ROOT.DmpChain("CollectionTree")

	for f in efilelist:
		echaine.Add(f)
			    
	if not echaine.GetEntries():
		raise IOError("0 events in DmpChain - something went wrong")
    
	return echaine
	
	
def protonselection(pev):
	'''
	Returns False if it's a bad event, or returns the list of energies in BGO layers if it's a good event
	'''	
	if not pev.pEvtHeader().GeneratedTrigger(3):		# High energy trigger
		return False
	
	particleID = pev.pEvtSimuPrimaries().pvpart_pdg   # 11 for electron, 2212 for proton	
	if particleID != 2212:
		return False
		
	# Begin Andrii's electron cut
	
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
			cog = 0.0
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
		
	for item in BHXS:
		templist.append( item )
	
	templist.append( pev.pEvtBgoRec().GetTotalEnergy())
		
	return templist
	
def electronselection(pev):
	'''
	Returns False if it's a bad event, or returns the list of energies in BGO layers if it's a good event
	'''	
	particleID = pev.pEvtSimuPrimaries().pvpart_pdg   # 11 for electron, 2212 for proton
	if particleID != 11:
		return False	
	HEtrigger = pev.pEvtHeader().GeneratedTrigger(3)
	if not HEtrigger:		# High energy trigger: a particle that has triggered all first four BGO layers with an energy of 20 MIP or more.
		return False
	
	# Begin Andrii's electron cut
	
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
			cog = 27.5 * im		# 27.5 = BARPITCH     It's the position of a bar in mm (thickness of bar + thickness of what's in between, or something like that)
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
		posrms = math.sqrt( posrms / enelayer)
		BHXS[i] = posrms
		BHER[i] = enelayer / BHET
		
		if im in [0,21]:
			SIDE[i] = True
			
	if len([r for r in BHER if r]) < 14: 			# BHER: ratio of energy per layer. Require that all have energy
		return False
	if [SIDE[s] for s in [1,2,3] if SIDE[s] ]: 		# Require no hit on the sides of layers 1,2,3 
		return False
	if bhm > 0.35: 				# "Max layer cut" ... ?
		return False
	
	# End Andrii's electron cut
	
	templist = []				
	for layer in [1,2,12,13]:	# Numbers of BGO layers, minus one  (starting from 0 to 13)
		templist.append(  pev.pEvtBgoRec().GetELayer(layer)  )

	if not any(templist):	# Making sure that there was at least one hit in one of the BGO layers
		return False
		
	for item in BHXS:
		templist.append( item )
	templist.append( pev.pEvtBgoRec().GetTotalEnergy())	
	return templist
		
def analysis(files,pid,nr):
	'''
	Select good events from a filelist and saves them as a numpy array
	'''
	
	if pid == 0:
		outstr = './tmp/elec_' + str(nr) + '.npy'
	else:
		outstr = './tmp/prot_' + str(nr) + '.npy'
		
	if os.path.isfile(outstr):
		arr = np.load(outstr)
		a = arr.shape[0]
		del arr
		return a
	
	dmpch = openRootFile(files)
	nvts = dmpch.GetEntries()
	arr = np.zeros([nvts,20])
	
	for i in xrange(nvts):
		pev = dmpch.GetDmpEvent(i)
		if pid == 0:
			templist = electronselection(pev)
		else:
			templist = protonselection(pev)
		
		if templist:
			for j in xrange(arr.shape[1] - 1):
				arr[i][j] = templist[j]
			if pid == 0:
				arr[i][-1] = 1		# electron
	arr = arr[np.any(arr,axis=1)]
	

	np.save(outstr,arr)
	
	return arr.shape[0]
		
		

if __name__ == "__main__" :
	
	t0 = time.time()
	
	try:
		if "proton" in sys.argv[1] or "Proton" in sys.argv[1] or "electron" in sys.argv[2] or "Electron" in sys.argv[2]:
			raise IOError("Usage : First the electron list and then the proton list, not the opposite")
	except IndexError:
		raise IndexError("Forgot arguments 1 and 2.   > python Histo_BGO_v7.py electronlist.txt protonlist.txt")
	efilelist = []
	pfilelist = []
	with open(sys.argv[1],'r') as f:
		for lines in f:
			efilelist.append(lines.replace('\n',''))
	with open(sys.argv[2],'r') as fg:
		for lines in fg:
			pfilelist.append(lines.replace('\n',''))
						
	nrofchunks = 200
	print "Iterating in ", nrofchunks, " chunks"
	
	if not os.path.isdir('tmp'):
		os.mkdir('tmp')
	
	iteratorPID = 0
	for filelist in [efilelist,pfilelist]:
		
		if iteratorPID == 0:
			print "------ Electrons ------"
		else:
			print "------ Protons --------"
		
		smallerlist = [ filelist[i::nrofchunks] for i in xrange(nrofchunks) ]
		iteratorChunk = 0
		selected = 0
		for chunk in smallerlist:
			print "--- Chunk ", iteratorChunk
			selected += analysis(chunk,iteratorPID,iteratorChunk)
			iteratorChunk += 1
		
		print "- Selected ", selected, " events"
		iteratorPID += 1
	

	
	print "----------------------"
	print "All loops done"	
	print " Concatenating..."
	
	for particle in ['elec','prot']:
		
		listofsubarr = glob.glob('./tmp/' + particle + '*.npy')
		bigarr = np.zeros([1,20])
		
		for miniarr in listofsubarr:
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
	print "Total running time: ", time.strftime('%H:%M:%S', time.gmtime(time.time() - t0))
