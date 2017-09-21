'''

Select BeamTest data/MC and build into NumPy ntuples for usage with a DNN

Source: Andrii's code

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
import pickle

from BTeventSelection import BTselection 
from BT_getValues import getValues

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

def analysis(files,pid,nr,dataset):
	
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
	
	#~ dmpch = openRootFile(files)
	dmpch = ROOT.TChain("CollectionTree")
	for f in files:
		dmpch.Add(f)
	nvts = dmpch.GetEntries()
	if not nvts: raise IOError('0 events in TChain!!')
	
	###
	
	bgorec = ROOT.DmpEvtBgoRec()
	dmpch.SetBranchAddress("DmpEvtBgoRec", bgorec)
	b_bgorec = dmpch.GetBranch("DmpEvtBgoRec")

	nudraw = ROOT.DmpEvtNudRaw()
	dmpch.SetBranchAddress("DmpEvtNudRaw", nudraw)
	b_nudraw = dmpch.GetBranch("DmpEvtNudRaw")

	evtheader = ROOT.DmpEvtHeader()
	dmpch.SetBranchAddress("EventHeader", evtheader)

	psdhits  = ROOT.DmpEvtPsdHits()
	dmpch.SetBranchAddress("DmpPsdHits", psdhits) 
	
	bgohits = ROOT.DmpEvtBgoHits()
	dmpch.SetBranchAddress("DmpEvtBgoHits",bgohits)
	
	stktracks = ROOT.TClonesArray("DmpStkTrack")
	dmpch.SetBranchAddress("StkKalmanTracks", stktracks)

	stkclusters = ROOT.TClonesArray("DmpStkSiCluster")
	dmpch.SetBranchAddress("StkClusterCollection",stkclusters)
	
	psdrec = ROOT.DmpEvtPsdRec()
	dmpch.SetBranchAddress("DmpEvtPsdRec",psdrec)
	
	trackhelper = ROOT.DmpStkTrackHelper(stktracks, False)
	
	###
	
	a = []
	selected = 0
	rejected = 0
	for i in xrange(nvts):
		
		dmpch.GetEntry(i)
		
		if BTselection(bgorec, b_bgorec, nudraw, b_nudraw, evtheader, psdhits, bgohits, stktracks, stkclusters, trackhelper, dataset):
			#~ a.append(getValues(pev,pid))
			a.append(getValues(bgorec, b_bgorec, nudraw, b_nudraw, evtheader, psdhits, bgohits, stktracks, stkclusters, trackhelper,psdrec,pid))
			selected += 1
		else:
			rejected += 1
			
	np.save(outstr,np.array(a))
	
	print "Selected ", selected, " events"
	print "Rejected ", rejected, " events"
	
	del a
	#~ dmpch.Terminate()
	return
	




if __name__ == '__main__' :
	
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
		
	analysis(filelist,particle,int(sys.argv[3]),sys.argv[4])
