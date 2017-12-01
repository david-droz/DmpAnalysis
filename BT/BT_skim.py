'''

Whatever quests drives you 
To read this source code,
Forget it.
You will find nothing
In these desolated wastelands.

----------------

Select BeamTest data/MC and write ROOT files

Source of cuts: Andrii's code

Inputs:
		1. Input file
		2. Run number
		3. Dataset name
		4. Run type (MC/BT)
		
Output:
	- Root files under ..../UserSpace/ddroz
	- Numpy files under ./tmp/

'''

from __future__ import division

import math
import numpy as np
import ctypes
import sys
import glob
import os
import time
from ROOT import gSystem
gSystem.Load("libDmpEvent.so")
import ROOT
import pickle

from BTeventSelection import BTselection 
from BT_getValues import getValues


def analysis(infile,nr,dataset,runtype):
	
	# Build file list
	if ".root" in infile:
		files=[infile]
	else:
		files = []
		with open(infile,'r') as f:
			for lines in f:
				if ".root" in lines:
					files.append(lines.replace('\n',''))
	
	# Build file name for output numpy file
	if not os.path.isdir('tmp'):
		os.mkdir('tmp')
	folder = './tmp/'
	temp_basename = os.path.basename(infile)
	temp_basename = os.path.splitext(temp_basename)[0]
	folder = folder + temp_basename
	
	if not os.path.isdir(folder): os.mkdir(folder)
	outstr = folder + '/array_' + str(nr) + '.npy'	
	if os.path.isfile(outstr):
		return
		
	# Build file name for skimmed data	
	skim_out = '/beegfs/dampe/prod/UserSpace/ddroz/BT/'
	if not os.path.isdir(skim_out): os.mkdir(skim_out)
	
	if runtype == "MC":
		skim_out += 'MC/'
	else:
		skim_out += 'BT/'	
	if not os.path.isdir(skim_out): os.mkdir(skim_out)

	baseInfile = os.path.basename(infile)
	if 'part' in baseInfile:
		k = baseInfile.find('-part')
		skim_out = skim_out + baseInfile[0:k] + '/'
	else:
		skim_out = skim_out + os.path.splitext(baseInfile)[0] + '/'
	if not os.path.isdir(skim_out): os.mkdir(skim_out)
	
	skim_out = skim_out + os.path.splitext(baseInfile)[0] + '_' + str(nr) + '.root'
	
	###
	
	# Initialise TChain
	
	#~ dmpch = ROOT.DmpChain("CollectionTree")
	dmpch = ROOT.TChain("CollectionTree")
	for f in files:
		dmpch.Add(f)
	nvts = dmpch.GetEntries()
	if not nvts: raise IOError('0 events in TChain!!')
	
	###
	
	# Set branches
	
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
	
	# Set output file
	
	skimFile = ROOT.TFile(skim_out,"recreate")
	dmpch.LoadTree(0)
	newTree = dmpch.GetTree().CloneTree(0)
	
	### Event loop
	
	a = []
	selected = 0
	rejected = 0
	for i in xrange(nvts):
		
		dmpch.GetEntry(i)
		
		if BTselection(bgorec, b_bgorec, nudraw, b_nudraw, evtheader, psdhits, bgohits, stktracks, stkclusters, trackhelper, dataset):
			#~ dmpch.SaveCurrentEvent()
			newTree.Fill()
			a.append(getValues(bgorec, b_bgorec, nudraw, b_nudraw, evtheader, psdhits, bgohits, stktracks, stkclusters, trackhelper,psdrec,pid))
			selected += 1
		else:
			rejected += 1
			
	
	
	print "Selected ", selected, " events"
	print "Rejected ", rejected, " events"
	
	del a
	#~ dmpch.Terminate()
	newTree.Write()
	skimFile.Close()
	
	np.save(outstr,np.array(a))
	
	return
	




if __name__ == '__main__' :
	
	'''
	Inputs:
	'''
	
	if len(sys.argv) < 5:
		print "ERROR - Expected arguments:"
		print "1. Input file"
		print "2. Run number"
		print "3. Dataset name"
		print "4. Run type (MC/BT)"
		sys.exit(-1)
	
	
	analysis(sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4])
