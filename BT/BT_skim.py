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
		2. Particle name
		3. Run number
		4. Dataset name
		5. Run type (MC/BT)
		
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
from BT_to_numpy import identifyParticle


def analysis(files,pid,nr,dataset,runtype):
	
	folder = './tmp/'
	temp_basename = os.path.basename(sys.argv[1])
	temp_basename = os.path.splitext(temp_basename)[0]
	folder = folder + temp_basename
	
	skim_out = '/beegfs/dampe/prod/UserSpace/ddroz/BT/'
	if not os.path.isdir(skim_out): os.mkdir(skim_out)
	
	if not os.path.isdir(folder): os.mkdir(folder)
	
	if pid == 11:
		outstr = folder + '/elec_' + str(nr) + '.npy'
		skim_out = skim_out + 'BT_Electron'
	elif pid == 2212:
		outstr = folder + '/prot_' + str(nr) + '.npy'
		skim_out = skim_out + 'BT_Electron'
	elif pid == 22:
		outstr = folder + '/gamma_' + str(nr) + '.npy'
	
	if runtype == "MC":
		skim_out += '_MC.root'
	else:
		skim_out += '_BT.root'	
		
	if os.path.isfile(outstr):
		return
	
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
			
	np.save(outstr,np.array(a))
	
	print "Selected ", selected, " events"
	print "Rejected ", rejected, " events"
	
	del a
	#~ dmpch.Terminate()
	newTree.Write()
	skimFile.Close()
	return
	




if __name__ == '__main__' :
	
	'''
	Inputs:
	'''
	
	if len(sys.argv) < 6:
		print "ERROR - Expected arguments:"
		print "1. Input file"
		print "2. Particle name"
		print "3. Run number"
		print "4. Dataset name"
		print "5. Run type (MC/BT)"
		sys.exit(-1)
	
	if ".root" in sys.argv[1]:
		filelist=[sys.argv[1]]
	else:
		filelist = []
		with open(sys.argv[1],'r') as f:
			for lines in f:
				if ".root" in lines:
					filelist.append(lines.replace('\n',''))
	
	particle = identifyParticle(sys.argv[2])
	
	if not os.path.isdir('tmp'):
		os.mkdir('tmp')
		
	analysis(filelist,particle,int(sys.argv[3]),sys.argv[4],sys.argv[5])
