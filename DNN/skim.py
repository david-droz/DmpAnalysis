'''

sys.argv :
1 - file name
2 - Beginning of loop
3 - End of loop

'''



print "Importing..."

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
import sys


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
	
	
def selection(pev):
	
	if pev.pEvtSimuPrimaries().pvpart_pdg not in [11,2112,2212]:
		return False
	HEtrigger = pev.pEvtHeader().GeneratedTrigger(3)
	if not HEtrigger:
		return False
	
	return True
	### 


		
def analysis(files):
	'''
	Select good events from a filelist and saves them as a numpy array
	'''
	
	dmpch = openRootFile(files)
	dmpch.SetOutputDir("skim_data")
	nvts = dmpch.GetEntries()

	a = 0
	b = 0

	
	for i in xrange(nvts):
		pev = dmpch.GetDmpEvent(i)
		
		if selection(pev):
			a += 1
			dmpch.SaveCurrentEvent()
		else:
			b += 1

	dmpch.Terminate()	
	return a, b
		
		

if __name__ == "__main__" :
	
	t0 = time.time()
	
	filelist = []
	with open(sys.argv[1],'r') as f:
		for lines in f:
			filelist.append(lines.replace('\n',''))

	begin = int(sys.argv[2])
	end = int(sys.argv[3])

	filelist = filelist[begin:end]
	
	print begin, ' - ', end
	
	selected, skipped = analysis(filelist)
	print "- Selected ", selected, " events"
	print "- Skipped ", skipped, " events"
	
	print "Total running time: ", time.strftime('%H:%M:%S', time.gmtime(time.time() - t0))
