'''

Get XTRL histograms from ROOT files

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



def computeSumRMS(pev):
	
	RMSarr = pev.pEvtBgoRec().GetRMS2()
	s = 0
	for i in xrange(14): s+=RMSarr[i]
	
	return s
	
	
def computeFLAST(pev):
	
	return pev.pEvtBgoRec().GetELayer(13) / pev.pEvtBgoRec().GetElectronEcor()
	
	
def buildXTRL(pev):
	
	Flast = computeFLAST(pev)
	SumRMS = computeSumRMS(pev)
	
	XTRL = Flast * SumRMS / 8e+6
	
	return XTRL
	
	
def run():
	
	with open('/dampe/data4/users/ddroz/getXTRL/allElectrons.txt','r') as f:
		allElectrons = []
		for line in f:
			allElectrons.append( line.replace('\n','') )
	with open('/dampe/data4/users/ddroz/getXTRL/allProtons.txt','r') as g:
		allProtons = []
		for line in g:
			allProtons.append( line.replace('\n','') )
			
	chain_e = ROOT.DmpChain("CollectionTree")
	for f in allElectrons:
		chain_e.Add(f)
	chain_p = ROOT.DmpChain("CollectionTree")
	for g in allProtons:
		chain_p.Add(g)
		
	nrofe = chain_e.GetEntries()
	if not nrofe: raise Exception("No electrons in DmpChain!")
	nrofp = chain_p.GetEntries()
	if not nrofp: raise Exception("No protons in DmpChain!")
	
	xtrl_e = []
	xtrl_p = []
	for ne in xrange(nrofe):
		pev = chain_e.GetDmpEvent(ne)
		xtrl_e.append( buildXTRL(pev) )
	for np in xrange(nrofp):
		ppev = chain_p.GetDmpEvent(np)
		xtrl_p.append( buildXTRL(ppev) )
		
	with open("XTRL.pick",'w') as h:
		pickle.dump([xtrl_e,xtrl_p],h)
		
	return xtrl_e , xtrl_p
	
def makeHisto(a,b):
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	
	ma = max(a)
	mb = max(b)
	m = max([ma,mb])
	
	binlist = [ x*float(m)/100. for x in range(101) ]
	
	fig1 = plt.figure()
	plt.hist(a,bins=binlist,histtype='step',label='e',color='green')
	plt.hist(b,bins=binlist,histtype='step',label='p',color='red',ls='dashed')
	plt.yscale('log')
	plt.xlabel('XTRL')
	plt.grid(True)
	plt.savefig('XTRL')
	
	

if __name__ == '__main__':
	
	xe, xp = run()
	
	makeHisto(xe,xp)
