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
import pickle


def computeSumRMS(pev):
	
	RMSarr = pev.pEvtBgoRec().GetRMS2()
	s = 0
	for i in xrange(14): s+=RMSarr[i]
	#~ 
	#~ s = pev.pEvtBgoRec().GetRMS_r()
	
	return s
	
	
def computeFLAST(pev):
	
	return pev.pEvtBgoRec().GetELayer(13) / pev.pEvtBgoRec().GetTotalEnergy()
	
	
def buildXTRL(pev):
	
	Flast = computeFLAST(pev)
	SumRMS = computeSumRMS(pev)
	
	XTRL = Flast * (SumRMS**4) / 8e+6
	
	return XTRL, Flast, SumRMS
	
def cuts(pev):
	
	if pev.pEvtBgoRec().GetTotalEnergy() < 1e+3 : return False
	if pev.pEvtBgoRec().GetELayer(13) == 0 : return False
	if computeFLAST(pev) > 0.2 : return False
	
	return True
	
	
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
	flast_e = []
	flast_p = []
	rms_e = []
	rms_p = []
	for ne in xrange(nrofe):
		pev = chain_e.GetDmpEvent(ne)
		
		if not cuts(pev): continue
		x,f,r = buildXTRL(pev)
		xtrl_e.append( x )
		flast_e.append( f )
		rms_e.append( r )
		
	for np in xrange(nrofp):
		ppev = chain_p.GetDmpEvent(np)
		
		if not cuts(pev): continue
			
		x,f,r = buildXTRL(pev)
		xtrl_p.append( x )
		flast_p.append( f )
		rms_p.append( r )
		
	with open("XTRL.pick",'w') as h:
		pickle.dump([xtrl_e,xtrl_p],h)
		
	return xtrl_e , xtrl_p, flast_e, flast_p, rms_e, rms_p
	
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
	plt.legend(loc='best')
	plt.savefig('XTRL')
	
def make2DHisto(a,b,c,d):
	
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	
	fig1 = plt.figure()
	plt.hist2d(c+d,a+b,bins=80,norm=matplotlib.colors.LogNorm())
	plt.colorbar()
	plt.yscale('log')
	plt.xlabel('Sum RMS')
	plt.ylabel('FLast')
	plt.savefig('2d')
	
	fig2 = plt.figure()
	plt.hist(c,100,histtype='step',label='e')
	plt.hist(d,100,histtype='step',label='p')
	plt.xlabel('Sum RMS')
	plt.legend(loc='best')
	plt.savefig('sumRMS')
	
	fig3 = plt.figure()
	plt.hist2d(c,a,bins=80,norm=matplotlib.colors.LogNorm())
	plt.colorbar()
	plt.yscale('log')
	plt.xlabel('Sum RMS')
	plt.ylabel('FLast')
	plt.savefig('2d_electrons')
	
	fig4 = plt.figure()
	plt.hist2d(d,b,bins=80,norm=matplotlib.colors.LogNorm())
	plt.colorbar()
	plt.yscale('log')
	plt.xlabel('Sum RMS')
	plt.ylabel('FLast')
	plt.savefig('2d_protons')
	

if __name__ == '__main__':
	
	xe, xp, fe, fp, re, rp = run()
	
	makeHisto(xe,xp)
	
	make2DHisto(fe,fp,re,rp)
