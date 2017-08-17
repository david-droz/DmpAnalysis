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

'''
./trunk/kernel/DmpParameterBgo.cc
./trunk/kernel/DmpBgoBase.cc
./trunk/Reconstruction/Bgo/src/DmpAlgBgoRecExtra.cc
./trunk/Geometry/Parameters/BGO.xml
'''

	
def CalcPosition():							# DmpAlgBgoRecExtra
	'''
	Godawful function patched together from DmpSoftware source code.
	'''

	# Begin: global geometric parameters
	m_BgoBarX = 25							# BGO.xml
	m_BgoBarY = 600
	m_BgoBarZ = 25
	m_BgoDetLayerSeparation = 4
	m_BgoDetBarSeparation = 2.5
	
	m_BgoDetLayerBar_Z = []					# Z position of every layer.  DmpParameterBgo.h
	m_BgoDetLayerBar_XY = []
	
	for i in range(14):						# DmpParameterBgo :: LoadBgoParameter()
		BarCenterZ = -196.25 + i*(m_BgoBarZ + m_BgoDetLayerSeparation) + 254.75		# BGO.xml ; DAMPE.xml
		m_BgoDetLayerBar_Z.append(BarCenterZ)
	
	FirstBarCenterXY = 0. - 10.5*(m_BgoBarX + m_BgoDetBarSeparation)
	for i in range(22):
		BarCenterXY = FirstBarCenterXY + i*(m_BgoBarX + m_BgoDetBarSeparation)
		m_BgoDetLayerBar_XY.append(BarCenterXY)
	# End: global geometric parameters
	
	def BGObarCenter(ID):					# DmpParameterBgo :: BarCenter(const short)	
		layerID = (ID>>11) &0x000f			# Black magic fuckery
		barID = (ID>>6) &0x001f				# Source: DmpBgoBase :: GetLayerID(const short) and GetBarID(const short)
		
		pos = [m_BgoDetLayerBar_XY[barID],m_BgoDetLayerBar_Z[layerID]]

		return pos
	
	position = np.zeros((14,22,2))
	for i in range(14):
		for j in range(22):
			barID = (i<<11) + (j<<6)		# DmpBgoBase :: ConstructGlobalBarID(const short, const short)
			a = BGObarCenter(barID)
			position[i,j,0] = a[0]			# XY position
			position[i,j,1] = a[1]			# Z position
			
	return position
	

def getEdep(pev):
	
	Edep = np.zeros((14,22))
	for i in xrange(14):
		for j in xrange(22):
			Edep[i,j] = pev.pEvtBgoRec().GetEdep(i,j)
			
	return Edep
	

def computeCOG(position,EdepArr):				# DmpAlgBgoRecExtra
	'''
	Center Of Gravity (COG) of a particle shower in the 14 layers of the BGO
	'''
	
	cog = np.zeros((14))
	
	for i in range(14):
		cog[i] = -999
		layerEnergy = np.sum(EdepArr[i])
		if layerEnergy < 0.1: continue
		maxEnergy = np.max(EdepArr[i])
		maxBar = np.argmax(EdepArr[i])
		
		if maxBar in [0,21] :				# Side event
			cog[i] = position[i,maxBar,0]
		else:
			
			cog_p1 = position[i,maxBar - 1,0] * EdepArr[i,maxBar - 1]
			cog_p2 = position[i,maxBar,0] * EdepArr[i,maxBar]
			cog_p3 = position[i,maxBar + 1,0] * EdepArr[i,maxBar + 1]
			
			cog[i] = sum([cog_p1,cog_p2,cog_p3]) / (EdepArr[i,maxBar - 1] + EdepArr[i,maxBar] + EdepArr[i,maxBar + 1])
			
	return cog
	
def calcRMS(pev,position,EdepArr):
	
	coG = computeCOG(position,EdepArr)
	RMS2 = np.zeros((14))
	
	for i in range(14):
		RMS2[i] = -999
		re = 0
		layerEnergy = np.sum(EdepArr[i])
		if coG[i] < -900 : continue
		
		for j in range(22):
			re += EdepArr[i,j] * (position[i,j,0] - coG[i]) * (position[i,j,0] - coG[i])
		
		if layerEnergy > 0.1:
			RMS2[i] = re / layerEnergy
	
	return RMS2
	

def computeSumRMS(pev,position):
	
	#~ RMSarr = pev.pEvtBgoRec().GetRMS2()
	#~ s = 0
	#~ for i in xrange(14): s+=RMSarr[i]
	#~ 
	#~ s = pev.pEvtBgoRec().GetRMS_r()
	
	RMS2 = calcRMS(pev,position,getEdep(pev))
	s = 0
	for i in xrange(14): s+= np.sqrt(RMS2[i])
	
	return s
	
	
def computeFLAST(pev):
	
	return (pev.pEvtBgoRec().GetELayer(13) / pev.pEvtBgoRec().GetTotalEnergy())
	
	
def buildXTRL(pev,position):
	
	Flast = computeFLAST(pev)
	SumRMS = computeSumRMS(pev,position)
	
	XTRL = Flast * (SumRMS**4) / 8e+6
	
	return XTRL, Flast, SumRMS
	
def cuts(pev):
	
	if pev.pEvtBgoRec().GetTotalEnergy() < 1e+3 : return False
	if pev.pEvtBgoRec().GetELayer(13) == 0 : return False
	
	f = computeFLAST(pev)
	if f > 0.2 or f < 1e-5 : return False
	
	#~ if computeSumRMS(pev) > 3e+5 : return False
	
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
	
	position = CalcPosition()
	
	xtrl_e = []
	xtrl_p = []
	flast_e = []
	flast_p = []
	rms_e = []
	rms_p = []
	energy_e = []
	energy_p = []
	for ne in xrange(nrofe):
		pev = chain_e.GetDmpEvent(ne)
		
		if not cuts(pev): continue
		x,f,r = buildXTRL(pev,position)
		xtrl_e.append( x )
		flast_e.append( f )
		rms_e.append( r )
		energy_e.append( pev.pEvtBgoRec().GetElectronEcor() )
		
	for np in xrange(nrofp):
		ppev = chain_p.GetDmpEvent(np)
		
		if not cuts(pev): continue
			
		x,f,r = buildXTRL(pev,position)
		xtrl_p.append( x )
		flast_p.append( f )
		rms_p.append( r )
		energy_p.append( pev.pEvtBgoRec().GetElectronEcor() )
		
	with open("XTRL.pick",'w') as h:
		pickle.dump([xtrl_e,xtrl_p],h)
		
	return xtrl_e , xtrl_p, flast_e, flast_p, rms_e, rms_p, energy_e, energy_p
	
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
	
def energyVSxtr(ee,xe,ep,xp):
	
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	
	fig1 = plt.figure()
	plt.hist2d(ee+ep,xe+xp,bins=80)
	plt.xlabel('Corrected BGO energy')
	plt.ylabel('XTR')
	plt.xscale('log')
	plt.colorbar()
	plt.savefig('2d_XTR')
	
	fig2 = plt.figure()
	plt.hist2d(ee,xe,bins=80)
	plt.xlabel('Corrected BGO energy')
	plt.ylabel('XTR')
	plt.xscale('log')
	plt.title('Electrons')
	plt.colorbar()
	plt.savefig('2d_XTR_electrons')
	
	fig3 = plt.figure()
	plt.hist2d(ep,xp,bins=80)
	plt.xlabel('Corrected BGO energy')
	plt.ylabel('XTR')
	plt.xscale('log')
	plt.title('Protons')
	plt.colorbar()
	plt.savefig('2d_XTR_protons')
	
	

if __name__ == '__main__':
	
	xe, xp, fe, fp, re, rp, ee, ep = run()
	
	makeHisto(xe,xp)
	
	make2DHisto(fe,fp,re,rp)
	
	energyVSxtr(ee,xe,ep,xp)
