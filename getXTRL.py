'''

Get XTRL histograms from ROOT files

'''

from __future__ import division

import math
import numpy as np
import sys
import os
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
		layerEnergy = np.sum(EdepArr[i])
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
		re = 0
		layerEnergy = np.sum(EdepArr[i])
		
		for j in range(22):
			re += EdepArr[i,j] * (position[i,j,0] - coG[i]) * (position[i,j,0] - coG[i])
		
		RMS2[i] = re / layerEnergy
	
	return RMS2
	

def computeSumRMS(pev,position,Edep):
	
	#~ RMSarr = pev.pEvtBgoRec().GetRMS2()
	#~ s = 0
	#~ for i in xrange(14): s+=RMSarr[i]
	#~ 
	#~ s = pev.pEvtBgoRec().GetRMS_r()
	
	RMS2 = calcRMS(pev,position,Edep)
	s = 0
	for i in xrange(14): s+= np.sqrt(RMS2[i])
	
	return s
	
	
def computeFLAST(pev):
	
	return (pev.pEvtBgoRec().GetELayer(13) / pev.pEvtBgoRec().GetTotalEnergy())
	
	
def buildXTRL(pev,position,Edep):
	
	Flast = np.sum(Edep[13]) / np.sum(Edep)
	SumRMS = computeSumRMS(pev,position,Edep)
	
	XTRL = Flast * (SumRMS**4) / 8e+6
	
	return XTRL, Flast, SumRMS
	
def cuts(pev,Edep):
	
	Etot = np.sum(Edep)
	
	if Etot < 1e+3 : return False
	
	#if Etot < 5e+5 or Etot > 1e+6 : return False

	if not all( [ np.sum(Edep[i]) > 0.1 for i in range(14)] ) : return False
	
	f = np.sum(Edep[13]) / np.sum(Edep)
	if f > 1.0 or f < 1e-5 : return False
	
	#~ if computeSumRMS(pev) > 3e+5 : return False
	
	return True
	
	
def run():
	
	from ROOT import gSystem
	gSystem.Load("libDmpEvent.so")
	import ROOT
	
	with open('/dampe/data4/users/ddroz/getXTRL/allElectrons.txt','r') as f:
		allElectrons = []
		for line in f:
			allElectrons.append( line.replace('\n','') )
	with open('/dampe/data4/users/ddroz/getXTRL/allProtons.txt','r') as g:
		allProtons = []
		for lines in g:
			allProtons.append( lines.replace('\n','') )
			
	chain_e = ROOT.DmpChain("CollectionTree")
	for f in allElectrons:
		chain_e.Add(f)
	
	nrofe = chain_e.GetEntries()
	if not nrofe: raise Exception("No electrons in DmpChain!")
	
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
		
		Edep = getEdep(pev)
		
		if not cuts(pev,Edep): continue
		x,f,r = buildXTRL(pev,position,Edep)
		xtrl_e.append( x )
		flast_e.append( f )
		rms_e.append( r )
		energy_e.append( pev.pEvtBgoRec().GetElectronEcor() )
		
	chain_e.Terminate()
	
	del chain_e
	
	chain_p = ROOT.DmpChain("CollectionTree")
	for g in allProtons:
		chain_p.Add(g)
	nrofp = chain_p.GetEntries()
	if not nrofp: raise Exception("No protons in DmpChain!")
		
	for np in xrange(nrofp):
		ppev = chain_p.GetDmpEvent(np)
		
		EdepP = getEdep(ppev)
		
		if not cuts(ppev,EdepP): continue
			
		x,f,r = buildXTRL(ppev,position,EdepP)
		xtrl_p.append( x )
		flast_p.append( f )
		rms_p.append( r )
		energy_p.append( ppev.pEvtBgoRec().GetElectronEcor() )
		
	chain_p.Terminate()
	
	## !!!
	returns = [xtrl_e , xtrl_p, flast_e, flast_p, rms_e, rms_p, energy_e, energy_p,official_rms_e,official_rms_p]
	## !!!
		
	with open("XTRL.pick",'w') as h:
		pickle.dump(returns,h)
		
	print "Selected:"
	print len(energy_e), " electrons"
	print len(energy_p), " protons"
		
	return returns
	


if __name__ == '__main__':
	
	if os.path.isfile('XTRL.pick'):
		with open('XTRL.pick','r') as f:
			aba = pickle.load(f)
			xe, xp, fe, fp, re, rp, ee, ep, of_rms_e, of_rms_p = aba
	else:
		xe, xp, fe, fp, re, rp, ee, ep, of_rms_e, of_rms_p = run()
	
	########
	## Graphs !
	########
	
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	
	
	##
	# Histogram of XTR distribution
	##
	
	ma = max(a)
	mb = max(b)
	m = max([ma,mb])
	
	binlist = [ x*float(m)/100. for x in range(101) ]
	
	fig1 = plt.figure()
	plt.hist(xe,bins=binlist,histtype='step',label='e',color='green')
	plt.hist(xp,bins=binlist,histtype='step',label='p',color='red',ls='dashed')
	plt.yscale('log')
	#~ plt.xscale('log')
	plt.xlabel('XTR')
	plt.grid(True)
	plt.legend(loc='best')
	plt.savefig('XTRL')
	plt.close(fig1)
	
	fig2 = plt.figure()
	plt.hist(np.log10(xe),bins=binlist,histtype='step',label='e',color='green')
	plt.hist(np.log10(xp),bins=binlist,histtype='step',label='p',color='red',ls='dashed')
	plt.yscale('log')
	plt.xlabel('log10(XTR)')
	plt.grid(True)
	plt.legend(loc='best')
	plt.savefig('XTRL_log')
	plt.close(fig2)
	
	
	##
	# Histogram of SumRMS distribution
	##
	
	fig2b = plt.figure()
	plt.hist(re,100,histtype='step',label='e',normed=True)
	plt.hist(rp,100,histtype='step',label='p',normed=True)
	plt.xlabel('Sum RMS [mm]')
	plt.ylabel('Fraction of events')
	plt.ylim((0,0.002))
	plt.legend(loc='best')
	plt.savefig('sumRMS')
	plt.close(fig2b)
	
	fig2bb = plt.figure()
	plt.hist(re,100,histtype='step',label='hand RMS elec',normed=False)
	plt.hist(of_rms_e,100,histtype='step',label='official RMS elec',normed=False)
	plt.xlabel('Sum RMS [mm]')
	plt.legend(loc='best')
	plt.savefig('sumRMS_offVShand_elec')
	plt.close(fig2bb)
	
	##
	# 2D plot of SumRMS vs FLast
	##
	
	fig1b = plt.figure()
	plt.hist2d(re+rp,fe+fp,bins=120,norm=matplotlib.colors.LogNorm())
	plt.colorbar()
	#~ plt.xlim((200,1200))
	plt.yscale('log')
	plt.xlabel('Sum RMS [mm]')
	plt.ylabel('FLast')
	plt.savefig('2d')
	plt.close(fig1b)
	
	fig3 = plt.figure()
	plt.hist2d(re,fe,bins=120,norm=matplotlib.colors.LogNorm())
	#~ plt.xlim((200,1200))
	plt.colorbar()
	plt.yscale('log')
	plt.xlabel('Sum RMS [mm]')
	plt.ylabel('FLast')
	plt.title('Electrons')
	plt.savefig('2d_electrons')
	plt.close(fig3)
	
	fig4 = plt.figure()
	plt.hist2d(rp,fp,bins=120,norm=matplotlib.colors.LogNorm())
	#~ plt.xlim((200,1200))
	plt.colorbar()
	plt.yscale('log')
	plt.xlabel('Sum RMS [mm]')
	plt.ylabel('FLast')
	plt.title('Protons')
	plt.savefig('2d_protons')
	plt.close(fig4)
	
	
	##
	# 2D Histograms of XTR vs Energy
	##
	
	eeG = [x / 1000. for x in ee]
	epG = [x / 1000. for x in ep]
	
	fig1c = plt.figure()
	plt.hist2d(eeG+epG,xe+xp,bins=200,norm=matplotlib.colors.LogNorm())
	plt.xlabel('Corrected BGO energy [GeV]')
	plt.ylabel('XTR [mm^4]')
	#~ plt.ylim((0,140))
	plt.xscale('log')
	#~ plt.yscale('log')
	plt.tight_layout()
	plt.colorbar()
	plt.savefig('2d_XTR')
	plt.close(fig1c)
	
	fig2c = plt.figure()
	plt.hist2d(eeG,xe,bins=200,norm=matplotlib.colors.LogNorm())
	#~ plt.hist2d(eeg,[x for x in xe if x < 140],bins=120,norm=matplotlib.colors.LogNorm())
	plt.xlabel('Corrected BGO energy [GeV]')
	plt.ylabel('XTR [mm^4]')
	plt.xscale('log')
	#~ plt.yscale('log')
	plt.tight_layout()
	plt.ylim((0,5))
	plt.title('Electrons')
	plt.colorbar()
	plt.savefig('2d_XTR_electrons')
	plt.close(fig2c)
	
	fig3c = plt.figure()
	plt.hist2d(epG,xp,bins=200,norm=matplotlib.colors.LogNorm())
	#~ plt.hist2d(epG,[x for x in xp if x < 140],bins=120,norm=matplotlib.colors.LogNorm())
	plt.xlabel('Corrected BGO energy [GeV]')
	plt.ylabel('XTR [mm^4]')
	plt.ylim((5.5e-2,5))
	plt.xscale('log')
	#~ plt.yscale('log')
	plt.tight_layout()
	plt.title('Protons')
	plt.colorbar()
	plt.savefig('2d_XTR_protons')
	plt.close(fig3c)
