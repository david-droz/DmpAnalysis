'''

Look at STK cluster width and see if there is value to them

'''

from __future__ import division

import numpy as np
import math
import os
import pickle
from ROOT import gSystem
gSystem.Load("libDmpEvent.so")
import ROOT

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from getXTRL import calcRMS, CalcPosition, computeCOG, getEdep, cuts


def run(fname,pfile):
	
	with open(fname,'r') as f:
		allF = []
		for line in f:
			allF.append( line.replace('\n','') )
			
	chain = ROOT.DmpChain("CollectionTree")
	for f in allF:
		chain.Add(f)
	
	nevents = chain.GetEntries()
	position = CalcPosition()
	
	hand_RMS = np.zeros((nevents,14))
	off_RMS = np.zeros((nevents,14))
	hand_COG = np.zeros((nevents,14))
	off_COG = np.zeros((nevents,14))
	
	for i in range(nevents):
		
		pev = chain.GetDmpEvent(i)
		Edep = getEdep(pev)
		if not cuts(pev,Edep): continue
		
		hand_RMS[i] = calcRMS(pev,position,Edep)
		hand_COG[i] = computeCOG(position,Edep)
		
		orms = pev.pEvtBgoRec().GetRMS2()
		for j in range(14):
			off_RMS[i,j] = orms[j]
		ocog = pev.pEvtBgoRec().GetCoG()
		for j in range(14):
			off_COG[i,j] = ocog[j]
		
		
	returns = [	hand_RMS,off_RMS,hand_COG,off_COG ]
		
	with open(pfile,'w') as f:
		pickle.dump(returns,f)
		
	return returns
	
	
	
	
def makePlots(hand_RMS,off_RMS,hand_CoG,off_CoG):		# 14-elements numpy array
	
	if not os.path.isdir('images'): os.mkdir('images')
	
	for i in range(14):
		
		fig1 = plt.figure()
		plt.hist(hand_RMS[:,i],100,histtype='step',label='hand')
		plt.hist(off_RMS[:,i],100,histtype='step',label='official',ls='dashed',color='red')
		plt.legend(loc='best')
		plt.yscale('log')
		plt.xlabel('RMS layer ' +str(i))
		plt.savefig('images/rms_'+str(i))
		plt.close(fig1)
		
		fig2 = plt.figure()
		plt.hist(hand_CoG[:,i],100,histtype='step',label='hand')
		plt.hist(off_CoG[:,i],100,histtype='step',label='official',ls='dashed',color='red')
		plt.legend(loc='best')
		plt.yscale('log')
		plt.xlabel('CoG layer ' +str(i))
		plt.savefig('images/cog_'+str(i))
		plt.close(fig2)
	
if __name__ == '__main__' :
	
	fname = '/dampe/data4/users/ddroz/getXTRL/allElectrons.txt'
	#fname = '/dampe/data4/users/ddroz/getXTRL/allProtons.txt'
	
	if "Elec" in fname: pfile = 'cogrms_e.pick'
	else: pfile = 'cogrms_p.pick'
	
	if os.path.isfile(pfile):
		with open(pfile,'r') as f:
			h_RMS, o_RMS, h_COG, o_COG = pickle.load(f)
	else:
		h_RMS, o_RMS, h_COG, o_COG = run(fname,pfile)

	makePlots(h_RMS, o_RMS, h_COG, o_COG)
