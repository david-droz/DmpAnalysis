'''

Code that takes as input one of Xin's MC ntuples, and returns numpy files for the machine learning

'''

from __future__ import print_function, division, absolute_import
from ROOT import TTree, TFile, TBranch
import numpy as np
import sys
import os
import math

def computeWeight(TT,f):
	'''
	From Xin's code
	'''
	
	ekin = TT.tt_ekin
	ekin_GeV = TT.tt_ekin * 0.001
	w = ekin / 100000.	# normalize to 100 GeV
	
	if "Electron" in f:
		evtPoid = 1/(w*w)
		if ekin_GeV < 100 :
			nElectronHighE = 133740000
			nElectronLowE  = 499600000
			return evtPoid * (nElectronHighE / nElectronLowE)
		return evtPoid
	
	elif "Proton" in f:
		evtPoid = w**(-1.7)
		nProtonHighE = 781390000
		nProtonVHE = 2*124698750
		nProtonLowE = 492600000
		if ekin_GeV < 100:
			return evtPoid * (nProtonHighE/nProtonLowE)
		elif ekin_GeV > 10000 :		# 10 TeV
			return evtPoid * nProtonHighE/nProtonVHE
		return evtPoid
		
	else:
		raise Exception("Particle non identified from name: ", f)


def main(f):
	
	'''
	List of variables:
		0 - 13 : Energy in BGO layer i
		14 - 27 : RMS of energy deposited in layer i
		28 - 41 : Number of hits in layer i
		
		42 : longitudinal RMS ( DmpEvtBgoRec::GetRMS_l )
		43 : radial RMS ( DmpEvtBgoRec::GetRMS_r )
		44 : total BGO energy 
		45 : total BGO hits
		46 : theta angle of BGO trajectory
		----
		47 : True energy (EKin). Set to 0 if missing (i.e. flight data)
		48 : Event weight (weight according to energy spectrum)
		49 : XTRL
		50 : Particle ID (0 for proton, 1 for electron)
	'''

	TF = TFile(f,'READ')
	TT = TF.Get("DmlNtup")
	predArray = np.zeros( (int(TT.GetEntries()), 51) )
	
	for n in range(0,TT.GetEntries()):
		pev = TT.GetEntry(n)
		
		erec = TT.tt_bgoTotalE_GeV * 1000		# DNN trained in MeV
		
		for frac_i in range(0,14):
			predArray[n,frac_i] = getattr(TT,"tt_F"+str(frac_i)) * erec	# Energy fraction goes like tt_F0, tt_F1, ...
			#~ predArray[n,frac_i] = getattr(TT,"tt_F"+str(frac_i))
		for rms_i in range(0,14):
			predArray[n,rms_i+14] = getattr(TT,"tt_Rms"+str(rms_i))
		for hits_i in range(0,14):
			predArray[n,hits_i+28] = ord(getattr(TT,"tt_nBarLayer"+str(hits_i)))
					
		predArray[n,42] = TT.tt_Rmsl
		predArray[n,43] = TT.tt_Rmsr
		predArray[n,44] = erec
		predArray[n,45] = TT.tt_nBgoHits
		
		XZ = TT.tt_bgoRecSlopeX
		YZ = TT.tt_bgoRecSlopeY
		tgZ = math.atan(np.sqrt( (XZ*XZ) + (YZ*YZ) ) )
		
		predArray[n,46] = tgZ*180./math.pi
		predArray[n,47] = TT.tt_ekin
		predArray[n,48] = computeWeight(TT,f)
		predArray[n,49] = TT.tt_Xtrl
		
		if "Electron" in f:
			predArray[n,50] = 1
		
	# END FOR
	
	if not os.path.isdir('outfiles'): os.mkdir('outfiles')
	
	outname = 'outfiles/' + os.path.splitext(os.path.basename(f))[0] + '.npy'
	
	np.save(outname,predArray)
	
	TF.Close()
	

if __name__ == '__main__' :
	
	main(sys.argv[1])
