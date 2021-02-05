'''

Code that takes as input one of Xin's MC ntuples, and returns numpy files for the machine learning

'''

from __future__ import print_function, division, absolute_import
from ROOT import TTree, TFile, TBranch
import numpy as np
import sys
import os
import math

def selection(TT):
	
	#~ selBits = TT.tt_selection_bits
	#~ cuts = [1,2]			#  Cut #1 is HE trigger, cut #2 is fiducial cut
	#~ for i in cuts:
		#~ if not (selBits & (0x1 << i)) : return False			# Bit mask shifting black magic fuckery
		
	##
		
	inSaa = not (TT.tt_selection_bits & (0x1 << 0))		# Cut #0 is SAA, cut #27 is "allCases" (full preselection)
	pass_all_cases = (TT.tt_decision_bits & (0x1 << 27))	# difference between selection_bits and decision_bits ?
	
	if inSaa or not pass_all_cases : return False
	
	return True

def hasTrack(TT):
	
	# Events that have a BGO matched track with at least 4-hits:
	passTrackMatch   =  TT.tt_selection_bits & (0x1 << 4) # SELECTION_BIT_trk
	#Events that failed the passTrackMatch cut, but have a BGO matched track with only 3-hits:
	passTrackMatch_3clu  = TT.tt_decision_bits & (0x1 << 11) # DECISION_BIT_trk_3clu
	
	return bool(passTrackMatch or passTrackMatch_3clu)

	
def getBarPosition(gid):
	lay = ((gid>>11) &0x000f)
	bar = ((gid>>6) &0x001f)
	return lay, bar

def buildImage(TT,applyThreshold):
	barID_vec = TT.tt_BgoHit_GlobalBarID	# C++ vectors, can use them as Python lists
	energyHit_vec = TT.tt_BgoHit_Energy
	thatHitArray = np.zeros((14,22),dtype='float32')
	for ihit in range(len(barID_vec)):
		lay,bar = getBarPosition(barID_vec[ihit])
		eh = energyHit_vec[ihit]
		if applyThreshold and eh < 10 :
			thatHitArray[lay,bar] = 0
		else:
			thatHitArray[lay,bar] = eh
	
	return thatHitArray
	
	
def buildValues(TT,f):
	'''
	00-13 : BGO energy fraction per layer
	14-27 : BGO energy per layer
	28-41 : log10 of BGO energy fraction per layer
	42-55 : RMS of energy per BGO layer
	   56 : Total BGO energy (corrected) in GeV
	   57 : Number of BGO hits
	   58 : SlopeX
	   59 : SlopeY
	   60 : Angle of BGO trajectory (??)
	   61 : XTRL / Zeta
	   62 : Has track
	   63 : True Energy (set to 0 if not MC)
	   64 : Event weight (set to 1 if not MC)
	   65 : Label 0/1 proton/electron
	'''
	
	values = []
	energyF = [0 for i in range(14)]
	erec = TT.tt_bgoTotalEcorr_GeV * 1000
	for frac_i in range(0,14):
		energyF[frac_i] = getattr(TT,"tt_F"+str(frac_i))	# Energy fraction goes like tt_F0, tt_F1, ...
	
	values += energyF
	values += [ x * erec for x in energyF ]
	for x in energyF:
		if x :
			values.append( np.log10(x) )
		else:
			values.append(0)
	
	for rms_i in range(0,14):
			values.append( getattr(TT,"tt_Rms"+str(rms_i)) )
			
	values.append(erec)
	values.append(TT.tt_nBgoHits)
	
	XZ = TT.tt_bgoRecSlopeX
	YZ = TT.tt_bgoRecSlopeY
	tgZ = math.atan(np.sqrt( (XZ*XZ) + (YZ*YZ) ) )
	
	values.append(XZ)
	values.append(YZ)
	values.append( tgZ*180./math.pi )
	
	values.append(TT.tt_Xtrl)
	
	if hasTrack(TT):
		values.append(1)
	else:
		values.append(0)
		
	values.append(TT.tt_ekin)
	values.append(TT.tt_evtPoid)
	
	if "lectron" in f :
		values.append(1)
	#elif "Proton" in f or "He" in f:
	else :
		values.append(0)
	
	return values

def main(f,applyThreshold=False):
	
	outdir = os.path.splitext(os.path.basename(f))[0]
	outname = 'outfiles/' + outdir + '/' + outdir + '.npy'
	if not os.path.isdir('outfiles'): os.mkdir('outfiles')
	if not os.path.isdir('outfiles/'+outdir): os.mkdir('outfiles/'+outdir)

	TF = TFile(f,'READ')
	TT = TF.Get("DmlNtup")
	
	
	predArray = np.zeros( (10000,14,22),dtype='float32' )	# Save arrays by chunks of 10k events
	valuesArray = []
	arrayNumIndex = 0
	arrayRowIndex = 0
	
	
	for n in range(0,TT.GetEntries()):
		pev = TT.GetEntry(n)
		
		if not selection(TT): continue
		
		
		'''
		--- IMAGE PART ---
		'''
			
		predArray[arrayRowIndex] = buildImage(TT,applyThreshold)
		arrayRowIndex += 1
		
		'''
		--- VALUES PART ---
		'''
		
		valuesArray.append( buildValues(TT,f) )	# List-of-lists dynamic approach so I can easily change nr of variables
		
		
		'''
		--- SAVE CHUNK ---
		'''
		
		if arrayRowIndex >= predArray.shape[0] :
			np.save(outname.replace('.npy','_image_%d.npy' % arrayNumIndex),predArray.astype('float32'))
			np.save(outname.replace('.npy','_%d.npy' % arrayNumIndex), np.array(valuesArray).astype('float32') )
			arrayNumIndex += 1
			arrayRowIndex = 0
			del predArray
			del valuesArray
			predArray = np.zeros( (10000,14,22),dtype='float32' )
			valuesArray = []
		
	# END FOR
	
	np.save(outname.replace('.npy','_image_%d.npy' % arrayNumIndex),predArray[:arrayRowIndex].astype('float32'))
	np.save(outname.replace('.npy','_%d.npy' % arrayNumIndex), np.array(valuesArray).astype('float32') )
	
	TF.Close()
	

if __name__ == '__main__' :
	
	try:
		threshold = bool(int(sys.argv[2]))
	except IndexError:
		threshold = False
	
	try:
		main(sys.argv[1],threshold)
	except AttributeError :
		print("--- ERROR IN : ", sys.argv[1])
		raise

	BSN = os.path.splitext(os.path.basename(sys.argv[1]))[0]

	with open('outfiles/'+BSN+'/DONE','w') as f:
		f.write(' \n')
