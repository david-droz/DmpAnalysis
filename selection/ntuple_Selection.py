'''

Code that takes as input one of Xin's MC ntuples, and returns numpy files for the machine learning

'''

from __future__ import print_function, division, absolute_import
from ROOT import TTree, TFile, TBranch
import numpy as np
import sys
import os
import math

DO_CORRECTION = True  # RMS CORR
CORRECT_XTRL = False

def computeWeight(TT,f):
	'''
	From Xin - UPDATED 2020-09-21
	OBSOLETE 2021-02-18, use tt.evt_poid instead
	Weight to E^-1
	'''
	
	ekin = TT.tt_ekin
	ekin_GeV = TT.tt_ekin * 0.001
	
	
	# Electron weight
	nElectronLowE  = 542400000  # allElectron-v6r0p10_1GeV_100GeV_FTFP_HP
	nElectronHighE_1TeV = 200540000 # allElectron-v6r0p10_100GeV_1TeV_FTFP_HP
	nElectronHighE_10TeV = 97617600 # allElectron-v6r0p10_1TeV_10TeV_FTFP
	nElectronVHE   = 18097000  # allElectron-v6r0p10_10TeV_20TeV_FTFP
		
	if "allElectron-v6r0p10_1GeV_100GeV_FTFP_HP" in f :
		return 2*nElectronHighE_1TeV/nElectronLowE
	elif "allElectron-v6r0p10_1TeV_10TeV_FTFP" in f : 
		return nElectronHighE_1TeV/nElectronHighE_10TeV
	elif "allElectron-v6r0p10_10TeV_20TeV_FTFP" in f :
		return 0.301029995663981*nElectronHighE_1TeV/nElectronVHE  # (the number is just log10(2)).
	
	
	# Proton Weight
	nProtonLowE_1_10GeV   = 2068200000/2.
	nProtonLowE_10_100GeV = 2068200000/2. + 1024280000
	nProtonHighE_low  = 535520000 + 525080000 + 540990000
	nProtonHighE_high =  120650400 + 132206400
	#nProtonVHE        = 17802450+13981200+20181825+14001000
	nProtonVHE 	= 86583900
	
	#if pro_lowE_1_10GeV :
	if "Proton" in f:
		if ekin_GeV < 10 :
			return nProtonHighE_low/nProtonLowE_1_10GeV
		# ~ elif ifpro_lowE_10_100GeV :
		elif ekin_GeV > 10 and ekin_GeV < 100 :
			return nProtonHighE_low/nProtonLowE_10_100GeV
		# ~ else ifpro_highE_high :
		elif ekin_GeV > 1000 and ekin_GeV < 1e+4 :
			return nProtonHighE_low/nProtonHighE_high
		# ~ else ifpro_VHE :
		elif ekin_GeV > 1e+4 :
			return nProtonHighE_low/nProtonVHE
	
	
	
	print("Sample not detected, defaulting to 1.0 :", f)
	return 1.
		
		
def correctRMSProton(il,bgoTotalE_GeV,rmsLay_il):
	'''
	Adapted from Xin's C++ code
	'''
	
	rmsCorr = 0 #to be subtracted
	x = bgoTotalE_GeV
	if il==0 :
		if x<150 :     rmsCorr = 2*(np.log10(x)-np.log10(25))/(np.log10(150)-np.log10(25)) + 5.0
		elif x<6000 :  rmsCorr = 7
		elif x<8000 :  rmsCorr = 8
		elif x<10000 : rmsCorr = 6
		else :         rmsCorr = 8
	
	elif il==1 : 
		if x<350 :        rmsCorr = 4
		else :            rmsCorr = 3

	elif il==2 : 
		if x<60 :      rmsCorr = 3
		elif x<290 :   rmsCorr = 2
		elif x<10000 : rmsCorr = 1 
		else :         rmsCorr = 0
	
	elif il==3 :
		if x<35 :         rmsCorr = 3   
		elif x<100 :      rmsCorr = 2 
		elif x<1000 :     rmsCorr = 1 
		else :             rmsCorr = 0
	
	elif il==4 : 
		if x<80 :         rmsCorr = 2
		elif x<250 :      rmsCorr = 1
		else :            rmsCorr = 0
	
	elif il==5 :
		if x<60 :         rmsCorr = 1.5
		elif x<85 :       rmsCorr = 2
		elif x<220 :      rmsCorr = 1 
		else :            rmsCorr = 0 
	
	elif il==6 :
		if x<125 :        rmsCorr = 2    
		elif x<380 :      rmsCorr = 1  
		else :            rmsCorr = 0 

	elif il==7 :
		if x<63 :         rmsCorr = 1.5    
		elif x<300 :   rmsCorr = 2  
		elif x<600 :   rmsCorr = 1  
		else :            rmsCorr = 0  
	
	elif il==8 :
		if x<250 :        rmsCorr = 3    
		elif x<550 :   rmsCorr = 2  
		elif x<1500 :  rmsCorr = 1  
		else :           rmsCorr = 0  
	
	elif il==9 :  
		if x<45 :          rmsCorr = 3    
		elif x<62 :     rmsCorr = 4.5    
		elif x<95 :     rmsCorr = -3*(np.log10(x)-np.log10(62))/(np.log10(95)-np.log10(62)) + 6  
		elif x<250 :    rmsCorr = 4    
		elif x<350 :    rmsCorr = 3    
		elif x<900 :    rmsCorr = -2*(np.log10(x)-np.log10(350))/(np.log10(900)-np.log10(350)) + 3.5  
		elif x<1500 :   rmsCorr = 1.5  
		elif x<10000 :  rmsCorr = 1  
		else :              rmsCorr = 0  
	 
	elif il==10 :  
		if x<42 :        rmsCorr = 5.5    
		elif x<62 :   rmsCorr = 4  
		elif x<250 :  rmsCorr = 5  
		elif x<570 :  rmsCorr = 4  
		elif x<2000 : rmsCorr = -2*(np.log10(x)-np.log10(700))/(np.log10(2000)-np.log10(700)) + 3  
		else :         rmsCorr = 1  
	 
	elif il==11 :
		if x<100 :        rmsCorr = 5.5    
		elif x<1700 :  rmsCorr = -5*(np.log10(x)-np.log10(100))/(np.log10(1700)-np.log10(100)) + 7  
		elif x<3500 :  rmsCorr = 2  
		else :            rmsCorr = 1  
	 
	elif il==12 :  
		if x<450 :       rmsCorr = 6 
		elif x<1000 : rmsCorr = -4*(np.log10(x)-np.log10(450))/(np.log10(3500)-np.log10(450)) + 5  
		elif x<3500 : rmsCorr = -4*(np.log10(x)-np.log10(450))/(np.log10(3500)-np.log10(450)) + 5.5  
		else :           rmsCorr = 1  
	 
	elif il==13 :  
		if x<36 :        rmsCorr = 7    
		elif x<55 :   rmsCorr = -2*(np.log10(x)-np.log10(36))/(np.log10(55)-np.log10(36)) + 9  
		elif x<83 :   rmsCorr = 11  
		elif x<96 :   rmsCorr = 7  
		elif x<110 :  rmsCorr = 9  
		elif x<220 :  rmsCorr = 8  
		elif x<3500 : rmsCorr = -4.5*(np.log10(x)-np.log10(220))/(np.log10(3500)-np.log10(220)) + 7   
		else :           rmsCorr = 2  
	
	newRms = rmsLay_il - rmsCorr
	
	if newRms < 0 : return 0
	else: return newRms
	 

def correctRMSElectron(il,bgoTotalE_GeV,rmsLay_il):
	'''
	Adaped from Xin's C++ code
	'''
	
	rmsCorr = 0 #to be subtracted
	x = bgoTotalE_GeV
	
	# from model 511
	
	if il==0 : 
		if x<42 :         rmsCorr = 2.5
		elif x<1300 :  rmsCorr = 3
		elif x<2000 :  rmsCorr = 4
		else :             rmsCorr = 3
	
	elif il==1 : 
		if x<40 :         rmsCorr = 1.5
		elif x<250 :   rmsCorr = 1
		elif x<1900 :  rmsCorr = 0.5
		else :             rmsCorr = 0
	
	elif il==2 : 
		if x<60 :         rmsCorr = 1
		elif x<150 :   rmsCorr = 0.5
		else :             rmsCorr = 0
	
	elif il==3 : 
		if x<80 :         rmsCorr = 1
		elif x<180 :   rmsCorr = 0.5
		else :             rmsCorr = 0
	
	elif il==4 : 
		if x<70 :         rmsCorr = 1
		elif x<150 :   rmsCorr = 0.5
		else :             rmsCorr = 0
	
	elif il==5 : 
		if x<220 :        rmsCorr = 0.5
		else :             rmsCorr = 0
	
	elif il==6 : 
		if x<300 :        rmsCorr = 1
		elif x<1100 :  rmsCorr = 0.5
		else :             rmsCorr = 1
	
	elif il==7 : 
		if x<38 :         rmsCorr = -0.5
		elif x<70 :    rmsCorr = 0
		elif x<660 :   rmsCorr = 1
		elif x<760 :   rmsCorr = 0
		elif x<1100 :  rmsCorr = 0.5
		elif x<1900 :  rmsCorr = 1
		else :             rmsCorr = 1.5
	
	elif il==8 : 
		if x<80 :         rmsCorr = 0.5
		elif x<180 :   rmsCorr = 1
		elif x<600 :   rmsCorr = 1.5
		elif x<1900 :  rmsCorr = 1
		else :             rmsCorr = 0.5
	
	elif il==9 : 
		if x<80 :         rmsCorr = 0.5
		elif x<190 :   rmsCorr = 1
		elif x<440 :   rmsCorr = 0.5*(np.log10(x)-np.log10(190))/(np.log10(440)-np.log10(190)) + 1.5
		elif x<800 :   rmsCorr = 2
		elif x<1500 :  rmsCorr = 1.5
		else :             rmsCorr = 2
	
	elif il==10 : 
		if x<380 :        rmsCorr = 1
		elif x<580 :   rmsCorr = 0.5*(np.log10(x)-np.log10(380))/(np.log10(580)-np.log10(380)) + 1.5
		elif x<800 :   rmsCorr = -1.5*(np.log10(x)-np.log10(580))/(np.log10(800)-np.log10(580)) + 3
		elif x<1300 :  rmsCorr = 2*(np.log10(x)-np.log10(800))/(np.log10(1300)-np.log10(800)) + 1.5
		elif x<1900 :  rmsCorr = -1.5*(np.log10(x)-np.log10(1300))/(np.log10(1900)-np.log10(1300)) + 3.5
		else :             rmsCorr = 0.5
	
	elif il==11 : 
		if x<35 :         rmsCorr = 0.5
		elif x<65 :    rmsCorr = 1
		elif x<300 :   rmsCorr = 1.5
		elif x<680 :   rmsCorr = 1.5*(np.log10(x)-np.log10(300))/(np.log10(680)-np.log10(300)) + 1
		elif x<14000 : rmsCorr = 2.5
	
	elif il==12 : 
		if x<35 :         rmsCorr = 0.5
		elif x<500 :   rmsCorr = 1
		elif x<1700 :  rmsCorr = 1.5
		else :             rmsCorr = 3.5
	
	elif il==13 : 
		if x<35 :         rmsCorr = 0.5
		elif x<83 :    rmsCorr = 1
		elif x<220 :   rmsCorr = 2
		elif x<500 :   rmsCorr = -0.5*(np.log10(x)-np.log10(220))/(np.log10(500)-np.log10(220)) + 2
		elif x<680 :   rmsCorr = 1.5*(np.log10(x)-np.log10(500))/(np.log10(680)-np.log10(500)) + 1.5
		elif x<1500 :  rmsCorr = -1.5*(np.log10(x)-np.log10(680))/(np.log10(1500)-np.log10(680)) + 3
		elif x<1900 :  rmsCorr = 3.5
		else :             rmsCorr = 1
	
	
	# ~ if  !(il >= 9 &&  rmsLayer[il]<14)  : 
		# ~ rmsLayerCorr[il] -= rmsCorr
		
	if not (il >= 9 and rmsLay_il < 14) :
		newRms = rmsLay_il - rmsCorr
	else:
		newRms = rmsLay_il
	
	if newRms < 0 : newRms = 0
	
	return newRms	
	

def correctFlast(bgoTotalEcorr_GeV,pType,Flast):
	
	if pType in ['e','electron','Electron'] or 'lectron' in pType:
		
		if 	 bgoTotalEcorr_GeV<190 :      nbinShfitFlast = 0;
		elif bgoTotalEcorr_GeV<500 :  	nbinShfitFlast = 1;
		elif bgoTotalEcorr_GeV<575 :  	nbinShfitFlast = 2;
		elif bgoTotalEcorr_GeV<870 :  	nbinShfitFlast = 1;
		elif bgoTotalEcorr_GeV<1000 : 	nbinShfitFlast = 2;
		elif bgoTotalEcorr_GeV<1150 : 	nbinShfitFlast = 1;
		elif bgoTotalEcorr_GeV<1300 : 	nbinShfitFlast = 2;
		elif bgoTotalEcorr_GeV<1500 : 	nbinShfitFlast = 1;
		else                        :     nbinShfitFlast = 0;
		
		return Flast * (10**(nbinShfitFlast*0.0430103) )
		
	elif pType in ['p','proton','Proton'] or 'roton' in pType:
		
		if bgoTotalEcorr_GeV<41 :         	nbinShfitFlast = 2;
		elif bgoTotalEcorr_GeV<63 :   	 nbinShfitFlast = 3;
		elif bgoTotalEcorr_GeV<660 :  	 nbinShfitFlast = 4;
		elif bgoTotalEcorr_GeV<2630 : 	 nbinShfitFlast = 3;
		elif bgoTotalEcorr_GeV<6000 : 	 nbinShfitFlast = 2;
		elif bgoTotalEcorr_GeV<10000 :	 nbinShfitFlast = 3;
		else :                             nbinShfitFlast = 2;
		return Flast* (10**(nbinShfitFlast*0.0430103) )
		
		
	else:
		return Flast



def selection(TT):
	
	#~ selBits = TT.tt_selection_bits
	#~ cuts = [1,2]			#  Cut #1 is HE trigger, cut #2 is fiducial cut
	#~ for i in cuts:
		#~ if not (selBits & (0x1 << i)) : return False			# Bit mask shifting black magic fuckery
		
	##
	
	# COMMENTED OUT 2021-02-18	
	# ~ inSaa = not (TT.tt_selection_bits & (0x1 << 0))		# Cut #0 is SAA, cut #27 is "allCases" (full preselection)
	# ~ pass_all_cases = (TT.tt_decision_bits & (0x1 << 27))	# difference between selection_bits and decision_bits ?
	
	# 2021-02-18 : New selection including final cleaning cuts
	inSaa = not (TT.tt_selection_bits & (0x1 << 0))
	pass_all_cases = (TT.tt_decision_bits1 & (0x1 << 28) )
	
	
	if inSaa or not pass_all_cases : return False
	
	return True


def main(f):
	
	'''
	List of variables:
		0 - 13 : Energy in BGO layer i
		14 - 27 : RMS of energy deposited in layer i
		28 - 41 : Number of hits in layer i
		
		42 : longitudinal RMS ( DmpEvtBgoRec::GetRMS_l )
		43 : radial RMS ( DmpEvtBgoRec::GetRMS_r )
		44 : total BGO energy 
		45 : FLast 
		46 : theta angle of BGO trajectory
		----
		47 : True energy (EKin). Set to 0 if missing (i.e. flight data)
		48 : Event weight (weight according to energy spectrum)
		49 : XTRL
		50 - 63 : Fraction of energy in BGO layer i
		64 - 77 : Log of energy in BGO layer i  (cut at 1 MeV)
		78 : tt_stkEcore1Rm_trk - sum of "energy" of clusters within 1 Rm (Moliere radius, not X0), obtained by the getEnergy() method of the DmpStkSiCluster class. Set to 0 if no matching track
		79 : tt_nStkClu_trk - number of STK clusters within 1 Rm of the Track. Set to 0 if no matching track
		80 : Particle ID (0 for proton, 1 for electron)
	'''
	if "roton" in os.path.basename(f) and DO_CORRECTION: 	# RMS proton correction
		print("Warning! Doing RMS correction")
	if "lectron" in os.path.basename(f) and DO_CORRECTION: 	# RMS electron correction
		print("Warning! Doing RMS correction")

	TF = TFile(f,'READ')
	TT = TF.Get("DmlNtup")
	#~ predArray = np.zeros( (int(TT.GetEntries()), 51) )
	predArray = np.zeros( (int(TT.GetEntries()), 81) , dtype='float32')
	
	foundAnError = 0
	
	for n in range(0,TT.GetEntries()):
		pev = TT.GetEntry(n)
		
		if not selection(TT): continue
		
		erec = TT.tt_bgoTotalEcorr_GeV * 1000		# DNN trained in MeV
		
		for frac_i in range(0,14):
			predArray[n,frac_i] = getattr(TT,"tt_F"+str(frac_i)) * erec	# Energy fraction goes like tt_F0, tt_F1, ...
			predArray[n,frac_i+50] = getattr(TT,"tt_F"+str(frac_i))

			tmpfrac = getattr(TT,"tt_F"+str(frac_i)) * erec
			if tmpfrac > 1 :  # 1 MeV
				predArray[n,frac_i+50+14] = np.log10(tmpfrac)
		for rms_i in range(0,14):
			# ~ predArray[n,rms_i+14] = getattr(TT,"tt_Rms"+str(rms_i))
			myRms = getattr(TT,"tt_Rms"+str(rms_i))


			if "roton" in os.path.basename(f) and DO_CORRECTION: 	# RMS proton correction
				correctedRMS = correctRMSProton(rms_i,TT.tt_bgoTotalEcorr_GeV,myRms)
				predArray[n,rms_i+14] = correctedRMS
				
			elif "lectron" in os.path.basename(f) and DO_CORRECTION:
				correctedRMS = correctRMSElectron(rms_i,TT.tt_bgoTotalEcorr_GeV,myRms)
				predArray[n,rms_i+14] = correctedRMS
								
			else:
				predArray[n,rms_i+14] = myRms
			
		#for hits_i in range(0,14):
		#	try:
		#		predArray[n,hits_i+28] = ord(getattr(TT,"tt_nBarLayer"+str(hits_i)))
		#	except AttributeError :
		#		if not foundAnError : 
		#			print("--- ERROR IN FILE: ", f)
		#			foundAnError += 1
		#		predArray[n,hits_i+28] = 0		# Not using BGO hits anyways
		
		
		FRACarr = predArray[n,50:64]
		FLast = FRACarr[ FRACarr > 0 ][-1]	
		if "roton" in os.path.basename(f) and DO_CORRECTION:	
				FLast = correctFlast(TT.tt_bgoTotalEcorr_GeV,'p',FLast)
		elif "lectron" in os.path.basename(f) and DO_CORRECTION:
				FLast = correctFlast(TT.tt_bgoTotalEcorr_GeV,'e',FLast)
				
				
					
		predArray[n,42] = TT.tt_Rmsl
		predArray[n,43] = TT.tt_Rmsr
		predArray[n,44] = erec
		predArray[n,45] = FLast
		
		XZ = TT.tt_bgoRecSlopeX
		YZ = TT.tt_bgoRecSlopeY
		tgZ = math.atan(np.sqrt( (XZ*XZ) + (YZ*YZ) ) )
		
		predArray[n,46] = tgZ*180./math.pi
		predArray[n,47] = TT.tt_ekin
		# ~ predArray[n,48] = computeWeight(TT,f)
		predArray[n,48] = TT.tt_evtPoid

		if CORRECT_XTRL:
			sumRms = predArray[n,14:28].sum()
			
			correctedXtrl = sumRms*sumRms*sumRms*sumRms * FLast / 8000000.
			predArray[n,49] = correctedXtrl
		else:
			predArray[n,49] = TT.tt_Xtrl

		predArray[n,78] = TT.tt_stkEcore1Rm_trk
		predArray[n,79] = TT.tt_nStkClu_trk
		
		if "Electron" in f:
			predArray[n,-1] = 1
		
	# END FOR
	
	if DO_CORRECTION and not CORRECT_XTRL : outdir = 'outfiles_RMScorrected'
	elif DO_CORRECTION and CORRECT_XTRL : outdir = 'outfiles_XtrlCorr'
	elif not DO_CORRECTION : outdir = 'outfiles_noCorrection'
	
	
	if not os.path.isdir(outdir): os.mkdir(outdir)
	
	outname = outdir + '/' + os.path.splitext(os.path.basename(f))[0] + '.npy'
	
	np.save(outname,predArray[np.any(predArray,axis=1)])
	
	TF.Close()
	

if __name__ == '__main__' :
	
	try:
		main(sys.argv[1])
	except AttributeError :
		print("--- ERROR IN : ", sys.argv[1])
		raise
