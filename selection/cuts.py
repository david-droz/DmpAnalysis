'''

Series of cuts applied as pre-selection. 
Source: Geneva electron analysis support document, SVN 291

'''

import numpy as np

from ROOT import TVector3

def getRMS(bgorec):
	
	NBGOLAYERS  = 14
	NBARSLAYER  = 22
	EDGEBARS    = [0,21]
	BARPITCH    = 27.5
	
	edep = np.zeros((NBGOLAYERS,NBARSLAYER))
	for i in range(NBGOLAYERS):
		for j in range(NBARSLAYER):
			edep[i,j] = bgorec.GetEdep(i,j)
	
	BHET = edep.sum()
	BHXS = [0. for i in xrange(NBGOLAYERS)]
	bhm  = 0.
	SIDE = [False for i in xrange(NBGOLAYERS)]
	
	for i in range(NBGOLAYERS):
		# Find the bar with max energy deposition of a layer and record its number as im
		im = None
		em = 0.0;
		for j in range(NBARSLAYER):
			ebar = edep[i,j]
			if ebar < em : continue 
			em = ebar
			im = j
		
		if not em: continue
		
		if im in EDGEBARS:
			cog = BARPITCH * im   #BHX[i][im]
		else:
			ene = 0.
			cog = 0.
			for  j in [im-1, im, im+1]: 
				ebar = edep[i,j]
				ene += ebar
				cog += BARPITCH * j * ebar
			cog /= ene
			
		posrms   = 0.
		enelayer = 0.
		for j in range(NBARSLAYER):
			ebar = edep[i,j]
			posbar = BARPITCH * j 
			enelayer += ebar
			posrms += ebar * (posbar-cog)*(posbar-cog)
		posrms = np.sqrt( posrms / enelayer)
		BHXS[i] = posrms
	
	del edep
	return BHXS
	

##########################
### SECTION 3.1
##########################

def rMaxELayer(bgorec,cutValue=0.35):
	'''
	Events having more than 35% of the total energy deposited in one layer are rejected
	'''
	
	ELayer_max = 0
	for i in range(14):
		e = bgorec.GetELayer(i)
		if e > ELayer_max: ELayer_max = e
	
	try:
		rMaxELayerTotalE = ELayer_max / bgorec.GetTotalEnergy()
	except ZeroDivisionError:
		return False
	if rMaxELayerTotalE > cutValue: 
		return False
		
	return True
	
def iBarMaxE(event):
	'''
	Events in which the maximum energy bar of layer 1, 2 ,3 are at the
	edge are rejected. (Note: 14 BGO layers are numbered 0-13)
	'''
	
	barNumberMaxEBarLay1_2_3 = [-1 for i in [1,2,3]]
	MaxEBarLay1_2_3 = [0 for i in [1,2,3]]
	for ihit in range(0, event.pEvtBgoHits().GetHittedBarNumber()):
		lay = (event.pEvtBgoHits().GetLayerID)(ihit)
		if lay in [1,2,3]:
			hitE = (event.pEvtBgoHits().fEnergy)[ihit]
			if hitE > MaxEBarLay1_2_3[lay-1]:
				iBar =  ((event.pEvtBgoHits().fGlobalBarID)[ihit]>>6) & 0x1f		# What the fuck?
				MaxEBarLay1_2_3[lay-1] = hitE
				barNumberMaxEBarLay1_2_3[lay-1] = iBar
	for j in range(3):
		if barNumberMaxEBarLay1_2_3[j] <=0 or barNumberMaxEBarLay1_2_3[j] == 21:
			return False
					
	return True

def FullBGO(bgorec):
	'''
	The reconstructed shower vector is extrapolated to the top and the
	bottom of the sensitive volume of the BGO. If the extrapolated position is more than
	280 mm from the center, either in X or in Y, the event is rejected. Events in which
	a shower vector failed to be reconstructed are rejected
	'''
	BGO_TopZ = 46
	BGO_BottomZ = 448
	
	bgoRec_slope = [  bgorec.GetSlopeYZ() , bgorec.GetSlopeXZ() ]
	bgoRec_intercept = [ bgorec.GetInterceptXZ() , bgorec.GetInterceptYZ() ]
	
	if (bgoRec_slope[1]==0 and bgoRec_intercept[1]==0) or (bgoRec_slope[0]==0 and bgoRec_intercept[0]==0): 
		return False
	
	topX = bgoRec_slope[1]*BGO_TopZ + bgoRec_intercept[1]
	topY = bgoRec_slope[0]*BGO_TopZ + bgoRec_intercept[0]
	bottomX = bgoRec_slope[1]*BGO_BottomZ + bgoRec_intercept[1]
	bottomY = bgoRec_slope[0]*BGO_BottomZ + bgoRec_intercept[0]
	if not all( [ abs(x) < 280 for x in [topX,topY,bottomX,bottomY] ] ):
		return False
		
	return True


##########################
### SECTION 3.2
##########################

def nBarLayer13(bgorec):
	'''
	The number of BGO bars with energy larger than 10 MeV in
	the last layer should not exceed an energy-dependent threshold: nBarLayer13 <
	8 log(bgoTotalE) − 5, where bgoTotalE is in GeV. With this formula the cut is 11
	bars at 100 GeV and 19 at 1 TeV.
	'''
	
	bgoTotalE = bgorec.GetTotalEnergy() / 1000. 	# in GeV
	nBar = 0
	for j in range(22):
		if bgorec.GetEdep(13,j) > 10 : nBar += 1
			
	if nBar < ( (8. * np.log10(bgoTotalE)) - 5) :
		return True
	return False
	
def maxRMS():
	'''
	The maximum shower width of all layers with energy more than 1%
	of the total energy should be less than 100 mm.
	'''
	
	elay = bgorec.GetLayerEnergy()
	efrac = [ elay[i]/bgorec.GetTotalEnergy() for i in range(14) ]
	rms = getRMS(bgorec)
	
	rms_reduced = [ rms[i] for i in range(14) if efrac[i] > 0.01 ]
	
	if max(rms_reduced) > 100 : return False
	
	return True


def lowEnergyCleaning(bgorec):
	'''
	Section 3.2, cut coming from Xin's code
	
	Needs to be clearly studied
	
	Andrii by e-mail: 
	I remember this issue, there were some low-energy protons, with very low energy deposit, 
	falling into the electron xtrl region. The cut was implemented "by eye" (as far as I know) by Xin. If you look at 
	Figure 5 (two plots on the top left), there is shower RMS w.r.t cos(theta), where you can see those events 
	(below magenta line). They occur only  below 250 GeV. 
	First of all, you could produce a plot similar to Figure 5 (using proton and electron MC), and apply your own cut 
	in a similar way to Xin. Or just use his formula (check with him) to put a cut on RMS-cos(theta) plane.
	'''
	
	bgoTotalE = bgorec.GetTotalEnergy() / 1000.		# in GeV
	if bgoTotalE >= 250 : return True 	# Save some computations below
	
	bgoRec_intercept = [ bgorec.GetInterceptYZ() , bgorec.GetInterceptXZ() ]
	bgoRec_slope = [ bgorec.GetSlopeYZ() , bgorec.GetSlopeXZ() ]
	vec_s0_a = TVector3(bgoRec_intercept[1],bgoRec_intercept[0],0.)
	vec_s1_a = TVector3(bgoRec_intercept[1]+bgoRec_slope[1],bgoRec_intercept[0]+bgoRec_slope[0],1.)
	bgoRecDirection = (vec_s1_a - vec_s0_a).Unit() # Unit vector pointing from front to back
	cosBgoRecTheta = np.cos(bgoRecDirection.Theta())
	
	sumRms = sum(getRMS(bgorec))
	
	if bgoTotalE < 100 :
		if sumRms < (270 - 100*(cosBgoRecTheta**6)) : return False
	elif bgoTotalE < 250 :
		if sumRms < (800 - 600*(cosBgoRecTheta**1.2)) : return False	
	
	return True

##########################
### SECTION 3.3 	-	 STK cuts
##########################	
	
	
	


def HET(event):
	
	if not event.pEvtHeader().GeneratedTrigger(3): 	# Should be able to return this directly
		return False								# But with ROOT/Python types you're never sure
	return True
