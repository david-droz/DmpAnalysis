from __future__ import division

import math
import numpy as np

def getXTRL(bgorec):
	
	'''
	From a given event, returns the energy ratio and energy RMS in all BGO layers, and XTRL/zeta
	'''
	
	NBGOLAYERS  = 14
	NBARSLAYER  = 22
	EDGEBARS    = [0,21]
	BARPITCH    = 27.5
	
	edep = np.zeros((NBGOLAYERS,NBARSLAYER))
	for i in xrange(NBGOLAYERS):
		for j in xrange(NBARSLAYER):
			edep[i,j] = bgorec.GetEdep(i,j)
	
	BHET = edep.sum()
	BHXS = [0. for i in xrange(NBGOLAYERS)]
	BHE = [0. for i in xrange(NBGOLAYERS)]
	BHER = [0. for i in xrange(NBGOLAYERS)]
	COG = [0. for i in xrange(NBGOLAYERS)]
	bhm  = 0.
	SIDE = [False for i in xrange(NBGOLAYERS)]
	
	for i in xrange(NBGOLAYERS):
		# Find the bar with max energy deposition of a layer and record its number as im
		im = None
		em = 0.0;
		for j in xrange(NBARSLAYER):
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
		for j in xrange(NBARSLAYER):
			ebar = edep[i,j]
			posbar = BARPITCH * j 
			enelayer += ebar
			posrms += ebar * (posbar-cog)*(posbar-cog)
		posrms = math.sqrt( posrms / enelayer)
		BHXS[i] = posrms
		COG[i] = cog
		BHE[i] = enelayer
		BHER[i] = enelayer / BHET
	
	sumRMS = sum(BHXS)
	F = [r for r in reversed(BHER) if r][0]
	XTRL = sumRMS**4.0 * F / 8000000.
	
	del edep
	
	return BHE, BHER, BHXS, XTRL

def getBGOvalues(bgorec):
	'''
	Extract values related to BGO and write them as a python list.
	'''
	templist = []
	
	#~ RMS2 = bgorec.GetRMS2()
	ELayer, ELayerFrac, RMS, zeta = getXTRL( bgorec )
	
	# Energy per layer
	for i in xrange(14): templist.append(  ELayer[i]  )
	
	# RMS2 per layer
	for j in xrange(14): templist.append( RMS[j] )
	
	# Hits on every layer		
	hitsPerLayer = bgorec.GetLayerHits()
	for k in xrange(14):
		templist.append(hitsPerLayer[k])
		
	templist.append( bgorec.GetRMS_l() )
	templist.append( bgorec.GetRMS_r() )

	#~ templist.append( bgorec.GetElectronEcor() )
	templist.append( bgorec.GetTotalEnergy() )
	templist.append( bgorec.GetTotalHits() )
	
	# Angle of reconstructed trajectory
	XZ = bgorec.GetSlopeXZ()
	YZ = bgorec.GetSlopeYZ()
	
	tgZ = math.atan(np.sqrt( (XZ*XZ) + (YZ*YZ) ) )
	templist.append(tgZ*180./math.pi)
	
	return templist

def getPSDvalues(psdrec):
	'''
	Extracts PSD values and return as a Python list
	'''
	templist = []
	
	templist.append(psdrec.GetLayerEnergy(0))
	templist.append(psdrec.GetLayerEnergy(1))
	templist.append(psdrec.GetLayerHits(0))
	templist.append(psdrec.GetLayerHits(1))

	return templist
	
def getSTKvalues(stktracks,stkclusters):
	'''
	Extracts STK values and return as list
	'''
	templist = []
	nBins = 4				# In DmpSoftware package, STK is not defined per layers. 
							# Here we treat it as a calorimeter
	# Nr of clusters, nr of tracks
	nrofclusters = stkclusters.GetLast()+1
	templist.append(nrofclusters)
	templist.append(stktracks.GetLast()+1)
	
	if nrofclusters == 0:
		for i in xrange(nBins):
			templist.append(0)
			templist.append(0)
		return templist
	
	# Below: compute the RMS of cluster distributions
	l_pos = np.zeros(nrofclusters)
	l_z = np.zeros(nrofclusters)
	l_energy = np.zeros(nrofclusters)
	
	for i in xrange(nrofclusters):
		pos = stkclusters.ConstructedAt(i).GetH()
		z = stkclusters.ConstructedAt(i).GetZ()
		energy = stkclusters.ConstructedAt(i).getEnergy()
		
		l_pos[i] = pos
		l_z[i] = z
		l_energy[i] = energy
	
	minz = np.min(l_z)
	maxz = np.max(l_z)
	bins = np.linspace(minz,maxz,nBins+1)
	
	ene_per_bin = []
	rms_per_bin = []
	
	for i in xrange(nBins):
		
		cog = 0
		ene_tot = 0
		for j in xrange(nrofclusters):
			if l_z[j] < bins[i] or l_z[j] > bins[i+1]:	# Wrong bin
				continue
			ene_tot = ene_tot + l_energy[j]
			cog = cog + l_pos[j]*l_energy[j]
		if ene_tot == 0:
			ene_per_bin.append(0)
			rms_per_bin.append(0)
			continue
		cog = cog/ene_tot
		rms = 0
		ene_per_bin.append(ene_tot)
		for j in xrange(nrofclusters):
			if l_z[j] < bins[i] or l_z[j] > bins[i+1]:	# Wrong bin
				continue
			rms = rms + (l_energy[j] * (l_pos[j] - cog) * (l_pos[j] - cog) )
		rms = math.sqrt(rms/ene_tot)
		rms_per_bin.append(rms)

	for x in ene_per_bin:
		templist.append(x)
	for y in rms_per_bin:
		templist.append(y)
	
	del l_pos, l_z, l_energy, ene_per_bin, rms_per_bin
	return templist	
	
def getNUDvalues(nudraw):
	'''
	Extract raw ADC signal from NUD
	'''
	templist = [0 for x in xrange(4)]
	f = nudraw.fADC
	for i in xrange(4): 
		templist[i] = f[i]
	return templist


# ~ def getStkNew():
	
	# ~ Rm_W   =  9.327
	# ~ STK_TopZ    = -210
	
	# ~ TClonesArray* stk_clusters      = new TClonesArray("DmpStkSiCluster");
	# ~ TBranch        *b_StkClusterCollection
	# ~ tree->SetBranchAddress("StkClusterCollection",     &stk_clusters, &b_StkClusterCollection);
	
	
	
	# ~ mindAngleTrack       = mindAngleTrack_3clu;   # mindAngleTrack_3clu
	# ~ mindAngleTrackBgoRec = mindAngleTrackBgoRec_3clu;  # mindAngleTrackBgoRec_3clu
	# ~ mindDrTopTrackBgoRec = mindDrTopTrackBgoRec_3clu;  # mindDrTrackBgoRec_3clu
	# ~ mindAngleTrackDirection     = mindAngleTrackDirection_3clu;   ####
	# ~ mindAngleTrack_slope[0]     = mindAngleTrack_slope_3clu[0];   ####
	# ~ mindAngleTrack_slope[1]     = mindAngleTrack_slope_3clu[1];
	# ~ mindAngleTrack_intercept[0] = mindAngleTrack_intercept_3clu[0];   ####
	# ~ mindAngleTrack_intercept[1] = mindAngleTrack_intercept_3clu[1];
	# ~ cosMindAngleTrackTheta = cos(mindAngleTrackDirection_3clu.Theta());

	# ~ TVector3 trackStkTopPoint
	# ~ trackStkTopPoint[2] = STK_TopZ;
	# ~ trackStkTopPoint[0] = mindAngleTrack_slope[1]*STK_TopZ + mindAngleTrack_intercept[1];
	# ~ trackStkTopPoint[1] = mindAngleTrack_slope[0]*STK_TopZ + mindAngleTrack_intercept[0];
	
	# ~ for iclu in xrange( stk_clusters->GetLast()+1 ):
	
		# ~ DmpStkSiCluster* cluster = (DmpStkSiCluster*)(stk_clusters->ConstructedAt(iclu));		
		
		# ~ int isX = cluster->isX();
		# ~ double hitX =  cluster->GetX();
		# ~ double hitY =  cluster->GetY();
	
		# ~ int hardID     = cluster->getLadderHardware();
		# ~ double hitE    = cluster->getEnergy();
	
	
		# ~ double thisCoord = isX ? hitX : hitY ;   
		
		# ~ TVector3 mindAnglePoint(mindAngleTrack_slope[1]*hitZ + mindAngleTrack_intercept[1], 
				# ~ mindAngleTrack_slope[0]*hitZ + mindAngleTrack_intercept[0], hitZ);		
		
		
		# ~ thisPoint = mindAnglePoint;
		# ~ thisDirection = mindAngleTrackDirection; 
		
		# ~ double predToCenter = 0;
		# ~ predToCenter = fabs(thisPoint[m]);   
		
		
		# ~ thisPoint[k] = thisCoord; #0 for x, 1 for y 
		# ~ TVector3 trackStkTopPointToThis = thisPoint - trackStkTopPoint; 
		
		# ~ TVector3 stripDirection = isX ? XbarDirection : YbarDirection;
		
		# ~ TVector3 crossStripTrack = stripDirection.Cross(thisDirection);
	
		# ~ transLength = fabs(trackStkTopPointToThis*crossStripTrack)/crossStripTrack.Mag();
	
	
		
	
		# ~ double dRmOther = predToCenter/Rm_W;		
		# ~ double dRm = transLength/Rm_W;	
	
	
		# ~ if dRm < 1 and dRmOther < 1 :
			# ~ stkEcore1Rm   += hitE
	
	# ~ # End for
		
	# ~ tt_stkEcore1Rm_trk  = stkEcore1Rm




def getValues(bgorec, b_bgorec, nudraw, b_nudraw, evtheader, psdhits, bgohits, stktracks, stkclusters, trackhelper,psdrec,pid):
	
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
		80 : sumRMS
		81 : BGO XZ slope
		82 : BGO YZ slope
		83 : Particle ID (0 for proton, 1 for electron)
	'''
	
	predArray = np.zeros( (84,) , dtype='float32')
	
	ELayer, ELayerFrac, RMS, zeta = getXTRL( bgorec )
	
	
	for ilay in range(14):
		predArray[ilay] = ELayer[ilay]
		
		predArray[ilay+14] = RMS[ilay]
		
		predArray[ilay+50] = ELayerFrac[ilay]
		
		if 100*ELayerFrac[ilay] > 1e-6 :
			predArray[ilay+64] = np.log10( 100*ELayerFrac[ilay] )
		else:
			predArray[ilay+64] = np.log10( 1e-6 )
	
	
	XZ = bgorec.GetSlopeXZ()
	YZ = bgorec.GetSlopeYZ()
	tgZ = math.atan(np.sqrt( (XZ*XZ) + (YZ*YZ) ) )
	
	FRACarr = np.array(ELayerFrac)
	FLast = FRACarr[ FRACarr > 0 ][-1]	
	
	
	predArray[42] = bgorec.GetRMS_l()
	predArray[43] = bgorec.GetRMS_r()
	predArray[44] = bgorec.GetElectronEcor()
	predArray[45] = FLast
	predArray[46] = tgZ*180./math.pi
	# ~ predArray[n,47] = TT.tt_ekin
	# ~ predArray[n,48] = TT.tt_evtPoid
	predArray[49] = zeta
	
	# ~ predArray[n,78] = TT.tt_stkEcore1Rm_trk
	# ~ predArray[n,79] = TT.tt_nStkClu_trk

	predArray[80] = np.sum(RMS)
	predArray[81] = XZ
	predArray[82] = YZ


	if pid == 11 :							# Electron
		predArray[-1] = 1
	elif pid == 22 :						# Photon
		predArray[-1] = 2
	else:									# Proton
		predArray[-1] = 0

	return predArray


	# ~ ### BGO
	# ~ templist = templist + getBGOvalues(bgorec)
	
	# ~ ### PSD
	# ~ '''
	# ~ 47 - 48 : Energy in PSD layer 1,2					---> REPLACE BY EKIN AND BY 0 (evtPoid)
	# ~ 49 - 50 : Nr of hits in PSD layer 1,2				---> 49, XTRL. 50-63, fracLayer
	# ~ '''
	# ~ templist = templist + getPSDvalues(psdrec)
	
	# ~ ### STK
	# ~ '''
	# ~ 51 : nr of Si clusters								---> 64-77, logLayer
	# ~ 52 : nr of tracks
	# ~ 53 - 56 : energy in STK clusters, 4 vertical bins	---> 78,  tt_stkEcore1Rm_trk
	# ~ 57 - 60 : RMS of energy in STK clusters, 4 vertical bins
	# ~ '''
	# ~ templist = templist + getSTKvalues(stktracks,stkclusters)
	
	# ~ ### NUD
	# ~ # 61 - 64 : Raw NUD signal
	# ~ templist = templist + getNUDvalues(nudraw)
	
	# ~ ### XTRL
	# ~ ELayer, ELayerFrac, RMS, zeta = getXTRL(bgorec)
	# ~ templist.append(zeta)	
	# ~ del ELayer, RMS, zeta
	
	# ~ ### Timestamp
	# ~ sec = evtheader.GetSecond()					# Timestamp is used as an unique particle identifier for data. If need be.
	# ~ msec = evtheader.GetMillisecond()
	# ~ if msec >= 1. :
		# ~ msec = msec / 1000.
	# ~ templist.append(sec + msec)
	
	# ~ if pid == 11 :							# Electron
		# ~ templist.append(1)
	# ~ elif pid == 22 :						# Photon
		# ~ templist.append(2)
	# ~ else:									# Proton
		# ~ templist.append(0)

	
	# ~ return templist
