from __future__ import division

import math
import numpy as np

def getBGOvalues(bgorec):
	'''
	Extract values related to BGO and write them as a python list.
	'''
	templist = []
	
	RMS2 = bgorec.GetRMS2()
	
	# Energy per layer
	for i in xrange(14): templist.append(  bgorec.GetELayer(i)  )
	
	# RMS2 per layer
	for j in xrange(14): 
		if RMS2[j] < 0 :		# In PMO code, if RMS is not defined then RMS2 = -999. Prefer to move it to 0.
			templist.append( 0 )
		else:
			templist.append( RMS2[j] )
	
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

def getValues(bgorec, b_bgorec, nudraw, b_nudraw, evtheader, psdhits, bgohits, stktracks, stkclusters, trackhelper,psdrec,pid):
	'''
	List of variables:
		0 - 13 : Energy in BGO layer i
		14 - 27 : RMS2 of energy deposited in layer i
		28 - 41 : Number of hits in layer i
		
		42 : longitudinal RMS ( DmpEvtBgoRec::GetRMS_l )
		43 : radial RMS ( DmpEvtBgoRec::GetRMS_r )
		44 : total BGO energy (corrected)
		45 : total BGO hits
		46 : theta angle of BGO trajectory
		----
		47 - 48 : Energy in PSD layer 1,2
		49 - 50 : Nr of hits in PSD layer 1,2
		----
		51 : nr of Si clusters
		52 : nr of tracks
		53 - 56 : energy in STK clusters, 4 vertical bins
		57 - 60 : RMS of energy in STK clusters, 4 vertical bins
		----
		61 - 64 : Raw NUD signal
		----
		65 : timestamp
		66 : Particle ID (0 for proton, 1 for electron, 2 for photon)
	'''
	templist = []

	### BGO
	templist = templist + getBGOvalues(bgorec)
	
	### PSD
	templist = templist + getPSDvalues(psdrec)
	
	### STK
	templist = templist + getSTKvalues(stktracks,stkclusters)
	
	### NUD
	templist = templist + getNUDvalues(nudraw)
	
	### Timestamp
	sec = evtheader.GetSecond()					# Timestamp is used as an unique particle identifier for data. If need be.
	msec = evtheader.GetMillisecond()
	if msec >= 1. :
		msec = msec / 1000.
	templist.append(sec + msec)
	
	if pid == 11 :							# Electron
		templist.append(1)
	elif pid == 22 :						# Photon
		templist.append(2)
	else:									# Proton
		templist.append(0)

	
	return templist
