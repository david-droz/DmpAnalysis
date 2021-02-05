'''

BTselection : Single function: from a DmpEvent instance, returns a boolean on whether or not this event should be kept.

Source: Andrii's code

'''

from __future__ import division

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
from ROOT import *
import pickle

from BTcuts import *   # Import global constants and definitions



##########################
##########################

def is_close_bgocrack(x):
	dist = abs(x) % BGO_BAR_STEP
	if dist > BGO_BAR_STEP * 0.5: dist = BGO_BAR_STEP - dist
	return dist < BGO_CRACK_DISTANCE + BGO_CRACK_ADDSAFETYMARGIN


def __xtr__(X,F): return X**4.0 * F / 8000000.

def bgo_hit_isx(bgohits,i):
	return bgohits.GetLayerID(i)%2

def clusterize(histo,thr1,thr2):
	nbins = histo.GetNbinsX()
	seeds = []
	for i in xrange(1,nbins+1):
		if i == 1:
			if histo.GetBinContent(i) < histo.GetBinContent(i+1) + thr2: continue
		elif i == nbins:
			if histo.GetBinContent(i) < histo.GetBinContent(i-1) + thr2: continue
		else:
			if  histo.GetBinContent(i) <= histo.GetBinContent(i-1) or  histo.GetBinContent(i) <= histo.GetBinContent(i+1): continue
			if  histo.GetBinContent(i) <= histo.GetBinContent(i-1) + thr1 and histo.GetBinContent(i) <= histo.GetBinContent(i+1) + thr1: continue
		seeds.append(i)
	return seeds

def get_clusters(bgohits):
	NBINS = 22
	THRESHOLD1 = 1000.
	THRESHOLD2 = 500.
	histox = TH1F("histox","histox", NBINS, -302.5,302.5)
	histoy = TH1F("histoy","histoy", NBINS, -302.5,302.5)
	
	for i in xrange(len(bgohits.fEnergy)):
		e = bgohits.fEnergy[i]
		if bgo_hit_isx(bgohits,i):
			histox.Fill(bgohits.GetHitX(i),e)
		else:
			histoy.Fill(bgohits.GetHitY(i),e)
	clustersx = clusterize(histox,THRESHOLD1,THRESHOLD2)
	clustersy = clusterize(histoy,THRESHOLD1,THRESHOLD2)
	return clustersx, clustersy

def get_maximum_peak_distance(clusters):
	distances = [clusters[i] - clusters[i-1] for i in xrange(1,len(clusters))]
	return max([0] + distances)

def __get_track_parameters__(track,stkclusters):
	chargeaverage   = 0.
	chargeaverage_n = 0
	chargemax       = 0.
	chargeaverage_atbgo    = 0.
	chargeaverage_atbgo_n  = 0
	chargemax_atbgo        = 0.
	chargeaverage_atpsdx   = 0. 
	chargeaverage_atpsdx_n = 0
	chargeaverage_atpsdy   = 0. 
	chargeaverage_atpsdy_n = 0
	nstrips_atpsdx       = 0
	nstrips_atpsdy       = 0
	for p in xrange(track.GetNPoints()):
		plane = track.getPlane(p)
		clusx = track.GetClusterX(p,stkclusters)
		clusy = track.GetClusterY(p,stkclusters)
		if clusx:
			chargex = clusx.GetSignalTotal()
			chargeaverage+=chargex
			chargeaverage_n+=1
			if chargex > chargemax : chargemax = chargex
			if plane in PLANESATBGO: chargeaverage_atbgo+=chargex
			if plane in PLANESATBGO: chargeaverage_atbgo_n+=1
			if plane in PLANESATBGO and chargex > chargemax_atbgo : chargemax_atbgo = chargex
			if plane in PLANESATPSD: chargeaverage_atpsdx+=chargex
			if plane in PLANESATPSD: chargeaverage_atpsdx_n+=1
			if plane in PLANESATPSD: nstrips_atpsdx+= clusx.getNstrip()
		if clusy:
			chargey = clusy.GetSignalTotal()
			chargeaverage+=chargey
			chargeaverage_n+=1
			if chargey > chargemax : chargemax = chargey
			if plane in PLANESATBGO: chargeaverage_atbgo+=chargey
			if plane in PLANESATBGO: chargeaverage_atbgo_n+=1
			if plane in PLANESATBGO and chargey > chargemax_atbgo : chargemax_atbgo = chargey
			if plane in PLANESATPSD: chargeaverage_atpsdy+=chargey
			if plane in PLANESATPSD: chargeaverage_atpsdy_n+=1
			if plane in PLANESATPSD: nstrips_atpsdy+=clusy.getNstrip()
	chargeaverage /= chargeaverage_n if chargeaverage_n else -1.
	chargeaverage_atbgo /= chargeaverage_atbgo_n if chargeaverage_atbgo_n else -1.
	chargeaverage_atpsdx /= chargeaverage_atpsdx_n if chargeaverage_atpsdx_n else -1.
	chargeaverage_atpsdy /= chargeaverage_atpsdy_n if chargeaverage_atpsdy_n else -1.
	nstrips_atpsdx /= chargeaverage_atpsdx_n if chargeaverage_atpsdx_n else -1.
	nstrips_atpsdy /= chargeaverage_atpsdy_n if chargeaverage_atpsdy_n else -1.
	return chargeaverage, chargeaverage_atbgo, chargemax, chargemax_atbgo, chargeaverage_atpsdx, chargeaverage_atpsdy, nstrips_atpsdx, nstrips_atpsdy

def incrementCutCount(dic,key):
	raise Exception("Don't use incrementCutCount! Key:" +key)
	try:
		dic[key] += 1
	except KeyError:
		dic[key] = 1

'''		
 Result:
 {'HIGH_REC_ENERGY_MIN': 2581, 'PMO_FIRSTLAYERNOTSIDE': 3961, 'PMO_NOTALLLAYERS': 531, 'REMOVE_PILEUP_BGO': 64, 
	'LONG_TRACK_NPOINTSPASSED': 28064, 'PMO_MAXLAYERCUT': 234, 'HIGH_ENERGY_TRIGGER_SELECTION': 1438, 
	'NO_BGO_CRACK_CHARGEAVERAGE': 28064, 'BT_E_250': 22672}
'''

def BTselection(bgorec, b_bgorec, nudraw, b_nudraw, evtheader, psdhits, bgohits, stktracks, stkclusters, trackhelper,dataset):
	
	#~ if os.path.isfile('cutCounts.pick'):
		#~ mydic = pickle.load(open('cutCounts.pick','rb'))
	#~ else:
		#~ mydic = {}
	
	DO_PMO_PRESELECTION, HIGH_REC_ENERGY_SELECTION, HIGH_ENERGY_TRIGGER_SELECTION, REMOVE_PILEUP, REMOVE_PILEUP_IMPACT, REMOVE_PILEUP_BGO, NO_BGO_CRACK_SELECTIONY, LONG_TRACK_SELECTION, NO_BGO_CRACK_SELECTIONX, INVERSE_NOCRACKSELECTION, HIGH_REC_ENERGY_MIN, HIGH_REC_ENERGY_MAX, BT_E_250 = BTcuts(dataset)
	
	if HIGH_ENERGY_TRIGGER_SELECTION:
		if not evtheader.GeneratedTrigger(3): 
			#~ incrementCutCount(mydic,'HIGH_ENERGY_TRIGGER_SELECTION')
			return False
	
	if BT_E_250 :
		YZ = bgorec.GetInterceptYZ()
		if YZ < 30. or YZ > 170:
			#~ incrementCutCount(mydic,'BT_E_250')
			return False
	
	BHET = sum([ 
		sum([
			bgorec.GetEdep(i,j) for j in xrange(NBARSLAYER)
			])  
		for i in xrange(NBGOLAYERS) 
	])

	# ENERGY CUT
	if HIGH_REC_ENERGY_SELECTION:
		if BHET < HIGH_REC_ENERGY_MIN: 
			#~ incrementCutCount(mydic,'HIGH_REC_ENERGY_MIN')
			return False 
		if BHET > HIGH_REC_ENERGY_MAX: 
			#~ incrementCutCount(mydic,'HIGH_REC_ENERGY_MAX')
			return False
		
	# LAYER CHARACTERISTICS
	BHXS = [0. for i in xrange(NBGOLAYERS)]
	BHER = [0. for i in xrange(NBGOLAYERS)]
	# MAXIMUM LAYER FRCTION
	bhm  = 0.

	SIDE = [False for i in xrange(NBGOLAYERS)]
	
	# ANALYZR LAYERS
	for i in xrange(NBGOLAYERS):
		# Find the bar with max energy deposition of a layer and record its number as im
		im = None
		em = 0.0
		for j in xrange(NBARSLAYER):
			ebar = bgorec.GetEdep(i,j)
			if ebar < em : continue 
			em = ebar
			im = j

		# non-zero em
		if not em: continue;
		

		# 1. find cog in the layer
		if im in EDGEBARS:
			cog = BARPITCH * im  			
		else:
			ene =0.0
			cog =0.0
			for  j in [im-1, im, im+1]: 
				ebar = bgorec.GetEdep(i,j)
				ene+=ebar;
				cog+= BARPITCH * j * ebar;
			cog/=ene

		# 2. find rms in the layer
		posrms   = 0.0
		enelayer = 0.0
		for j in xrange(NBARSLAYER):
			ebar = bgorec.GetEdep(i,j)
			posbar = BARPITCH * j 
			enelayer += ebar
			posrms += ebar * (posbar-cog)*(posbar-cog)
		posrms = math.sqrt( posrms / enelayer)
		BHXS[i] = posrms
		BHER[i] = enelayer / BHET

		# 3. maximum among layers
		if im not in EDGEBARS:
			if BHER[i]>bhm: bhm=BHER[i]

		# Andrii
		if im in EDGEBARS:
			SIDE[i] = True

	# BGO PMO pre-selection
	if DO_PMO_PRESELECTION:
		if len([r for r in BHER if r]) < NBGOLAYERS: 
			#~ incrementCutCount(mydic,'PMO_NOTALLLAYERS')
			return False
		if [SIDE[s] for s in FIRSTLAYERSNOTSIDE if SIDE[s] ]: 
			#~ incrementCutCount(mydic,'PMO_FIRSTLAYERNOTSIDE')
			return False
		if bhm > MAXLAYERCUT: 
			#~ incrementCutCount(mydic,'PMO_MAXLAYERCUT')
			return False
		
	# TRACK CUTS
	sortedtracks = False
	chargeaverage = None
	if TRACK_SELECTION or TRACK_NOTPROTON_SELECTION or NO_BGO_CRACK_SELECTIONY or NO_BGO_CRACK_SELECTIONX:
		trackhelper.SortTracks(2, False)
		if trackhelper.GetSize():
			chargeaverage, chargeaverage_atbgo, chargemax, chargemax_atbgo, chargeaverage_atpsdx, chargeaverage_atpsdy, nstrips_atpsdx, nstrips_atpsdy = __get_track_parameters__(trackhelper.GetTrack(0),stkclusters)
		else:
			chargeaverage = None
		if chargeaverage is None: 
			#~ incrementCutCount(mydic,'NO_BGO_CRACK_CHARGEAVERAGE')
			return False
		if TRACK_SELECTION and chargeaverage > CHARGE_AVERAGE_CUT: 
			#~ incrementCutCount(mydic,'TRACK_SELECTION_CHARGE_AVERAGE_CUT')
			return False
		if TRACK_NOTPROTON_SELECTION and chargeaverage < CHARGE_AVERAGE_CUT: 
			#~ incrementCutCount(mydic,'TRACK_NOTPROTON_CHARGE_AVERAGE_CUT')
			return False
		if NO_BGO_CRACK_SELECTIONY: 
			for t in xrange(stktracks.GetLast()+1):
				trackparams = stktracks.ConstructedAt(t).getTrackParams()
				if is_close_bgocrack(trackparams.getInterceptY()) and not INVERSE_NOCRACKSELECTION: 
					#~ incrementCutCount(mydic,'NO_BGO_CRACK_Y_INVERSE_1')
					return False
				if not is_close_bgocrack(trackparams.getInterceptY()) and INVERSE_NOCRACKSELECTION: 
					#~ incrementCutCount(mydic,'NO_BGO_CRACK_Y_INVERSE_2')
					return False
		if NO_BGO_CRACK_SELECTIONX:
			for t in xrange(stktracks.GetLast()+1):
				trackparams = stktracks.ConstructedAt(t).getTrackParams()
				if is_close_bgocrack(trackparams.getInterceptX()) and not INVERSE_NOCRACKSELECTION: 
					#~ incrementCutCount(mydic,'NO_BGO_CRACK_X_INVERSE_1')
					return False
				if not is_close_bgocrack(trackparams.getInterceptX()) and INVERSE_NOCRACKSELECTION: 
					#~ incrementCutCount(mydic,'NO_BGO_CRACK_X_INVERSE_2')
					return False
		sortedtracks = True
		
	# ONE LONG TRACK SELECTION
	if LONG_TRACK_SELECTION:
		npointspassed = False
		for t in xrange(stktracks.GetLast()+1):				# stktracks.GetLast() always evaluate at -1 !! So this loop does not even start
			track = stktracks.ConstructedAt(t)
			if abs(bgorec.GetInterceptYZ() - track.getTrackParams().getInterceptY()) > BGO_STK_INTERCEPTMATCH: continue
			if abs(bgorec.GetSlopeYZ() - track.getTrackParams().getSlopeY()) > BGO_STK_ANGMATCH: continue
			if track.GetNPoints() < TRACK_MINNPOINTS_SELECTION: continue
			npointspassed = True
			break
		if not npointspassed:
			#~ incrementCutCount(mydic,'LONG_TRACK_NPOINTSPASSED')
			return False
			
	# TRACK_BASED PILEUP
	if REMOVE_PILEUP:
		bgoclosetrack = False
		bgoslx = bgorec.GetSlopeXZ()
		bgosly = bgorec.GetSlopeYZ()
		bgpprojx = bgorec.GetInterceptXZ() +  PILEUP_BGO_Z * bgorec.GetSlopeXZ()
		bgpprojy = bgorec.GetInterceptYZ() +  PILEUP_BGO_Z * bgorec.GetSlopeYZ()
		for track1_i in xrange(stktracks.GetLast()+1):
			track1 = stktracks.ConstructedAt(track1_i)
			slx1 = track1.getTrackParams().getSlopeX()
			sly1 = track1.getTrackParams().getSlopeY()
			inx1 = track1.getTrackParams().getInterceptX()
			iny1 = track1.getTrackParams().getInterceptY()
			for track2_i in xrange(track1_i + 1, stktracks.GetLast()+1):
				track2 = stktracks.ConstructedAt(track2_i)
				slx2 = track2.getTrackParams().getSlopeX()
				sly2 = track2.getTrackParams().getSlopeY()
				inx2 = track2.getTrackParams().getInterceptX()
				iny2 = track2.getTrackParams().getInterceptY()
				differences = [ z for z in  PILEUP_ZCOORDINATES
						if math.sqrt((inx1 + z*slx1 - inx2 - z*slx2)**2 + (iny1 + z*sly1 - iny2 - z*sly2)**2) < PILEUP_DISTANCE
						]
				if differences: continue
				# found pileup
				#~ incrementCutCount(mydic,'REMOVE_PILEUP')
				return False

			# BGO
			stkprojx = inx1 + slx1 * PILEUP_BGO_Z
			stkprojy = iny1 + sly1 * PILEUP_BGO_Z
			xydiff = math.sqrt((bgpprojx-stkprojx)**2 + (bgpprojy-stkprojy)**2 )				
			angldiff = math.sqrt((bgoslx-slx1)**2 + (bgosly-sly1)**2 )
			if xydiff  <  PILEUP_BGO_XYCUT and  angldiff < PILEUP_BGO_INCLCUT:
				bgoclosetrack = True
		
	# CHECK BGO 
	if REMOVE_PILEUP_IMPACT and not bgoclosetrack: 
		#~ incrementCutCount(mydic,'REMOVE_PILEUP_IMPACT')
		return False

	# BGO BASED PILE UP
	if REMOVE_PILEUP_BGO:
		clustersx, clustersy = get_clusters(bgohits)
		maxdistance = max(get_maximum_peak_distance(clustersx), get_maximum_peak_distance(clustersy))
		if maxdistance > 1:
			#~ incrementCutCount(mydic,'REMOVE_PILEUP_BGO')
			return False
			
	# PSD PRESELECTION
	if PSD_DOUBLEMIP_CUT:
		if [i for i in xrange(len(psdhits.fEnergy)) if psdhits.fEnergy[i]>PSD_HIT_ENERGY_DOUBLEMIP_CUT]: 
			#~ incrementCutCount(mydic,'PSD_DOUBLEMIP_CUT')
			return False
			
	#~ with open('cutCounts.pick','wb') as f:
		#~ pickle.dump(mydic,f)

	return True 
