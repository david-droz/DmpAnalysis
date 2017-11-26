'''

High-energy skimmer for Monte Carlo data.

Applies the following cuts:
	*
	*
	*
	* High-Energy Trigger


@author: David Droz
@date: 2017-11-27
'''

from __future__ import division, absolute_import, print_function

from ROOT import gSystem
gSystem.Load('libDmpEvent.so')
from ROOT import *

import sys
import os
import yaml

def containmentCut(bgorec):
	'''
	Cut description here
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
	bgoRec_slope[0]*BGO_BottomZ + bgoRec_intercept[0]
	if not all( [ abs(x) < 280 for x in [topX,topY,bottomX,bottomY] ] ):
		return False
		
	return True
	
	
def cutMaxELayer(bgorec,cutValue=0.35):
	'''
	The maximum of the energy deposited in a single layer must be lower than 35% of the total energy
	This cut removes side-entering events, that would deposit most of their energy in a single layer (since layers are horizontal)
	cutValue=0.35 corresponds to the cut applied on flight data.
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
	
	
def maxBarCut(event):
	'''
	The maximum of the energy deposited in the first 3 layers must not be on the side of the layer.
	This cut removes events whose showers won't be well contained.
	'''
	
	barNumberMaxEBarLay1_2_3 = [-1 for i in [1,2,3]]
	MaxEBarLay1_2_3 = [0 for i in [1,2,3]]
	for ihit in range(0, event.pEvtBgoHits().GetHittedBarNumber()):
		hitE = (event.pEvtBgoHits().fEnergy)[ihit]
		lay = (event.pEvtBgoHits().GetLayerID)(ihit)
		if lay in [1,2,3]:
			if hitE > MaxEBarLay1_2_3[lay-1]:
				iBar =  ((event.pEvtBgoHits().fGlobalBarId)[ihit]>>6) & 0x1f		# What the fuck?
				MaxEBarLay1_2_3[lay-1] = hitE
				barNumberMaxEBarLay1_2_3[lay-1] = iBar
	for j in range(3):
		if barNumberMaxEBarLay1_2_3[j] <=0 or barNumberMaxEBarLay1_2_3[j] == 21:
			return False
					
	return True
	
	
	
def main(filelist,outputdir='skim'):
	
	if not os.path.isdir(outputdir): os.mkdir(outputdir)
	if not os.path.isdir('skimStats'): os.mkdir('skimStats')
	
	dmpch = DmpChain("CollectionTree")
	for f in filelist:
		dmpch.Add(f)
		
	dmpch.SetOutputDir(outputdir)
	
	nevents = dmpch.GetEntries()
	
	cuts = { 'HET' : { 'passed' : 0 , 'cut' : 0},
			 'Containment' : { 'passed' : 0, 'cut' : 0},
			 'MaxELayer' : {'passed' : 0, 'cut' : 0},
			 'MaxBar' : {'passed' : 0, 'cut' : 0},
			 'ZeroEnergy' : {'passed' : 0, 'cut' : 0},
			 'selected': 0,
			 'cut': 0				
				}
	
	print("Processing ", nevents, " events")
				
	for i in range(nevents):
		
		pev = dmpch.GetDmpEvent(i)
		bgorec = pev.pEvtBgoRec()
		
		goodEvent = True
		
		listOfCuts = [ ['Containment',containmentCut(bgorec)] , ['MaxELayer',cutMaxELayer(bgorec)] , ['MaxBar',maxBarCut(pev)] , ['HET',pev.pEvtHeader().GeneratedTrigger(3)], ['ZeroEnergy',pev.pEvtBgoRec().GetTotalEnergy() > 0 ]]
		
		for tag,result in listOfCuts:
			if not result:
				cuts[tag]['cut'] += 1
				goodEvent = False
			else:
				cuts[tag]['passed'] += 1
			
		if goodEvent:
			cuts['passed'] += 1
			dmpch.SaveCurrentEvent()
		else:
			cuts['cut'] += 1
	
	
	dmpch.Terminate()
	
	try:
		outname = 'skimStats/'+sys.argv[1]+'.yaml'
	except IndexError:
		print("Error when writing skim statistics. Writing them to 'stats.yaml' instead")
		outname = 'stats.yaml'
		
	with open(outname,'w') as f:
		yaml.dump(cuts,f)
	
		
	
	
if __name__ == '__main__':
	
	if len(sys.argv) == 1:
		raise Exception("Not enough arguments! Expected at least one: list of files to skim")
	
	l = []
	with open(sys.argv[1]) as f:
		for line in f:
			l.append(line.replace('\n',''))
			
	if len(sys.argv) == 2:
		main(l)
	else:		
		main(l,sys.argv[2])
