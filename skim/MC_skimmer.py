'''

High-energy skimmer for Monte Carlo data.

Applies the following cuts, and records their statistics:
	* Non-zero energy
	* BGO Acceptance cut: top (Z=46) and bottom BGO (Z=448) projection within 280 mm from the center
	* Fraction maximum layer cut: EneLayerMax/Etot > 0.35 is excluded 
	* BGO Max Bar cut: the maximum of the 1st,2nd and 3rd (count starts from 0) layer does not have to be in the first and last column. 
	* High-Energy Trigger


Takes two arguments:
	1. Text file with the list of Root files to process
	2. Output directory to write the new root files  (default:  ./skim/)
	
Writes a yaml file containing the statistics of all cuts under  ./skimStats/

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
import time
import signal


class GracefulKiller:
	'''
	https://stackoverflow.com/questions/18499497/how-to-process-sigterm-signal-gracefully
	'''
	kill_now = False
	def __init__(self):
		signal.signal(signal.SIGINT, self.exit_gracefully)
		signal.signal(signal.SIGTERM, self.exit_gracefully)
	
	def exit_gracefully(self,signum, frame):
		self.kill_now = True

def alreadyskimmed(f,out):
	
	existingFile = out + '/' + os.path.basename(f).replace('.root','_UserSel.root')
	if os.path.isfile(existingFile):
		return True
	else:
		return False

def containmentCut(bgorec):
	'''
	BGO Acceptance cut: top (Z=46) and bottom BGO (Z=448) projection within 280 mm from the center
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
				iBar =  ((event.pEvtBgoHits().fGlobalBarID)[ihit]>>6) & 0x1f		# What the fuck?
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
		if not alreadyskimmed(f,outputdir):
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
	
	killer = GracefulKiller()
				
	for i in range(nevents):
		
		pev = dmpch.GetDmpEvent(i)
		
		goodEvent = True
		
		listOfCuts = [ ['Containment',containmentCut(pev.pEvtBgoRec())] , ['MaxELayer',cutMaxELayer(pev.pEvtBgoRec())] , ['MaxBar',maxBarCut(pev)] , ['HET',pev.pEvtHeader().GeneratedTrigger(3)], ['ZeroEnergy',pev.pEvtBgoRec().GetTotalEnergy() > 0 ]]
		
		for tag,result in listOfCuts:
			if not result:
				cuts[tag]['cut'] += 1
				goodEvent = False
			else:
				cuts[tag]['passed'] += 1
			
		if goodEvent:
			cuts['selected'] += 1
			try:
				dmpch.SaveCurrentEvent()
			except SystemError:
				currentFile = dmpch.GetFile().GetName()
				try:
					dmpch.Terminate()
				except:
					pass
				writtenFile = outputdir + '/' + os.path.basename(f).replace('.root','_UserSel.root')
				os.remove(writtenFile)	
				raise
		else:
			cuts['cut'] += 1
			
		if killer.kill_now:		# Job killed, e.g. by PBS or Slurm
								# Killing a job while it runs can corrupt the opened file. So
								# we remove the file that was under writing
			currentFile = dmpch.GetFile().GetName()
			try:
				dmpch.Terminate()
			except:
				pass
			writtenFile = outputdir + '/' + os.path.basename(f).replace('.root','_UserSel.root')
			os.remove(writtenFile)	
			sys.exit("Job killed")
	
	
	dmpch.Terminate()
	
	try:
		outname = 'skimStats/'+os.path.splitext(os.path.basename(sys.argv[1]))[0]+'.yaml'
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
