'''

Skimmer v0.1

How to run:
> python skim.py files.txt -p ParticleType -o Outputdir
		files.txt is an ASCII list of files
		ParticleType (optional) : specify kind of particle to select. Default: Proton, Photon and Electrons are selected
		Outputdir (optional) : specify output directory
		
Terminal output:
	A lot of output caused by the DmpSoftware. At the end, run time and nr of events selected/skipped

Files output:
	Creates one directory, user defined. Contains skimmed files

Other features:
	Skips files that have already been skimmed


'''

import sys
import os
import time
from ROOT import gSystem
gSystem.Load("libDmpEvent.so")
from ROOT import DmpChain
import cPickle as pickle
import argparse


class Skim(object):
	
	def __init__(self,filename,particle=None,outputdir='skim_output'):
		
		self.filename = filename
		with open(filename,'r') as fi:
			self.filelist = []
			for line in fi:
				self.filelist.append(line.replace('\n',''))
		
		self.addPrefix()
		
		self.outputdir = outputdir
		if not os.path.isdir(outputdir): os.mkdir(outputdir)
		
		self.particle = self.identifyParticle(particle)
		
		self.openRootFile()
		
		self.t0 = time.time()
		self.selected = 0
		self.skipped = 0
		
	
	def addPrefix(self):
		'''
		adds the XrootD prefix to the filelist, in case it is missing
		'''
		if not 'root://' in self.filelist[0]:
			if not os.path.isfile(self.filelist[0]):
				self.filelist = ['root://xrootd-dampe.cloud.ba.infn.it/' + x for x in self.filelist]
		
	def identifyParticle(self,part):
		'''
		Particle identification based on either the argument or the file name
		'''
		e = ['e','elec','electron','11','E','Elec','Electron']
		p = ['p','prot','proton','2212','P','Prot','Proton']
		gamma = ['g','gamma','photon','22','Gamma','Photon']
		
		if part is None:
			for cat in [e,p,gamma]:
				for x in cat[1:]:
					if x in self.filename:
						return int(cat[3])
			return None
		else:
			for cat in [e,p,gamma]:
				if part in cat:
					return int(cat[3])
			return None
		
	def openRootFile(self):
		'''
		Creates the DmpChain
		'''
		self.chain = DmpChain("CollectionTree")
		for f in self.filelist:
			if not self.alreadyskimmed(f):
				self.chain.Add(f)
		if not self.chain.GetEntries():
			raise IOError("0 events in DmpChain - something went wrong")
		self.chain.SetOutputDir(self.outputdir)
		self.nvts = self.chain.GetEntries()
		
	def getRunTime(self):
		return (time.time() - self.t0)
	def getSelected(self):
		return self.selected
	def getSkipped(self):
		return self.skipped 
	
	def alreadyskimmed(self,f):
		'''
		Checks if file f has already been skimmed
		'''
		temp_f = os.path.basename(f).replace('.root','_UserSel.root')
		temp_f = self.outputdir + '/' + temp_f
		
		if os.path.isfile(temp_f):
			print os.path.basename(f), " already skimmed"
			del temp_f
			return True
		del temp_f
		return False
	
	def run(self):
		self.analysis()
		self.end()
		
	def selection(self,event):
		
		if self.particle is None:
			if event.pEvtSimuPrimaries().pvpart_pdg not in [11,2212,22]:
				return False
		else:
			if event.pEvtSimuPrimaries().pvpart_pdg != self.particle :
				return False
			
		# High energy trigger
		if not event.pEvtHeader().GeneratedTrigger(3):
			return False
		
		# Here: Russlan skimmer
		BGO_TopZ = 46
		BGO_BottomZ = 448
		
		# "BGO tack containment cut" -Russlan
		bgoRec_slope = [  event.pEvtBgoRec().GetSlopeYZ() , event.pEvtBgoRec().GetSlopeXZ() ]
		bgoRec_intercept = [ event.pEvtBgoRec().GetInterceptXZ() , event.pEvtBgoRec().GetInterceptYZ() ]
		if (bgoRec_slope[1]==0 amd bgoRec_intercept[1]==0) or (bgoRec_slope[0]==0 and bgoRec_intercept[0]==0): 
			return False
		
		# "BGO containment cut"
		topX = bgoRec_slope[1]*BGO_TopZ + bgoRec_intercept[1]
		topY = bgoRec_slope[0]*BGO_TopZ + bgoRec_intercept[0]
		bottomX = bgoRec_slope[1]*BGO_BottomZ + bgoRec_intercept[1]
		bgoRec_slope[0]*BGO_BottomZ + bgoRec_intercept[0]
		if not all( [ abs(x) < 280 for x in [topX,topY,bottomX,bottomY] ] ):
			return False
			
		ELayer_max_XZ = 0
		ELayer_max_YZ = 0
		for i in range(1,14,2):
			e = event.pEvtBgoRec().GetELayer(i)
			if e > ELayer_max_XZ: ELayer_max_XZ = e
		for i in range(0,14,2):
			e = event.pEvtBgoRec().GetELayer(i)
			if e > ELayer_max_YZ: ELayer_max_YZ = e
		if 

############################################################################
############################################################################
        double ELayer_max_XZ = 0;
    double ELayer_max_YZ = 0;
    for (int i=1; i<14; i=i+2) { double e=bgorec->GetELayer(i); if (e > ELayer_max_XZ) {ELayer_max_XZ = e; }}
    for (int i=0; i<14; i=i+2) { double e=bgorec->GetELayer(i); if (e > ELayer_max_YZ) {ELayer_max_YZ = e; }}
        
    
    double MaxELayer;
    if (ELayer_max_XZ > ELayer_max_YZ) MaxELayer = ELayer_max_XZ;
    else MaxELayer = ELayer_max_YZ;
    bool passed_maxELayerTotalE_cut = true;
    double rMaxELayerTotalE = MaxELayer/bgoTotalE;
    if(rMaxELayerTotalE>0.35) passed_maxELayerTotalE_cut = false;

    // cut maxBarLayer
    bool  passed_maxBarLayer_cut = true;
    short  barNumberMaxEBarLay1_2_3[3] = {-1}; //bar number of maxE bar in layer 1, 2, 3
    double MaxEBarLay1_2_3[3] = {0};           //E of maxE bar in layer 1, 2, 3
    for (int ihit = 0; ihit <nBgoHits; ihit++){
      double hitE = (bgohits->fEnergy)[ihit];
      short   lay = bgohits->GetLayerID(ihit);
      if(lay==1 || lay==2 || lay==3) {
	if(hitE>MaxEBarLay1_2_3[lay-1]) {
	  int iBar = ((bgohits->fGlobalBarID)[ihit]>>6) & 0x1f;
	  MaxEBarLay1_2_3[lay-1] = hitE;
	  barNumberMaxEBarLay1_2_3[lay-1] = iBar; }}}
    for (int j = 0; j <3; j++){
      if(barNumberMaxEBarLay1_2_3[j] <=0 || barNumberMaxEBarLay1_2_3[j] == 21) passed_maxBarLayer_cut = false; }



    if (apply_cut>1) {
      if (!passed_maxELayerTotalE_cut) continue;
      if (!passed_maxBarLayer_cut) continue;
    }		
		
		
		############################################################################
		############################################################################		
		return True
		
	def analysis(self):
		for i in xrange(self.nvts):
			self.pev = self.chain.GetDmpEvent(i)
			if self.selection(self.pev):
				self.selected += 1
				self.chain.SaveCurrentEvent()
			else:
				self.skipped += 1
				
	def end(self):
		print "- Selected ", self.selected, " events"
		print "- Skipped ", self.skipped, " events"
		print "Run time: ", time.strftime('%H:%M:%S', time.gmtime(self.getRunTime()))
		self.chain.Terminate()
		
		

if __name__ == "__main__" :
	
	parser = argparse.ArgumentParser()
	parser.add_argument("infile",help="Input list of files")
	parser.add_argument("-p", "--particle", help="Particle type, default to None",default=None)
	parser.add_argument("-o", "--output", help="Output folder", default='skim_output')
	args = parser.parse_args()
	
	skim = Skim(args.infile,args.particle,args.output)
	
	skim.run()
	
