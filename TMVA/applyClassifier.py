"""
	@author: zimmer
	@created: 2017-07-18
	@brief: script to apply classifiers to existing 2A data
	@todo: may extend to running many files at once.
"""

from sys import argv
from shutil import copy
from numpy import zeros
from argparse import ArgumentParser

def main(args=None):
	usage = "Usage: %(prog)s [options]"
	description = "adding MVAtree to existing 2A file, will create a copy"
	parser = ArgumentParser(usage=usage, description=description)
	parser.add_argument("-i","--infile",dest='infile',type=str,default=None, help='name of input file',required=True)
	parser.add_argument("-o","--outfile",dest='outfile',type=str,default=None, help='name of output file',required=True)
	opts = parser.parse_args(args)
	from ROOT import TFile, TTree, gSystem, TObject, gROOT
	gSystem.Load("libDmpEvent.so")
	gROOT.SetBatch(True)
	from ROOT import DmpChain
	# first, make copy of outfile
	copy(opts.infile,opts.outfile)
	# next, load CollectionTree
	dpch = DmpChain("CollectionTree")
	dpch.Add(opts.infile)
	fout= TFile(opts.outfile,"update")
	fTree = TTree("MVAtree","MVA scores")
	# modify, add new variables here
	# basically, you need to add an 'array' of doubles "d" with length 1.
	DNN_score = zeros(1, dtype=float)
	BDT_score = zeros(1, dtype=float)
	# register branch in tree
	fTree.Branch("DNN_score",DNN_score,"DNN_score/D")
	fTree.Branch("BDT_score",BDT_score,"BDT_score/D")
	
	# next is the usual event loop
	nevts = dpch.GetEntries()
	print 'found {i} events in {ifile}'.format(i=nevts,ifile=opts.infile)
	for i in xrange(nevts):
		pev = dpch.GetDmpEvent(i)
		# here you can add the usual logic, just *never* use continue
		
		# finally, compute scores in the end.
		DNN_score[0] = 0.5
		BDT_score[0] = 0.5
		
		# do not touch below
		fTree.Fill()
	dpch.Terminate()
	fTree.Write()
	fout.Write("",TObject.kOverwrite)
	fout.Close()

if __name__ == "__main__":
	main()
