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

from keras.models import load_model
from sklearn.externals import joblib

def main(args=None):
	usage = "Usage: %(prog)s [options]"
	description = "adding MVAtree to existing 2A file, will create a copy"
	parser = ArgumentParser(usage=usage, description=description)
	parser.add_argument("-i","--infile",dest='infile',type=str,default=None, help='name of input file',required=True)
	parser.add_argument("-o","--outfile",dest='outfile',type=str,default=None, help='name of output file',required=True)
	parser.add_argument("-m","--model",dest="model",type=str,default="model.model",help="name of Keras model file")
	parser.add_argument("-b","--bdt",dest="bdt",type=str,default="bdt.pick",help="name of sklearn BDT model file")
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
	# space to declare your variables
	BgoTotalE = zeros(nevts, dtype=float)
	# event loop to read out variables
	print 'read out events'
	for i in xrange(nevts):
		pev = dpch.GetDmpEvent(i)
		# here you can add the usual logic, just *never* use continue
		BgoTotalE[i] = pev.pEvtBgoRec().GetTotalEnergy()
	dpch.Terminate()
	# here comes some keras 'magic'
	# ...
	# i'm assuming you compute scores as DNN_sk & BDT_sk
	
	DNN = load_model(args.model)
	DNN_sk = DNN.predict(X_norm)
	
	BDT = joblib.load(args.bdt)
	BDT_sk = BDT.predict_proba(X)[:,1]
	
	# now loop again, creating scoring variables
	print 'store scores'
	for i in xrange(nevts):
		DNN_score[0] = DNN_sk[i]
		BDT_score[0] = BDT_sk[i]
		fTree.Fill()
	fTree.Write()
	fout.Write("",TObject.kOverwrite)
	fout.Close()

if __name__ == "__main__":
	main()
