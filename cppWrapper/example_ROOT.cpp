/**
 * 
 * Small working example using standard ROOT analysis and the Classifier class for DNN-based analysis
 * 
 **/


#include <vector>
#include <iostream>

//ROOT includes
#include <TROOT.h>
#include <TSystem.h>
#include <TFile.h>
#include <TTree.h>
#include <TClonesArray.h>
#include <TChain.h>
#include <TBranch.h>
#include "DmpEvtBgoRec.h"

#include "Classifier.h"

int main(int argc, char **argv){
	
	TFile* f = new TFile("/dampe/data3/users/public/high_energy_events__trunk_r6512/6.0.0/data/2017/08/07_data_500_000.root","READ");
	TTree* t = (TTree*)f->Get("CollectionTree");
	
	DmpEvtBgoRec* bgorec = new DmpEvtBgoRec();
	t->SetBranchAddress("DmpEvtBgoRec",&bgorec);
	
	// Initialise
	Classifier c;
	
	// Storage for the scores. Can fill a Root histogram instead
	std::vector <double> scores;
	
	long int nevents = t->GetEntries();
	std::cout << "Processing " << nevents << " events" << std::endl;
	
	for(long int event(0);event<nevents;event++){
		
		t->GetEntry(event);
		
		// Get the score associated to that event
		double score = c.getScore(bgorec);
		scores.push_back( score );
		
	}

	std::cout << "Analysed " << scores.size() << " events" << std::endl;
	
	// Print a few scores:
	for(unsigned int i(0);i<4;i++){
		std::cout << scores[i] << std::endl;
	}
	// These scores were cross-checked with usual Python analysis and are consistent
	
	// Important! Close the Python interpreter
	c.finalize();
	
	
	return 0;
	
}
