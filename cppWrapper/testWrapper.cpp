/**
 * 
 * Small working example using DmpChain/DmpEvent package and the Classifier class for DNN-based analysis
 * 
 **/


#include <vector>
#include <iostream>
#include "DmpEvtBgoRec.h"
#include "DmpRootEvent.h"
#include "DmpChain.h"
#include "Classifier.h"


int main(int argc, char **argv){
	
	// Initialise
	Classifier c;
	
	// Storage for the scores. Can fill a Root histogram instead
	std::vector <double> scores;

	// Declare a DmpChain and add a file to it
	DmpChain *dmpch = new DmpChain("CollectionTree");
	
	// File 1 : 66k events (skimmed MC). Long!
	//dmpch->Add("/dampe/data3/users/public/high_energy_events__trunk_r5202/5.4.2/fullBGO/allProton-v5r4p2_100GeV_10TeV_data3_p2_fullBGO/allProton-v5r4p2_100GeV_10TeV_data3_p2.noOrb.206001_207000.reco.root");
	
	// File 2 : 32 events (skimmed flight data). Short.
	dmpch->Add("/dampe/data3/users/public/high_energy_events__trunk_r6512/6.0.0/data/2017/08/07_data_500_000.root");
	
	long int nevents = dmpch->GetEntries();
	std::cout << "Processing " << nevents << " events" << std::endl;
	
	DmpEvent *pev = DmpEvent::GetHead();
	for(long int event(0);event<nevents;event++){
		
		pev = dmpch->GetDmpEvent(event);
		
		// Get the score associated to that event
		scores.push_back( c.getScore(pev) );
		
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
