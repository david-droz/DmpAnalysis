#include <vector>
#include <iostream>
#include "DmpEvtBgoRec.h"
#include "DmpRootEvent.h"
#include "DmpChain.h"
#include "Classifier.h"


int main(int argc, char **argv){
	
	Classifier c;
	
	std::vector <double> v;
	v.push_back(15);
	v.push_back(33);
	v.push_back(42);
	v.push_back(56);
	
	std::cout << c.runTest(v) << endl;
	
	std::vector <double> u;
	u.push_back(30);
	u.push_back(66);
	u.push_back(84);
	u.push_back(112);
	
	std::cout << c.runTest(u) << endl;
	
	std::cout << "Now for Dampe Events..." << endl;
	
	std::vector <double> w;
	for(unsigned int i(0);i<48;i++){
		w.push_back(100*i);
	}
	std::cout << c.getScore(w) << endl;
	
	//~ const char* file = "/dampe/data3/users/public/high_energy_events__trunk_r5202/5.4.2/fullBGO/allProton-v5r4p2_100GeV_10TeV_data3_p2_fullBGO/allProton-v5r4p2_100GeV_10TeV_data3_p2.noOrb.206001_207000.reco.root";
	
	//~ DmpChain *dmpch = new DmpChain("CollectionTree");
	//~ dmpch->Add(file);
	//~ long int nevents = dmpch->GetEntries();
	
	//~ DmpEvent* evt01 = dmpch->GetEvent();
	
	//~ std::cout << c.getScore(evt01) << endl;
	
	//~ DmpEvent* evt02 = dmpch->GetEvent();
	//~ DmpEvtBgoRec* bgorec = evt02->pEvtBgoRec();
	
	//~ std::cout << c.getScore(bgorec) << endl;
	
	c.finalize();
	
	
	return 0;
	
	
}
