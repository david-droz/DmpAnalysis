#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <Python.h>

//@@ DMPSW includes
#include "DmpEvtBgoHits.h"
#include "DmpEvtBgoRec.h"
#include "DmpRootEvent.h"
#include "DmpChain.h"

//@@ ROOT includes
#include <TROOT.h>
#include <TSystem.h>
#include <TFile.h>
#include <TTree.h>
#include <TClonesArray.h>
#include <TChain.h>
#include <TBranch.h>
#include <TDirectory.h>
#include <TMath.h>


#ifndef Classifier
#define Classifier




class Classifier{
	
	PyObject *pName, *pModule;
	
	PyObject *pFunc, pTest;
	
	public:	
	
	Classifier(); 
	Classifier(std::string moduleName);
	
	double getScore(DmpEvent* evt);
	double getScore(DmpEvtBgoRec* bgorec);
	double getScore(std::vector <double> array);
	double getScore(std::vector <double> eneLayer,std::vector <double> rmsLayer, std::vector <double> hitsLayer, double longitudinalRMS, double radialRMS, double EtotCorrected, double hits, double XZslope, double YZslope);
	double runTest(std::vector <double> array);
	void Finalize();
	
	
};












#endif
