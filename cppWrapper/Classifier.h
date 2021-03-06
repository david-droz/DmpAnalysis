/** Classifier
 * 
 * Purpose: Call an external Python script that contains machine learning related code. Given a DAMPE Event, the code returns a "score", a number to make the distinction between protons and electrons
 * 
 * Usage: First make sure to have the files "calculateScore.py", "X_max.npy", "trainedDNN.h5" in your working directory. 
 * 		In your analysis, initialise first the classifier:   Classifier c;
 * 		Then, call the method Classifier::getScore . This method returns a double. Arguments can be either a DmpEvent*, a DmpEvtBgoRec*, or a combination of vectors and doubles.
 * 		At the end of your code, call the method Classifier::finalize() to do some cleaning-up.
 * 
 * Example:
 * 		Classifier c;
 * 		double score = c.getScore(bgorec);
 * 		c.finalize()
 * 
 * 
 * 
 * List of variables for the DNN classifier:
 * 
 *  | Variable index	|	Variable name					| 	DmpSoftware function call
	------------------------------------------------------------------------------------
	|		0 - 13		|	Energy in BGO layer i			|	DmpChain->pEvtBgoRec()->GetELayer(i)
	|		14 - 27		|	RMS2 in individual BGO layers	|			---    		  ->GetRMS2()[i]
	|		28 - 41		|	Hits in individual BGO layers	|			---    		  ->GetLayerHits()[i]
	|		42			|	Longitudinal RMS				|			---    		  ->GetRMS_l()
	|		43			| 	Radial RMS						|			---    		  ->GetRMS_r()
	|		44			|	Total BGO energy				|			---    		  ->GetTotalEnergy()
	|		45			|	Total BGO hits					|			---    		  ->GetTotalHits() 
	|		46			|	XZ slope (angle calculation)	|			---    		  ->GetSlopeXZ()
	|		47			|	YZ slope (angle calculation)	|			---    		  ->GetSlopeYZ()
 * 
 * 
 *   
 * @author David Droz (david.droz@cern.ch)
 * @version 0.1
 * @date 2017-11-12
 */


#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <Python.h>

//@@ DMPSW includes
#include "DmpEvtBgoRec.h"
#include "DmpRootEvent.h"


#ifndef Classifier_H
#define Classifier_H




class Classifier {
	
	PyObject *pName, *pModule;
	
	PyObject *pFunc, *pTest;
	
	public:	
	
	/// Default constructor
	Classifier(void); 
	
	/// Constructor. Argument is the name of the Python file. Default is "calculateScore"
	Classifier(std::string moduleName);
	
	/// Given a DmpEvent, returns the neural network score calculated from the BGO signal
	double getScore(DmpEvent* evt);
	
	/// Returns the neural network score calculated from the BGO signal
	double getScore(DmpEvtBgoRec* bgorec);
	
	/// Returns the neural network score calculated from the raw values (vector of 48 elements)
	double getScore(std::vector <double> array);
	
	/// Returns the neural network score calculated from the raw values (array of 48 elements)
	double getScore(double array[48]);
	
	/// Returns the neural network score calculated from the raw values
	double getScore(double eneLayer[14],double rmsLayer[14], double hitsLayer[14], double longitudinalRMS, double radialRMS, double Etot, double hits, double XZslope, double YZslope);
	
	///Runs a small test. Ignore.
	double runTest(std::vector <double> array);
	
	/// Cleans-up the process. Must be called at the end of every code
	void finalize();
	
	
};












#endif
