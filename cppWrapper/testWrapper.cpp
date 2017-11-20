#include <vector>
#include <iostream>
#include "DmpEvtBgoRec.h"
#include "DmpRootEvent.h"
#include "DmpChain.h"
#include "Classifier.h"


int main(int argc, char **argv){
	
	// Initialisation
	Classifier c;
	
	std::vector <double> v;
	v.push_back(15);
	v.push_back(33);
	v.push_back(42);
	v.push_back(56);
	
	// Run a first test
	std::cout << c.runTest(v) << endl;
	
	std::vector <double> u;
	u.push_back(30);
	u.push_back(66);
	u.push_back(84);
	u.push_back(112);
	
	// Run a second test. Purpose was to see if all libraries were still in memory or if the Python interpreter had to reload everything
	// Result: doesn't have to, the second test is much faster than the first one
	std::cout << c.runTest(u) << endl;
	
	
	//std::cout << "Now for Dampe Events..." << endl;
	
	
	// Building a random vector and calling the classifier on it. Will return a meaningless number, this is only to see
	// if the method and wrapper work
	std::vector <double> w;
	for(unsigned int i(0);i<48;i++){
		w.push_back(100*i);
	}
	std::cout << c.getScore(w) << endl;
	
	c.finalize();
	
	
	return 0;
	
	
}
