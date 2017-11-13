#include <vector>
#include <iostream>
#include "Classifier.h"


void main(int argc, char **argv){
	
	Classifier c;
	
	std::vector <double> v;
	v.push_back(15);
	v.push_back(33);
	v.push_back(42);
	v.push_back(56);
	
	std::cout << c.runTest(v) << endl;
	
	c.finalize();
	
	
	
}
