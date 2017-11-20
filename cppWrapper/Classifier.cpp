#include "Classifier.h"

Classifier::Classifier(void){
	Py_Initialize();
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append(\".\")");
	
	pName = PyString_FromString("calculateScore");
	
	pModule = PyImport_Import(pName);
	Py_DECREF(pName);
	
	if (pModule == NULL){
		PyErr_Print();
		//fprintf(stderr, "Failed to load Python module: \"%s\"\n", "calculateScore");
		throw std::runtime_error("Failed to load Python module: \"calculateScore\"\n");
	}
	
	pFunc = PyObject_GetAttrString(pModule, "calculateScore");
	pTest = PyObject_GetAttrString(pModule, "testMethod");
	if (!pFunc or !PyCallable_Check(pFunc)) {
		if (PyErr_Occurred()){
				PyErr_Print();
			}
		//fprintf(stderr, "Cannot find Python function \"%s\"\n", "calculateScore");
		throw std::runtime_error("Cannot find Python function \"calculateScore\"\n");
	}
	if (!pTest or !PyCallable_Check(pTest)) {
		if (PyErr_Occurred()){
				PyErr_Print();
			}
		//fprintf(stderr, "Cannot find Python function \"%s\"\n", "testMethod");
		throw std::runtime_error("Cannot find Python function \"testMethod\"\n");
	}
}

Classifier::Classifier(std::string moduleName){
	Py_Initialize();
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append(\".\")");
	
	const std::string ext(".py");
	if (moduleName.find(ext) != std::string::npos){
		moduleName = moduleName.substr(0, moduleName.size() - ext.size());
	}
	
	pName = PyString_FromString(moduleName.c_str());
	
	pModule = PyImport_Import(pName);
	Py_DECREF(pName);
	
	if (pModule == NULL){
		PyErr_Print();
		//~ fprintf(stderr, "Failed to load Python module: \"%s\"\n", "calculateScore");
		throw std::runtime_error("Failed to load Python module: \"calculateScore\"\n");
	}
	pFunc = PyObject_GetAttrString(pModule, "calculateScore");
	pTest = PyObject_GetAttrString(pModule, "testMethod");
	if (!pFunc or !PyCallable_Check(pFunc)) {
		if (PyErr_Occurred()){
				PyErr_Print();
			}
		//fprintf(stderr, "Cannot find Python function \"%s\"\n", "calculateScore");
		throw std::runtime_error("Cannot find Python function \"calculateScore\"\n");
	}
	if (!pTest or !PyCallable_Check(pTest)) {
		if (PyErr_Occurred()){
				PyErr_Print();
			}
		//fprintf(stderr, "Cannot find Python function \"%s\"\n", "testMethod");
		throw std::runtime_error("Cannot find Python function \"testMethod\"\n");
	}
}


double Classifier::runTest(std::vector <double> array){
	PyObject *pArgs, *pValue;
	PyObject *PList = PyList_New(0);
	
	std::vector<double>::iterator it;
	for(it = array.begin(); it != array.end() ; it++ ){
		PyList_Append(PList, Py_BuildValue("d", *it));
	}
	
	pArgs = PyTuple_New(1);
	PyTuple_SetItem(pArgs, 0, PList);
		
	pValue = PyObject_CallObject(pTest, pArgs);
	
	Py_DECREF(pArgs);
	if (pValue == NULL){
		Py_DECREF(pTest);
		Py_DECREF(pModule);
		PyErr_Print();
		throw std::runtime_error("Calling Python testMethod failed");
	}
	return PyFloat_AsDouble(pValue);
	
	// Some test that can be done: make sure that Keras is loaded only once, and not at every call.
	
}


// Then implement the various methods for getting the classifier score
double Classifier::getScore(DmpEvent* evt){
	DmpEvtBgoRec* prec = evt->pEvtBgoRec();
	return getScore(prec);
}
double Classifier::getScore(DmpEvtBgoRec* bgorec){
	std::vector <double> values;
	
	for(unsigned int i(0);i<14;i++){
		values.push_back((double)bgorec->GetELayer(i));
	}
	
	// Need to know the format of bgorec->GetRMS2() and bgorec->GetLayerHits(). Arrays of size 14? How to declare?
	// double RMS2[14] = bgorec->GetRMS2()  ????
	
	float* rms2 = bgorec->GetRMS2();
	int* hits = bgorec->GetLayerHits();
	
	for(unsigned int i(0);i<14;i++){
		values.push_back((double)rms2[i]);
	}
	for(unsigned int i(0);i<14;i++){
		values.push_back((double)hits[i]);
	}
	
	double rms_l = bgorec->GetRMS_l();
	double rms_r = bgorec->GetRMS_r();
	double Ecor = bgorec->GetTotalEnergy();
	double totalHits = (double)bgorec->GetTotalHits();
	double slope_xz = bgorec->GetSlopeXZ();
	double slope_yz = bgorec->GetSlopeYZ();
	
	values.push_back(rms_l);
	values.push_back(rms_r);
	values.push_back(Ecor);
	values.push_back(totalHits);
	values.push_back(slope_xz);
	values.push_back(slope_yz);
	
	return getScore(values);
}

double Classifier::getScore(double eneLayer[14],double rmsLayer[14], double hitsLayer[14], double longitudinalRMS, double radialRMS, double Etot, double hits, double XZslope, double YZslope){
	std::vector <double> values;
	
	for(unsigned int i(0);i<14;i++){
		values.push_back(eneLayer[i]);
	}
	for(unsigned int i(0);i<14;i++){
		values.push_back(rmsLayer[i]);
	}
	for(unsigned int i(0);i<14;i++){
		values.push_back(hitsLayer[i]);
	}
	
	// There has to be a smarter way of doing what I'm doing here...
	values.push_back(longitudinalRMS);
	values.push_back(radialRMS);
	values.push_back(Etot);
	values.push_back(hits);
	values.push_back(XZslope);
	values.push_back(YZslope);
	
	return getScore(values);
}

double Classifier::getScore(double array[48]){
	std::vector <double> values;
	
	for(unsigned int i(0); i<48; i++){
		values.push_back(array[i]);
	}
	
	return getScore(values);
}

double Classifier::getScore(std::vector <double> array){
	
	if (array.size() != 48){
		std::cerr << "Warning! Vector given to Classifier::getScore appears to have wrong size (expected: 48)" << endl;
	}
	
	
	PyObject *pArgs, *pValue;
	PyObject *PList = PyList_New(0);
	
	std::vector<double>::iterator it;
	for(it = array.begin(); it != array.end() ; it++ ){
		PyList_Append(PList, Py_BuildValue("d", *it));
	}
	
	pArgs = PyTuple_New(1);
	PyTuple_SetItem(pArgs, 0, PList);
		
	pValue = PyObject_CallObject(pFunc, pArgs);
	
	Py_DECREF(pArgs);
	if (pValue == NULL){
		Py_DECREF(pFunc);
		Py_DECREF(pModule);
		PyErr_Print();
		throw std::runtime_error("Calling Python calculateScore failed");
	}
	return PyFloat_AsDouble(pValue);
}

// Finally implement the Finalize() method.

void Classifier::finalize(){
	Py_XDECREF(pFunc);
	Py_XDECREF(pTest);
    Py_DECREF(pModule);
    
	Py_Finalize();
}
