#include <Python.h>
#include <vector>
#include <iostream>

std::vector <double> getValues(){
	
	// Function that builds a vector of double, containing the values to be fed to DNN
	
	unsigned int size = 6;
	
	std::vector <double> v(size);
	
	for (unsigned int i(0);i<size;i++){
		v[i] = i;
	}
	
	return v;
	
	
}

double getScore(std::vector <double> v){
	
	// Function that, given a suitable vector of double, returns the corresponding DNN score
	
	Py_Initialize();
	
	PyObject *pName, *pModule, *pFunc;
	PyObject *pArgs, *pValue;
	PyObject *PList = PyList_New(0);
	
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append(\".\")");
	
	
	//~ for(unsigned int i(0); i < v.size() ; i++){
		//~ PyList_Append(PList, Py_BuildValue("i", v[i]));
	//~ }
	
	std::vector<double>::iterator it;
	for(it = v.begin(); it != v.end() ; it++ ){
		PyList_Append(PList, Py_BuildValue("d", *it));
	}
  
	pName = PyString_FromString("calculateScore");
	pModule = PyImport_Import(pName);
	Py_DECREF(pName);
	
	if (pModule == NULL){
		PyErr_Print();
		fprintf(stderr, "Failed to load \"%s\"\n", "calculateScore");
		return -1;
	}
	
	pFunc = PyObject_GetAttrString(pModule, "testMethod");
	if (pFunc && PyCallable_Check(pFunc)) {
		pArgs = PyTuple_New(1);
		PyTuple_SetItem(pArgs, 0, PList);
		
		pValue = PyObject_CallObject(pFunc, pArgs);
		
		Py_DECREF(pArgs);
		
		if (pValue == NULL){
			Py_DECREF(pFunc);
			Py_DECREF(pModule);
			PyErr_Print();
			fprintf(stderr,"Call failed\n");
			return -1;
		}
		
		
	}
	else {
		if (PyErr_Occurred()){
				PyErr_Print();
			}
		fprintf(stderr, "Cannot find function \"%s\"\n", "testMethod");
		return -1;
	}
    Py_XDECREF(pFunc);
    Py_DECREF(pModule);
    
	Py_Finalize();
	
	return PyFloat_AsDouble(pValue);
}


int main(int argc, char *argv[]){
	
	std::cout << argv[0] << std::endl;
	
	Py_SetProgramName(argv[0]);
	
	double my_score(0);
	
	my_score = getScore(getValues());
	
	std::cout << my_score << std::endl;
	
	
	return 0;
}
