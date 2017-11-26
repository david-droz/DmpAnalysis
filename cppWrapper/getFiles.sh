#!/bin/bash

wget -nv -N -O X_max.npy https://github.com/david-droz/DmpAnalysis/blob/master/cppWrapper/X_max.npy?raw=true
wget -nv -N -O trainedDNN.h5 https://github.com/david-droz/DmpAnalysis/blob/master/cppWrapper/trainedDNN.h5?raw=true
wget -nv -N -O calculateScore.py https://raw.githubusercontent.com/david-droz/DmpAnalysis/master/cppWrapper/calculateScore.py
wget -nv -N -O Classifier.cpp https://raw.githubusercontent.com/david-droz/DmpAnalysis/master/cppWrapper/Classifier.cpp
wget -nv -N -O Classifier.h https://raw.githubusercontent.com/david-droz/DmpAnalysis/master/cppWrapper/Classifier.h
wget -nv -N -O example_DmpEvent.cpp https://raw.githubusercontent.com/david-droz/DmpAnalysis/master/cppWrapper/example_DmpEvent.cpp
wget -nv -N -O example_ROOT.cpp https://raw.githubusercontent.com/david-droz/DmpAnalysis/master/cppWrapper/example_ROOT.cpp
wget -nv -N -O compile.sh https://raw.githubusercontent.com/david-droz/DmpAnalysis/master/cppWrapper/compile.sh
