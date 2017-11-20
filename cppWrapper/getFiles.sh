#!/bin/bash

wget -nv -N -O X_max.npy https://github.com/david-droz/DmpAnalysis/blob/master/cppWrapper/X_max.npy?raw=true
wget -nv -N -O trainedDNN.h5 https://github.com/david-droz/DmpAnalysis/blob/master/cppWrapper/trainedDNN.h5?raw=true
wget -nv -N -O calculateScore.py https://raw.githubusercontent.com/david-droz/DmpAnalysis/master/cppWrapper/calculateScore.py
wget -nv -N -O Classifier.cpp https://raw.githubusercontent.com/david-droz/DmpAnalysis/master/cppWrapper/Classifier.cpp
wget -nv -N -O Classifier.h https://raw.githubusercontent.com/david-droz/DmpAnalysis/master/cppWrapper/Classifier.h
wget -nv -N -O testWrapper.cpp https://raw.githubusercontent.com/david-droz/DmpAnalysis/master/cppWrapper/testWrapper.cpp
