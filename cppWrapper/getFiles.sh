#!/bin/bash

wget -nv -O X_max.npy https://github.com/david-droz/DmpAnalysis/blob/master/cppWrapper/X_max.npy?raw=true
wget -nv -O trainedDNN.h5 https://github.com/david-droz/DmpAnalysis/blob/master/cppWrapper/trainedDNN.h5?raw=true
wget -nv https://raw.githubusercontent.com/david-droz/DmpAnalysis/master/cppWrapper/calculateScore.py
wget -nv https://raw.githubusercontent.com/david-droz/DmpAnalysis/master/cppWrapper/Classifier.cpp
wget -nv https://raw.githubusercontent.com/david-droz/DmpAnalysis/master/cppWrapper/Classifier.h
