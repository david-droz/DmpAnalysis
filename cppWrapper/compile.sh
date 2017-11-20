#!/bin/bash

g++ \
 -I/cvmfs/dampe.cern.ch/rhel6-64/opt/externals/python2.7/latest/build/include/python2.7 \
 -I/cvmfs/dampe.cern.ch/rhel6-64/opt/releases/latest/include \
 -I/cvmfs/dampe.cern.ch/rhel6-64/opt/externals/root/latest/include \
 -L/cvmfs/dampe.cern.ch/rhel6-64/opt/releases/latest/lib \
 -l DmpEvent \
 -o classifier Classifier.cpp \
 -fno-strict-aliasing -g -O2 -DNDEBUG -g -fwrapv -O3 -Wall -lpython2.7 testWrapper.cpp
