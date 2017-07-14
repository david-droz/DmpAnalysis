#!/bin/bash

INFILE=
particle=

###
cd /dampe/data3/users/ddroz/analysis/selection_v3/

source /cvmfs/dampe.cern.ch/rhel6-64/etc/setup.sh
dampe_init

i=0
for f in $(cat $INFILE)
do
	python /dampe/data3/users/ddroz/analysis/DmpAnalysis/selection/selection.py ${f} ${particle} ${i}
	((i+=1))
done
