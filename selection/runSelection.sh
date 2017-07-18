#!/bin/bash

INFILE=
particle=

###
cd /dampe/data3/users/ddroz/analysis/selection_v4/

source /cvmfs/dampe.cern.ch/rhel6-64/etc/setup.sh
dampe_init

NLINES=$(cat $INFILE | wc -l)

PREFIX=${INFILE/".txt"/"-part"}
DIRNAME=${INFILE/".txt"/""}

split -l 10 -d $INFILE $PREFIX

mkdir $DIRNAME
mv ${PREFIX}* $DIRNAME

i=0
for f in $(ls ${DIRNAME}/*)
do
	python /dampe/data3/users/ddroz/analysis/DmpAnalysis/selection/selection.py ${f} ${particle} ${i}
	((i+=1))
done
