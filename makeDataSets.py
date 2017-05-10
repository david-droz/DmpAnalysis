'''

Run after selection.py

'''

import sys
import numpy as np









# Pick training/validation/testing set:
# Make a list of all available indexes for both electrons and protons
# Randomly choose indexes:
#		for i in xrange(training_size):
#			training_indexes.append(  random_elements_from_list_of_indexes  )
# Can use method "pop" to be sure to not repeat indexes
#
# This allows to select a number N of electrons and protons for all sets
# For the validation and testing sets, can use oversampling (bootstrapping ?) for protons


# Also don't forget to implement Stephan's method to turn that into a ROOT file


if __name__ == '__main__':
	
