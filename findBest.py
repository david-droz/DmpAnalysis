'''

Find the best results from the machine learning grid searching

'''

from glob import glob
import numpy as np

if __name__ == '__main__':
	
	runs = glob('./results/*/')
	
	l_precision = []
	l_recall = []
	l_ID = []
	
	for f in runs:
		
		arr = np.loadtxt(f,dtype=str)
		l_precision.append(float(arr[0,1]))
		l_recall.append(float(arr[1,1]))
		
		# Find ID
		s1 = f.replace('/','X',1).find('/') + 1
		s2 = f.replace('/','X',2).find('/')
		l_ID.append(f[s1:s2])
		
	best = max(l_recall)
	index = l_recall.index(best)
	
	print 'Precision: ', l_precision[index]
	print 'Recall: ', l_recall[index]
	print 'ID: ', l_ID[index]
