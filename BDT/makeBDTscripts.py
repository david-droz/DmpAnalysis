import os

def _writescript(f,n_est,lr,mxdpth,leaf):

	f.write('#!/bin/env bash\n')

	f.write('#SBATCH --ntasks=1\n')

	f.write('#SBATCH --time=02:00:00\n')
	f.write('#SBATCH --partition=shared\n')

	f.write('#SBATCH --constraint="V3|V4|V5|V6"\n')

	f.write('\n')
	f.write('\n')
	f.write('module load GCC/4.9.3-2.25 OpenMPI/1.10.2 Python/2.7.11\n')
	f.write('\n')
	f.write('source ~/BDT/BDT_env/bin/activate')
	f.write('\n')
	f.write('srun python /home/drozd/BDT/BDT.py ' + str(n_est) + ' ' + str(lr) + ' ' + str(mxdpth) + ' ' + str(leaf))
	
	

if __name__ == '__main__' :
	
	estimators = [10,30,100,300,1000,3000]
	learning_rate = [1,0.3,0.1,0.03,0.01,0.001]
	max_depth = [1,2,3,4,5,6]
	leaves = [1,10,100,1000,10000]
	
	if not os.path.isdir('scripts'):
		os.mkdir('scripts')
	
	for e in estimators:
		for l in learning_rate:
			for m in max_depth:
				for le in leaves:
					
					scriptname = 'scripts/run_' + str(e) + '_' + str(l) + '_' + str(m) + '_' + str(le) + '.sh'
					with open(scriptname,'w') as f:
						_writescript(f,e,l,m,le)
					

