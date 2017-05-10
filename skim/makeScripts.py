from os import getcwd

def makeScript(out,i,entree,part,outdir,begin,end):

	scriptname = out + '_' + str(i) + '.txt'
	tmpfile = out + '_' + str(i) + '.temp'

	with open(scriptname,'w') as f:

		f.write('#!/bin/bash\n')
		f.write(' \n')	
		cwd = getcwd()
			
		f.write('export HOME=/atlas/users/ddroz/\n')
		f.write(' \n')	
		f.write('cd ' + str(cwd) + '\n')
		f.write(' \n')	
		f.write('source /cvmfs/dampe.cern.ch/rhel6-64/etc/setup.sh\n')
		f.write(' \n')
		f.write('dampe_init\n')
		f.write('\n')
		f.write('head -n ' + str(begin) + ' ' + str(entree) + ' | tail -n ' + str(end) + ' > ' + tmpfile + '\n')
		f.write(' \n')
		command = 'python skim.py ' + tmpfile + ' -p ' + str(part) + ' -o ' + str(outdir)
		f.write(command + '\n')
		#f.write('rm ' + tmpfile + '\n')



for entree in ['allElectron_100G-10T-p2.txt','allElectron_100G-10T.txt','allProton_100G-10T-p2.txt','allProton_100G-10T-p3.txt']:
	filelist = []	
	with open(entree,'r') as f:
		for lines in f:
			filelist.append(lines.replace('\n',''))

	if 'Elec' in entree:
		out = 'scripts/s_electron'
		part = 'electron'
		outdir = 'skimmed_electron'
	elif "Prot" in entree:
		out = 'scripts/s_proton'
		part = 'proton'
		outdir = 'skimmed_proton'
	for x in ['-p2','-p3']:
		if x in entree:
			out = out + x
			outdir = outdir + x

	subdivision = len(filelist)/50

	for i in range(50):

		begin = (i+1)*subdivision
		end = subdivision

		makeScript(out,i,entree,part,outdir,begin,end)
	
	if 50*subdivision != len(filelist):
		begin = len(filelist)
		end = subdivision
		
		makeScript(out,50,entree,part,outdir,begin,end)
