import numpy as np

#ET asds to consider
asd_names = ['ET10km','ET15km','ET20km']
#ET configuration each column in asd_name+'columns.txt' corresponds to
configurations=['HF', 'LF', 'HFLF']

#loop over asd names
for asd_name in asd_names:

	#load the psd
	psd_file = np.loadtxt(asd_name+'columns.txt')

	#loop over ET configurations
	for iconf, conf in enumerate(configurations):
		
		#save the array corresponding to the desired configuration
		np.savetxt(asd_name+conf+'.txt', np.transpose([psd_file[:,0], np.sqrt(psd_file[:,1+iconf])]))
