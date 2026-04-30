import numpy as np

#necessary stuff to locate ASDs
path_2_ASD = './'
ASD_names = ['AplusDesign', 'avirgo_O5high_NEW']
ASD_suff = '_ASD.txt'

#output name format
PSD_suff = '_PSD.txt'

#loop over ASDs
for ASD_name in ASD_names:
    #load the asd
    freqs_PSD = np.loadtxt(path_2_ASD+'/'+ASD_name+ASD_suff)
    #compute the PSD as the square of the ASD
    freqs_PSD[:,1] = np.square(freqs_PSD[:,1])
    #save the psd
    np.savetxt(path_2_ASD+'/'+ASD_name+PSD_suff, freqs_PSD)

