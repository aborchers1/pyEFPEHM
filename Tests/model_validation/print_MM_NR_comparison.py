import numpy as np
import pickle

#######################################################################################

#path where the runs are located
outdir = 'outdir/NR_comparisons/'

#name to identify the run
name_ID = '%s_vs_SXSBBH%s_Nconf_100_fref_19_fmin_20_fmax_0.8ISCO_De_0.4_Dp_0.1_AplusDesign'

#Runs to study
sxs_nums = ['2619', '2621', '2538', '2549', '2561', '3951', '0088', '4286', '4290']

#waveforms to study
approx_strings = ['pyEFPEHM', 'pyEFPE', 'TEOBResumS', 'SEOBNRv5EHM', 'SEOBNRv5PHM']

#format to output numbers
output_format = '%.2g'
blanck_str = '----'

#######################################################################################

#initialize strings to put tables in
MM_strs = ['Avg mismatches', 'Max mismatches']

#in the first line we put the name of the approximants
MM_strs = [MM_str + '\n ' for MM_str in MM_strs]
for approx_string in approx_strings: MM_strs = [MM_str + '  &\t' + approx_string for MM_str in MM_strs]

#loop over sxs events
for isxs, sxs_num in enumerate(sxs_nums):

	#add a new line and the the event at the beginning of each line
	MM_strs = [MM_str + '\n' + sxs_num for MM_str in MM_strs]

	#loop over approximants
	for iapprox, approx_string in enumerate(approx_strings):
		
		#Add separator between columns
		MM_strs = [MM_str + '\t&\t' for MM_str in MM_strs]
		
		#reconstruct filename for this case
		filename = outdir+'/result_'+name_ID%(approx_string, sxs_num)+'.pickle'
		
		#try to open the result file
		try:
			with open(filename, 'rb') as handle: result = pickle.load(handle)
			
			#add string with average mismatch
			MM_strs[0] += '%s'%(output_format%(result['MM_avg']))
			#add string with maximum mismatch
			MM_strs[1] += '%s'%(output_format%(np.amax(result['min_mm'])))

		#if result file could not be loaded leave mismatch blank
		except:
			MM_strs = [MM_str + blanck_str for MM_str in MM_strs]

#print strings
for MM_str in MM_strs: print('\n' + MM_str + '\n')
