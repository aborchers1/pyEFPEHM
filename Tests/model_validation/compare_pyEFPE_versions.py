import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

import pyEFPE
import pyEFPEHM
from utils_compute_mismatches import *

import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({
        'axes.grid': False,
        'axes.labelsize': 28,
        'axes.linewidth': 1.75,
        'axes.titlesize': 32,
        'font.size': 24,
        'legend.fontsize': 24,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'font.family': 'serif',
        'font.sans-serif': ['Bitstream Vera Sans'],
        'font.serif': ['Times New Roman'],
        'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts}',
        'text.usetex':True,
        'patch.force_edgecolor':True,})

import time
start_runtime = time.time()

#universal constants
t_sun_s = 4.92549094831e-6  #GMsun/c**3 [s]

#store the different comparisons
comparison_info = {}

comparison_info["equalized_pyEFPEHM_vs_pyEFPE"] = dict(
pyEFPE_1 = pyEFPEHM,
pyEFPE_2 = pyEFPE,
labels = ['pyEFPEHM', 'pyEFPE'],
params_pyEFPE_1 = {'pn_amplitude_order': 0,
                   'pn_spin_order': 4,
                   'pn_phase_order': 6,
                   'SUA_kmax': 1,
                   'Amplitude_tol': 1e-4,
                   },
update_params_pyEFPE_2 = {},
)

comparison_info["default_pyEFPEHM_vs_pyEFPE"] = dict(
pyEFPE_1 = pyEFPEHM,
pyEFPE_2 = pyEFPE,
labels = ['pyEFPEHM', 'pyEFPE'],
params_pyEFPE_1 = {},
update_params_pyEFPE_2 = {},
)

comparison_info["Amps_0PN_vs_Amps_1PN"] = dict(
pyEFPE_1 = pyEFPEHM,
pyEFPE_2 = pyEFPEHM,
labels = ['Amps_0PN', 'Amps_1PN'],
params_pyEFPE_1 = {'pn_amplitude_order': 0,},
update_params_pyEFPE_2 = {'pn_amplitude_order': 2,},
)

#Chirp mass bounds as a function of segment length
mc_low_of_seglen  = {4: 12, 8:  8, 16: 5, 32: 3.3, 64: 2.2, 128: 1.4, 256: 0.95}
mc_high_of_seglen = {4: 20, 8: 12, 16: 8, 32:   5, 64: 3.3, 128: 2.2, 256:  1.4}

############################################################################

#function to evaluate and time pyEFPE using multiple tries
#we pick the timings of the case with the smallest total evaluation time
def evaluate_and_time_pyEFPE(pyEFPE_class, params, Ntries=5):
	
	#construct arrays to place the runtimes
	ini_runtimes, h_runtimes = np.zeros(Ntries), np.zeros(Ntries)
	
	#loop over number of tries
	for i in range(Ntries):
		#initialize waveform class
		t0 = time.time()
		wf = pyEFPE_class(params)
		tf = time.time()
		ini_runtimes[i] = tf - t0
		#compute polarizations
		t0 = time.time()
		hp, hc = wf.generate_waveform(freqs)
		tf = time.time()
		h_runtimes[i] = tf - t0
	
	#compute total time
	tot_runtimes = ini_runtimes + h_runtimes
	
	#find for what realization the total time is the smallest
	imin = np.argmin(tot_runtimes)
	
	#return polarizations and timings
	return (hp, hc), (ini_runtimes[imin], h_runtimes[imin], tot_runtimes[imin])

############################################################################

#duration to use to compute delta_f = 1/seglen
seglen = 4

#Setting up boundary conditions for random parameter generation
q_low = 0.05
q_high = 1
ecc_low = 0.0
ecc_high = 0.4
s1_low, cths1_low, phs1_low = 0, -1, 0
s1_high, cths1_high, phs1_high = 0.9, 1, 2.0*np.pi
s2_low, cths2_low, phs2_low = 0, -1, 0
s2_high, cths2_high, phs2_high = 0.9, 1, 2.0*np.pi
ciota_low = -1
ciota_high = 1
phiref_low = 0
phiref_high = 2*np.pi
mean_anomaly_low = 0
mean_anomaly_high = 2*np.pi
pol_low = 0
pol_high = 2*np.pi
f_min_gen_fact = 0.8 #fraction of fmin to start waveform generation from

#setting frequency array to call the waveform
f_min = 20
f_max = 4096
distance_Mpc = 10  # 10 Mpc is default 

#psd to use
psd_name = 'AplusDesign'

#Number of waveforms to test
N_tests = 2000
#Number of tries for timings
Ntries_timings = 5

#type of comparison
comparison_name = "equalized_pyEFPEHM_vs_pyEFPE"

#output directory
outdir = './outdir/pyEFPE_version_comparisons/'
plotdir = outdir+'/Plots'

#Number of top mismatches to print
N_top_MM_print = 10
#Number of top runtimes to print
N_top_runtime_print = 5

#number of bins for histograms
Nbins=100

############################################################################

#extract parameters
mc_low  =  mc_low_of_seglen[seglen]
mc_high = mc_high_of_seglen[seglen]

#default parameters for pyEFPE
params_pyEFPE_1 = comparison_info[comparison_name]['params_pyEFPE_1']
params_pyEFPE_1['distance'] = distance_Mpc
params_pyEFPE_1['f22_start'] = f_min_gen_fact*f_min

#parameters to update in pyEFPE_2
update_params_pyEFPE_2 = comparison_info[comparison_name]['update_params_pyEFPE_2']

#labels for pyEFPE_1 and pyEFPE_2
labels = comparison_info[comparison_name]['labels']

#extract the different pyEFPEs
pyEFPE_1 = comparison_info[comparison_name]['pyEFPE_1']
pyEFPE_2 = comparison_info[comparison_name]['pyEFPE_2']

#compute params low and high
params_low = np.array([mc_low, q_low, ecc_low, s1_low, cths1_low, phs1_low, s2_low, cths2_low, phs2_low, ciota_low, phiref_low, mean_anomaly_low, pol_low])
params_high = np.array([mc_high, q_high, ecc_high, s1_high, cths1_high, phs1_high, s2_high, cths2_high, phs2_high, ciota_high, phiref_high, mean_anomaly_high, pol_high])

#compute delta_f from seglen
delta_f = 1/seglen

#initialize frequencies
freqs = np.arange(f_min, f_max, delta_f)

#name of the result
string_id = '_compare_pyEFPE_%s_N_%s_Mc_%.4g_%.4g_seglen_%s'%(comparison_name, N_tests, mc_low, mc_high, seglen)

#if required, compute the ASD
if type(psd_name)==str:
	asd = compute_asd(delta_f, f_min, f_max, psd_name=psd_name)
	string_id += '_psd_'+psd_name
else:
	asd = None

#try to load the result
try:
	#load the result dictionary
	with open(outdir+'/result'+string_id+'.pickle', 'rb') as handle: result = pickle.load(handle)
	
	#make sure it is the same
	assert np.all(result['freqs']==freqs)
	assert np.all(result['params_low']==params_low)
	assert np.all(result['params_high']==params_high)
	assert len(result['MM'])==N_tests
#otherwise, create the result
except:
	#make sure directories to save stuff exist
	if not os.path.exists(outdir): os.makedirs(outdir)
	if not os.path.exists(plotdir): os.makedirs(plotdir)

	#make random parameters
	rand_params = np.random.uniform(params_low, params_high, size=(N_tests,len(params_low)))

	#generate random waveforms and compute their mismatches
	print('Seglen = %ss,  f_min = %sHz'%(seglen, f_min))
	mismatches = np.zeros(N_tests)
	all_params_pyEFPE_1 = []
	wf_run_times = np.full((N_tests, 6), np.nan)
	for iparam, [mc, q, e, s1, cths1, phs1, s2, cths2, phs2, ciota, phiref, mean_anomaly, pol] in enumerate(rand_params):
		
		#compute component masses in solar masses
		m1 = mc*(q**-0.6)*((1 + q)**0.2)
		m2 = m1*q
		
		#compute inclination
		iota = np.arccos(ciota)
		
		#compute cartesian spin components
		s1x, s1y, s1z = spherical_to_cartesian(np.array([s1, np.arccos(cths1), phs1]))
		s2x, s2y, s2z = spherical_to_cartesian(np.array([s2, np.arccos(cths2), phs2]))
		
		#print mass information
		print('\n%s/%s -> mc=%.3g, q=%.2f, e=%.2g, s1z=%.2g, s1p=%.2g, s2z=%.2g, s2p=%.2g'%(iparam+1, N_tests, mc, q, e, s1z, (s1x**2 + s1y**2)**0.5, s2z, (s2x**2 + s2y**2)**0.5))
		
		#put the params to pass to pyEFPE in dictionary
		params_pyEFPE_1['mass1'] = m1
		params_pyEFPE_1['mass2'] = m2
		params_pyEFPE_1['e_start'] = e
		params_pyEFPE_1['spin1x'] = s1x
		params_pyEFPE_1['spin1y'] = s1y
		params_pyEFPE_1['spin1z'] = s1z
		params_pyEFPE_1['spin2x'] = s2x
		params_pyEFPE_1['spin2y'] = s2y
		params_pyEFPE_1['spin2z'] = s2z
		params_pyEFPE_1['inclination'] = iota
		params_pyEFPE_1['phi_start'] = phiref
		params_pyEFPE_1['mean_anomaly_start'] = mean_anomaly
		all_params_pyEFPE_1.append(params_pyEFPE_1.copy())
		
		#update necessary params for pyEFPE_2
		params_pyEFPE_2 = params_pyEFPE_1.copy()
		params_pyEFPE_2.update(update_params_pyEFPE_2)
		
		#evaluate pyEFPE waveforms
		(hp_1, hc_1), (ini_runtime_1, h_runtime_1, tot_runtime_1) = evaluate_and_time_pyEFPE(pyEFPE_1.pyEFPE, params_pyEFPE_1, Ntries=Ntries_timings)
		(hp_2, hc_2), (ini_runtime_2, h_runtime_2, tot_runtime_2) = evaluate_and_time_pyEFPE(pyEFPE_2.pyEFPE, params_pyEFPE_2, Ntries=Ntries_timings)
		
		#save runtimes
		wf_run_times[iparam] = [ini_runtime_1, h_runtime_1, tot_runtime_1, ini_runtime_2, h_runtime_2, tot_runtime_2]
			
		#print timing info
		print('Ini time: %s: %.3gs\t %s: %.3gs\t Ratio: %.3g'%(labels[0], ini_runtime_1, labels[1], ini_runtime_2, ini_runtime_2/ini_runtime_1))
		print('h time:   %s: %.3gs\t %s: %.3gs\t Ratio: %.3g'%(labels[0],   h_runtime_1, labels[1],   h_runtime_2,   h_runtime_2/h_runtime_1))
		print('Tot time: %s: %.3gs\t %s: %.3gs\t Ratio: %.3g'%(labels[0], tot_runtime_1, labels[1], tot_runtime_2, tot_runtime_2/tot_runtime_1))
	
		#compute mismatches
		mismatches[iparam], dt_shift, ang_sh, amp_sh = mismatch_t_ph_amp_minimized(pol_response(hp_1, hc_1, pol), pol_response(hp_2, hc_2, pol), freqs, asd=asd)
		print('MM: %.3g'%(mismatches[iparam]))
	
	#make result dictionary
	result = {'MM': mismatches,
	          'run_times': wf_run_times,
	          'params': rand_params, 'params_pyEFPE': all_params_pyEFPE_1, 'update_params_pyEFPE_2': update_params_pyEFPE_2,
	          'freqs': freqs, 'params_low': params_low, 'params_high': params_high,
	          }
	
	#save the result file
	with open(outdir+'/result'+string_id+'.pickle', 'wb') as handle: pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

#If there are any nan mismatches, flag them
if np.any(np.isnan(result['MM'])):
	print('\nRealizations for which mismatch in Nan\n')
	for iMM in np.nonzero(np.isnan(result['MM']))[0]:
		print('\nMismatch: %.4g'%(result['MM'][iMM]))
		print('params =', result['params_pyEFPE'][iMM])

print('\n###########################################################')
print('Top %s realizations with largest mismatch'%(N_top_MM_print))
print('###########################################################')
#print the parameters associated with the top ones
for iMM in np.argsort(-result['MM'])[:N_top_MM_print]:
	print('\nMismatch: %.4g'%(result['MM'][iMM]))
	print('Ini time: %s: %.3gs\t %s: %.3gs\t Ratio: %.3g'%(labels[0], result['run_times'][iMM,0], labels[1], result['run_times'][iMM,3], result['run_times'][iMM,3]/result['run_times'][iMM,0]))
	print('h time:   %s: %.3gs\t %s: %.3gs\t Ratio: %.3g'%(labels[0], result['run_times'][iMM,1], labels[1], result['run_times'][iMM,4], result['run_times'][iMM,4]/result['run_times'][iMM,1]))
	print('params =', result['params_pyEFPE'][iMM])

for ilabel in range(2):
	print('\n###########################################################')
	print('Top %s %s realizations with largest runtime'%(N_top_runtime_print, labels[ilabel]))
	print('###########################################################')
	for ir in np.argsort(-result['run_times'][:,2+3*ilabel])[:N_top_runtime_print]:
		print('\nMismatch: %.4g'%(result['MM'][ir]))
		print('Ini time: %s: %.3gs\t %s: %.3gs\t Ratio: %.3g'%(labels[0], result['run_times'][ir,0], labels[1], result['run_times'][ir,3], result['run_times'][ir,3]/result['run_times'][ir,0]))
		print('h time:   %s: %.3gs\t %s: %.3gs\t Ratio: %.3g'%(labels[0], result['run_times'][ir,1], labels[1], result['run_times'][ir,4], result['run_times'][ir,4]/result['run_times'][ir,1]))
		print_params = result['params_pyEFPE'][ir].copy()
		if ilabel==1: print_params.update(update_params_pyEFPE_2)
		print('params =', print_params)

#compute mismatch summary
print('\n[5, 50, 95] mismatch percentiles')
print(np.percentile(result['MM'], [5, 50, 95], axis=0))

#compute and show timing stuff
wf_run_speedup = result['run_times'][:,3:]/result['run_times'][:,:3]
wf_t_percs = np.transpose(np.percentile(result['run_times'], [5, 50, 95], axis=0))
wf_s_percs = np.transpose(np.percentile(wf_run_speedup, [5, 50, 95], axis=0))
print('\n[5, 50, 95] timing percentiles')
print('ini  : %s: %ss\t%s: %ss\tSpeedup: %s'%(labels[0], wf_t_percs[0], labels[1], wf_t_percs[3], wf_s_percs[0]))
print('h    : %s: %ss\t%s: %ss\tSpeedup: %s'%(labels[0], wf_t_percs[1], labels[1], wf_t_percs[4], wf_s_percs[1]))
print('total: %s: %ss\t%s: %ss\tSpeedup: %s'%(labels[0], wf_t_percs[2], labels[1], wf_t_percs[5], wf_s_percs[2]))

#make a histogram of the mismatches
mismatches = np.maximum(result['MM'], 1e-16)
bins = np.geomspace(np.amin(mismatches), np.amax(mismatches), Nbins)
plt.figure(figsize=(12,8))
plt.hist(mismatches, bins=bins, histtype='step')
plt.xlabel(r'Mismatch')
plt.ylabel(r'Number of waveforms')
plt.xscale('log')
plt.xlim(bins[0], bins[-1])
plt.tight_layout()
plt.savefig(plotdir+'/mismatches'+string_id+'.pdf')

#make a figure of the speedup
plt.figure(figsize=(12,8))
for ip, plabel in enumerate([r'Initialization', r'Evaluation', r'Total']):
	bins = np.geomspace(np.amin(wf_run_speedup[:,ip]), np.amax(wf_run_speedup[:,ip]), Nbins)
	plt.hist(wf_run_speedup[:,ip], label=plabel, bins=bins)
plt.xlabel(r'Ratio between Run Times')
plt.ylabel(r'Number of waveforms')
plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.savefig(plotdir+'/speedup'+string_id+'.pdf')


#Runtime
print("\nRuntime: %s seconds" % (time.time() - start_runtime))

plt.show()
