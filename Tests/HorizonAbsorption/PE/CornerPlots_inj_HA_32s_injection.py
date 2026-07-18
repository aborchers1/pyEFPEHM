import os
import numpy as np
import h5py
from bilby.gw.conversion import generate_all_bbh_parameters
from matplotlib import pyplot as plt

import matplotlib

import sys
sys.path.append('../../PE/injections/')
from PE_plot_utils import *

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

#####################################################

#label of the analysis
analysis_label = 'Inj_QCHA_32s' #['Inj_QCHA_32s', 'Inj_eccHA_32s']

#inputs
files_name = './%s/%s/outdir/final_result/inj_HA_rec_%s_data0_1262276684-0_analysis_H1L1V1_merge_result.hdf5'
runs ={
           r'HA ($d_L = 500\,\mathrm{Mpc}$)': ('HA', 'lowSNR'),
           r'HA ($d_L = 200\,\mathrm{Mpc}$)': ('HA', 'highSNR'),
           r'HA ($d_L = 100\,\mathrm{Mpc}$)': ('HA', 'vhighSNR'),
           r'HA  ($d_L = 50\,\mathrm{Mpc}$)': ('HA', 'ehighSNR'),
           r'no HA ($d_L = 500\,\mathrm{Mpc}$)': ('noHA', 'lowSNR'),
           r'no HA ($d_L = 200\,\mathrm{Mpc}$)': ('noHA', 'highSNR'),
           r'no HA ($d_L = 100\,\mathrm{Mpc}$)': ('noHA', 'vhighSNR'),
           r'no HA  ($d_L = 50\,\mathrm{Mpc}$)': ('noHA', 'ehighSNR'),
}
waveform_kwargs = {'f22_start': 20}

#linestyles for each run config
ls_per_config = {'HA': '-', 'noHA': '--'}

#color for each run case
color_per_case = {'lowSNR': 'C0', 'highSNR': 'C1', 'vhighSNR': 'C2', 'ehighSNR': 'C3'}

#interferometers to use
ifos = ['H1', 'L1', 'V1']

#parameters to plot
plot_params ={'intrinsic_important': {'params': ['chirp_mass', 'mass_ratio', 'chi_eff', 'chi_p', 'eccentricity'], 'runs': list(runs.keys()), 'kwargs': {}},
              'physically_important': {'params': ['mass_1', 'mass_2', 'eccentricity', 'a_1', 'a_2', 'tilt_1', 'tilt_2'], 'runs': list(runs.keys()), 'kwargs': {}},
              'all_important': {'params': ['chirp_mass', 'mass_ratio', 'chi_eff', 'chi_p', 'eccentricity', 'iota', 'luminosity_distance', 'ra', 'dec', 'dt_inj_ms'], 'runs': list(runs.keys()), 'kwargs': {}},
              'ecc_prec': {'params': ['eccentricity', 'mean_anomaly', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'chi_eff', 'chi_p'], 'runs': list(runs.keys()), 'kwargs': {}},
              }

#injection parameters
injection_keys = ['mass_1', 'mass_2', 'a_1', 'a_2', 'phi_12', 'phi_jl', 'tilt_1', 'tilt_2', 'luminosity_distance', 'theta_jn', 'psi', 'phase', 'geocent_time', 'ra', 'dec', 'eccentricity', 'mean_anomaly']

#output directory
outdir = './Plots/'

########################################################

#construct posterior dictionaries
posteriors = {}
injection_parameters = {}
print('Analyzing', analysis_label)
for run, case in runs.items():

	#load the hdf5 file with the final result
	f = h5py.File(files_name%(analysis_label,'rec_'+case[0]+'_'+case[1], case[0]), 'r')

	try: assert set(ifos)=={x.decode('utf-8').strip("'") for x in f['meta_data']['command_line_args']['detectors'][...]}
	except: print('Warning: Requested ifos are not the same as the ifos in the PE run')
	
	#extract the injection parameters as a dictionary
	inj_params = {key: f['injection_parameters'][key][...].item() for key in injection_keys}
	#add the correct reference frequency
	inj_params['reference_frequency'] = waveform_kwargs['f22_start']
	#compute all bbh parameters of injection
	injection_parameters[run] = generate_all_bbh_parameters(inj_params)
	for key, item in injection_parameters[run].items():
		if isinstance(item, np.ndarray): injection_parameters[run][key] = item.item()

	#extract the posterior as a dictionary of numpy arrays
	posteriors[run] = {key: f['posterior'][key][...] for key in f['posterior']}
	posteriors[run]['dt_inj_ms'] = 1000*(posteriors[run]['geocent_time'] - injection_parameters[run]['geocent_time'])
	injection_parameters[run]['dt_inj_ms'] = 0
	
	#compute also the matched filter SNR
	for i_ifo, ifo in enumerate(ifos):
		snr_opt_i = posteriors[run][ifo+'_optimal_snr']
		snr_mf_i  = posteriors[run][ifo+'_matched_filter_snr']
		if i_ifo==0:
			posteriors[run]['optimal_snr']        = snr_opt_i**2
			posteriors[run]['matched_filter_snr'] = snr_opt_i*snr_mf_i
		else:
			posteriors[run]['optimal_snr']        += snr_opt_i**2
			posteriors[run]['matched_filter_snr'] += snr_opt_i*snr_mf_i

	posteriors[run]['optimal_snr'] = np.sqrt(posteriors[run]['optimal_snr'])
	posteriors[run]['matched_filter_snr'] = np.real(posteriors[run]['matched_filter_snr']/posteriors[run]['optimal_snr'])
	posteriors[run]['full_log_likelihood'] = posteriors[run]['optimal_snr']*(posteriors[run]['matched_filter_snr'] - 0.5*posteriors[run]['optimal_snr']) + f['log_noise_evidence'][...]
	
	posteriors[run]['log_bayes_factor'] = f['log_bayes_factor'][...]
	posteriors[run]['log_evidence_err'] = f['log_evidence_err'][...]
	injection_parameters[run]['snr']    = np.sqrt(-2*f['log_noise_evidence'][...])

	print('rec_'+case[0]+'_'+case[1]+': Log Bayes Factor %.2f +- %.2f,  Maximum Likelihood: %.4f,  SNR: %.2f'%(f['log_bayes_factor'][...], f['log_evidence_err'][...], np.amax(posteriors[run]['full_log_likelihood']), np.sqrt(-2*f['log_noise_evidence'][...])))

#make sure outdir for plots exists
if not os.path.exists(outdir): os.makedirs(outdir)

#loop over lists of parameters to plot corners
corner_figs = []
for key in plot_params:
	#create corner figure
	corner_figs.append(plot_multiple_in_corner(posteriors, plot_params[key]['params'], plot_runs=plot_params[key]['runs'], injection_params=injection_parameters, **plot_params[key]['kwargs']))
	#save it
	corner_figs[-1].savefig(outdir+'/corner_'+analysis_label+'_'+key+'.pdf')

#loop over lists to make density plots of
for key in plot_params:
	#create density figure
	fig = plot_multiple_densities(posteriors, plot_params[key]['params'], plot_runs=plot_params[key]['runs'], injection_params=injection_parameters,
		linestyles=[ls_per_config[runs[run][0]] for run in plot_params[key]['runs']],
		colors=[color_per_case[runs[run][1]] for run in plot_params[key]['runs']],
		posterior_fraction_to_plot=0.98, linewidth=2.5)
	fig.savefig(outdir+'/densities_'+analysis_label+'_'+key+'.pdf')

#########################  This code is very non-general   ###############################

#compute the inverse of the run dictionary
inverse_runs = {item: key for key, item in runs.items()}

#compute bayes factors between the two hypothesis
lnB_HAnoHA, err_lnB_HAnoHA, inj_snrs = [], [], []
for run_case in ['lowSNR', 'highSNR', 'vhighSNR', 'ehighSNR']:

	#extract required information from the runs
	key_HA   = (  'HA', run_case)
	key_noHA = ('noHA', run_case)
	lnB_HA    ,     lnB_noHA = [posteriors[inverse_runs[key]]['log_bayes_factor'] for key in [key_HA, key_noHA]]
	err_lnB_HA, err_lnB_noHA = [posteriors[inverse_runs[key]]['log_evidence_err'] for key in [key_HA, key_noHA]]
	inj_snr_HA, inj_snr_noHA = [injection_parameters[inverse_runs[key]]['snr']    for key in [key_HA, key_noHA]]
	
	#compute bayes factors between the two hypothesis
	lnB_HAnoHA.append(lnB_HA - lnB_noHA)
	
	#compute their error
	err_lnB_HAnoHA.append(np.linalg.norm([err_lnB_HA, err_lnB_noHA]))
	
	#make sure both runs had the same snr and store it
	assert inj_snr_HA == inj_snr_noHA
	inj_snrs.append(inj_snr_HA)

	print(run_case+r': lnB_HAnoHA = %.2f \pm %.2f'%(lnB_HAnoHA[-1], err_lnB_HAnoHA[-1]))


#save array
np_file_name = outdir+'/%s_injsnrs_lnBHAnoHAerr_lnBHAnoHA.npy'
np.save(np_file_name%(analysis_label), np.array([inj_snrs, lnB_HAnoHA, err_lnB_HAnoHA]))

#make a plot
plt.figure(figsize=(10,8))
for alabel, plabel in zip(['Inj_QCHA_32s', 'Inj_eccHA_32s'],[r'$e_0^\mathrm{inj} = 0$', r'$e_0^\mathrm{inj} = 0.3$']):
	#this overlay needs the .npy saved by a previous run with each analysis_label; skip the ones not generated yet
	if not os.path.exists(np_file_name%(alabel)):
		print('Warning: %s not found, run this script with analysis_label=%s to include it in the overlay'%(np_file_name%(alabel), alabel))
		continue
	inj_snrs, lnB_HAnoHA, err_lnB_HAnoHA = np.load(np_file_name%(alabel))
	plt.errorbar(inj_snrs, lnB_HAnoHA, yerr=err_lnB_HAnoHA, linestyle='', marker='o', markersize=7, label=plabel)
plt.legend()
plt.xlabel(r'Injection SNR')
plt.ylabel(r'$\log\mathcal{B}^\text{HA}_\text{no HA}$')
plt.axhline(y=0, color='k')
plt.savefig(outdir+'/lnB_HAnoHA.pdf')

#Runtime
print("\nRuntime: %s seconds" % (time.time() - start_runtime))

plt.show()

