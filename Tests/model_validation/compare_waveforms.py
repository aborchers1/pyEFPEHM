import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

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

#Chirp mass bounds as a function of segment length
mc_low_of_seglen  = {4: 12, 8:  8, 16: 5, 32: 3.3, 64: 2.2, 128: 1.4, 256: 0.95}
mc_high_of_seglen = {4: 20, 8: 12, 16: 8, 32:   5, 64: 3.3, 128: 2.2, 256:  1.4}

#store the different comparisons
comparison_info = {}

# Entries to compare pyEFPE vs SpinTaylorT4 with pn_spin_order=4 and pn_spin_order=6
comparison_info["SpinTaylorT4_pn_spin_4"] = dict(
ecc_high = 0,
approximant = 'SpinTaylorT4',
pn_spin_order = 4,
pn_phase_order = 7,
pn_amplitude_order = 2,
minimize_parameters = ['phase', 'phase_s'],
params_pyEFPE = {},
precessing = True,
flim_type = 'ISCO'
)

comparison_info["SpinTaylorT4_pn_spin_6"] = comparison_info["SpinTaylorT4_pn_spin_4"].copy()
comparison_info["SpinTaylorT4_pn_spin_6"].update(dict(
pn_spin_order = 6,
))

# Entries to compare pyEFPE vs SEOBNRv5EHM and SEOBNRv5PHM
comparison_info["SEOBNRv5EHM"] = dict(
ecc_high = 0.4,
approximant = 'SEOBNRv5EHM',
pn_spin_order = 8,
pn_phase_order = 9,
pn_amplitude_order = 2,
minimize_parameters = ['phase', 'eccentricity', 'mean_anomaly'],
params_pyEFPE = {'mode_array': [[2,1],[2,2],[3,2],[3,3],[4,4]],},
precessing = False,
flim_type = 'MECO'
)

comparison_info["SEOBNRv5PHM"] = comparison_info["SEOBNRv5EHM"].copy()
comparison_info["SEOBNRv5PHM"].update(dict(
ecc_high = 0,
approximant = 'SEOBNRv5PHM',
minimize_parameters = ['phase', 'phase_s'],
precessing = True,
))

# Entries to compare pyEFPE vs SEOBNRv6E(P)HM
comparison_info["SEOBNRv6EHM"] = dict(
ecc_high = 0.4,
approximant = 'SEOBNRv6EHM',
pn_spin_order = 8,
pn_phase_order = 9,
pn_amplitude_order = 2,
minimize_parameters = ['phase', 'eccentricity', 'mean_anomaly'],
params_pyEFPE = {'mode_array': [[2,1],[2,2],[3,2],[3,3],[4,4]],},
precessing = False,
flim_type = 'MECO'
)

comparison_info["SEOBNRv6EPHM"] = comparison_info["SEOBNRv6EHM"].copy()
comparison_info["SEOBNRv6EPHM"].update(dict(
approximant = 'SEOBNRv6EPHM',
minimize_parameters = ['phase', 'phase_s', 'eccentricity', 'mean_anomaly'],
precessing = True,
))

# Entries to compare pyEFPE vs TEOBResumS(EPHM) and TEOBResumS(PHM)
comparison_info["TEOBResumS_EPHM"] = dict(
ecc_high = 0.4,
approximant = 'TEOBResumS',
pn_spin_order = 8,
pn_phase_order = 9,
pn_amplitude_order = 2,
minimize_parameters = ['phase', 'phase_s', 'eccentricity', 'mean_anomaly'],
params_pyEFPE = {'mode_array': [[2,1],[2,2],[3,1],[3,2],[3,3],[4,2],[4,4]],},
precessing = True,
flim_type = 'MECO'
)

comparison_info["TEOBResumS_EHM"] = comparison_info["TEOBResumS_EPHM"].copy()
comparison_info["TEOBResumS_EHM"].update(dict(
minimize_parameters = ['phase', 'eccentricity', 'mean_anomaly'],
precessing = False,
))

############################################################################

#choose the comparison to perform
comparison_name = "SEOBNRv5EHM"

#Setting up boundary conditions for random parameter generation
seglen = 4
q_low = 0.05
q_high = 1
s1_high = 0.9
s2_high = 0.9
f_min = 20
distance_Mpc = 10  # 10 Mpc is default 
fmax_flim = 0.8 #fraction of flim that fmax represents
f_min_gen_fact = 0.8 #fraction of fmin to start waveform generation from

#output directory
outdir='./outdir/waveform_comparisons'

#PSD to use, put psd_name=None to turn off the psd
psd_name = 'AplusDesign' #'AplusDesign' 'aLIGO175MpcT1800545' 'aLIGOAPlusDesignSensitivityT1800042'

#Number of waveforms to test
Nsamples = 2000

#Differential evolution kwargs
rtol_e=0.3
rtol_p=0.1
maxiter=100
popsize=15

#number of workers for parallelization
nworkers = 8 #set it to -1 to set nworkers=max(mp.cpu_count()-1, 1)

#Number of top mismatches to print
N_top_MM_print = 50

#choose to show plots or not
show_plots = True

##############################################################################

#extract parameters
mc_low  =  mc_low_of_seglen[seglen]
mc_high = mc_high_of_seglen[seglen]
ecc_high            = comparison_info[comparison_name]["ecc_high"]
approximant         = comparison_info[comparison_name]["approximant"]
pn_spin_order       = comparison_info[comparison_name]["pn_spin_order"]
pn_phase_order      = comparison_info[comparison_name]["pn_phase_order"]
pn_amplitude_order  = comparison_info[comparison_name]["pn_amplitude_order"]
minimize_parameters = comparison_info[comparison_name]["minimize_parameters"]
params_pyEFPE       = comparison_info[comparison_name]["params_pyEFPE"]
precessing          = comparison_info[comparison_name]["precessing"]
flim_type           = comparison_info[comparison_name]["flim_type"]

if __name__ == "__main__":

	#generate string id
	string_id  = generate_string_ID(Nsamples, precessing, approximant, psd_name, mc_low, mc_high, q_low, q_high, s1_high, s2_high, flim_type, pn_spin_order, pn_phase_order, pn_amplitude_order, params_pyEFPE, minimize_parameters=minimize_parameters, ecc_high=ecc_high, fmax_flim=fmax_flim)

	#Try to load a dictionary with the samples
	try:
		with open(outdir+'/params'+string_id+'.pickle', 'rb') as handle: samples = pickle.load(handle)
	except:
		print('Failed to load %s\nComputing Mismatches:'%(outdir+'/params'+string_id+'.pickle'))
		#compute the mismatches using our function
		samples = random_mismatches_with_EFPE(Nsamples, approximant, seglen=seglen, f_min=f_min, flim_type=flim_type, fmax_flim=fmax_flim, psd_name=psd_name, distance_Mpc=distance_Mpc, mc_low=mc_low, mc_high=mc_high, q_low=q_low, q_high=q_high, s1_high=s1_high, s2_high=s2_high, ecc_high=ecc_high, precessing=precessing, params_pyEFPE=params_pyEFPE, pn_spin_order=pn_spin_order, pn_phase_order=pn_phase_order, pn_amplitude_order=pn_amplitude_order, outdir=outdir, f_min_gen_fact=f_min_gen_fact, minimize_parameters=minimize_parameters, nworkers=nworkers, rtol_e=rtol_e, rtol_p=rtol_p, maxiter=maxiter, popsize=popsize)

	print('\nTop %s realizations with largest mismatch\n'%(N_top_MM_print))
	#print the parameters associated with the top ones
	for iMM in np.argsort(-samples['mismatches'])[:N_top_MM_print]:
		print('\nMismatch: ', samples['mismatches'][iMM])
		print('params =', samples['all_params_pyEFPE'][iMM])

	#if the directory for plots does not exist, create it
	plots_dir = outdir+'/Plots/'
	if not os.path.exists(plots_dir): os.makedirs(plots_dir)

	#make a histogram of the mismatches
	bins = np.geomspace(np.nanmin(samples['mismatches']), np.nanmax(samples['mismatches']), 25)
	plt.figure(figsize=(12,8))
	plt.hist(samples['mismatches'], bins=bins)
	plt.xlabel(r'Mismatch')
	plt.ylabel(r'Number of waveforms')
	plt.xscale('log')
	plt.xlim(bins[0], bins[-1])
	plt.savefig(plots_dir+'histo_mismatches'+string_id+'.pdf')
	plt.tight_layout()

	#make a scatter plot of chi_eff-q
	plt.figure(figsize=(12,8))
	plt.scatter(samples['chi_eff'], samples['mismatches'], c=samples['q'], vmin=q_low, vmax=q_high)
	plt.xlabel(r'$\chi_\mathrm{eff}$')
	if s1_high==s2_high: plt.xlim(-s1_high, s1_high)
	plt.ylabel(r'$\overline{\mathcal{MM}}$')
	plt.yscale('log')
	plt.colorbar(label=r'$q$')
	plt.tight_layout()
	plt.savefig(plots_dir+'mismatches_chi_eff_q'+string_id+'.pdf')

	#make a scatter plot of chi_p-chi_eff
	if precessing:
		plt.figure(figsize=(12,8))
		plt.scatter(samples['chi_p'], samples['mismatches'], c=np.abs(np.cos(samples['iota'])))
		plt.xlabel(r'$\chi_\mathrm{p}$')
		plt.ylabel(r'$\overline{\mathcal{MM}}$')
		plt.yscale('log')
		plt.colorbar(label=r'$|\cos{\iota}|$')
		plt.tight_layout()
		plt.savefig(plots_dir+'mismatches_chip_cinc'+string_id+'.pdf')

	#if the system is eccentric, make a scatter plot of ecc-q
	if ecc_high>0:
		plt.figure(figsize=(12,8))
		plt.scatter(samples['ecc'], samples['mismatches'], c=samples['q'], vmin=q_low, vmax=q_high)
		plt.xlabel(r'$e$')
		plt.xlim(0, ecc_high)
		plt.ylabel(r'$\overline{\mathcal{MM}}$')
		plt.yscale('log')
		plt.colorbar(label=r'$q$')
		plt.tight_layout()
		plt.savefig(plots_dir+'mismatches_ecc_q'+string_id+'.pdf')

	#if we have performed eccentricity minimization, compare eccentricities of both models
	if "eccentricity" in minimize_parameters:

		#extract the best fit pyEFPE eccentricity
		e_pyEFPE = np.array([params_pyEFPE_i['eccentricity'] for params_pyEFPE_i in samples['all_params_pyEFPE']])

		#find maximum eccentricity to consider
		e_max = max(ecc_high, np.amax(e_pyEFPE))

		plt.figure(figsize=(12,8))
		plt.scatter(samples['ecc'], e_pyEFPE, c=np.log10(samples['mismatches']))
		plt.plot([0, e_max], [0, e_max], 'k--')
		plt.plot([0, e_max], [0, (1+rtol_e)*e_max], 'k--')
		plt.plot([0, e_max], [0, (1-rtol_e)*e_max], 'k--')
		plt.xlim(0, e_max)
		plt.ylim(0, e_max)
		plt.xlabel(r'$e_\mathtt{%s}$'%(approximant))
		plt.ylabel(r'$e_\mathtt{pyEFPE}$')
		plt.colorbar(label=r'$\log_{10}(\overline{\mathcal{MM}})$')
		plt.tight_layout()
		plt.savefig(plots_dir+'compare_eccs'+string_id+'.pdf')


	#Runtime
	print("\nRuntime: %s seconds" % (time.time() - start_runtime))

	if show_plots:
		plt.show()

		#make plots comparing waveforms explicitly
		compare_waveforms_from_params(samples, plot_order='maximum mismatch')

