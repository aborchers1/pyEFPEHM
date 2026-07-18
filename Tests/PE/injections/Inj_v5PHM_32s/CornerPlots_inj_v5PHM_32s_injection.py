import os
import numpy as np
import h5py
from bilby.gw.conversion import generate_all_bbh_parameters
from matplotlib import pyplot as plt

import matplotlib

import sys
sys.path.append('../')
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

#inputs
files_name = './%s/outdir/final_result/inj_v5PHM_%s_data0_1262276684-0_analysis_H1L1V1_merge_result.hdf5'
run_cases ={
           r'$\mathtt{SEOBNRv5PHM}$':          'rec_v5PHM',
           r'$\mathtt{pyEFPEHM}$ ($e_0 = 0$)': 'rec_EFPE_QC',
           r'$\mathtt{pyEFPEHM}$':             'rec_EFPE',
}
waveform_kwargs = {'f22_start': 20}

#label of the analysis
analysis_label = 'inj_v5PHM_32s'

#parameters to plot
plot_params ={'intrinsic_important': {'params': ['chirp_mass', 'mass_ratio', 'chi_eff', 'chi_p', 'eccentricity'], 'runs': list(run_cases.keys()), 'kwargs': {}},
              'all_important': {'params': ['chirp_mass', 'mass_ratio', 'chi_eff', 'chi_p', 'eccentricity', 'iota', 'luminosity_distance', 'ra', 'dec'], 'runs': list(run_cases.keys()), 'kwargs': {}}
              }

#injection parameters
injection_keys = ['mass_1', 'mass_2', 'a_1', 'a_2', 'phi_12', 'phi_jl', 'tilt_1', 'tilt_2', 'luminosity_distance', 'theta_jn', 'psi', 'phase', 'geocent_time', 'ra', 'dec']

#output directory
outdir = 'Plots/'

########################################################

#construct posterior dictionaries
posteriors = {}
injection_parameters = {}
for run, case in run_cases.items():

	#load the hdf5 file with the final result
	f = h5py.File(files_name%(case, case), 'r')

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
	injection_parameters[run]['eccentricity'] = 0

#make sure outdir for plots exists
if not os.path.exists(outdir): os.makedirs(outdir)

#loop over lists of parameters to plot
corner_figs = []
for key in plot_params:
	#create corner figure
	corner_figs.append(plot_multiple_in_corner(posteriors, plot_params[key]['params'], plot_runs=plot_params[key]['runs'], injection_params=injection_parameters, **plot_params[key]['kwargs']))
	#save it
	corner_figs[-1].savefig(outdir+'/'+analysis_label+'_'+key+'.pdf')

#make a plot of the eccentricity histogram
plot_case = r'$\mathtt{pyEFPEHM}$'
fig = plot_density(posteriors[plot_case], 'eccentricity', injection_params=injection_parameters[plot_case])
fig.savefig(outdir+'/'+analysis_label+'_'+run_cases[plot_case]+'_eccentricity.pdf')

#make a plot of the eccentricity-chirpmass correlation
posteriors[plot_case]['eccentric_chirp_mass'] = compute_McEcc(posteriors[plot_case]['chirp_mass'], posteriors[plot_case]['eccentricity'])
injection_parameters[plot_case]['eccentric_chirp_mass'] = compute_McEcc(injection_parameters[plot_case]['chirp_mass'], injection_parameters[plot_case]['eccentricity'])
fig = plot_corner(posteriors[plot_case], ['eccentric_chirp_mass', 'chirp_mass', 'eccentricity'], injection_params=injection_parameters[plot_case])
fig.savefig(outdir+'/'+analysis_label+'_'+run_cases[plot_case]+'_EFPE_McEcc.pdf')

#Runtime
print("\nRuntime: %s seconds" % (time.time() - start_runtime))

plt.show()

