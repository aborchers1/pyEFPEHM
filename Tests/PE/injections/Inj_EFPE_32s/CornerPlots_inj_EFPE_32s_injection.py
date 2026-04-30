import os
import numpy as np
import h5py
from bilby.gw.conversion import generate_all_bbh_parameters
from matplotlib import pyplot as plt

import matplotlib
print('Matplotlib version:', matplotlib.__version__)

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
file_name = "rec_EFPE/outdir/final_result/inj_EFPE_rec_EFPE_data0_1262276684-0_analysis_H1L1V1_merge_result.hdf5"
waveform_kwargs = {'f22_start': 20}

#label of the analysis
analysis_label = 'inj_EFPE_rec_EFPE'

#parameters to plot
plot_params ={'intrinsic_important': ['chirp_mass', 'mass_ratio', 'chi_eff', 'chi_p', 'eccentricity'],
              'all_important':  ['chirp_mass', 'mass_ratio', 'chi_eff', 'chi_p', 'eccentricity', 'mean_anomaly', 'iota', 'luminosity_distance', 'ra', 'dec', 'dt_inj_ms'],
              'ecc_prec': ['eccentricity', 'mean_anomaly', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'chi_eff', 'chi_p']}

#injection parameters
injection_keys = ['mass_1', 'mass_2', 'a_1', 'a_2', 'phi_12', 'phi_jl', 'tilt_1', 'tilt_2', 'luminosity_distance', 'theta_jn', 'psi', 'phase', 'geocent_time', 'ra', 'dec', 'eccentricity', 'mean_anomaly']

#output directory
outdir = 'Plots/'

########################################################

#load the hdf5 file with the final result
f = h5py.File(file_name, 'r')

#extract the injection parameters as a dictionary
injection_parameters = {key: f['injection_parameters'][key][...].item() for key in injection_keys}
#add the correct reference frequency
injection_parameters['reference_frequency'] = waveform_kwargs['f22_start']
#compute all bbh parameters of injection
injection_parameters_all = generate_all_bbh_parameters(injection_parameters)
for key, item in injection_parameters_all.items(): 
	if isinstance(item, np.ndarray): injection_parameters_all[key] = item.item()

#extract the posterior as a dictionary of numpy arrays
post = {key: f['posterior'][key][...] for key in f['posterior']}
post['dt_inj_ms'] = 1000*(post['geocent_time'] - injection_parameters_all['geocent_time'])
injection_parameters_all['dt_inj_ms'] = 0

#make sure outdir for plots exists
if not os.path.exists(outdir): os.makedirs(outdir)

#loop over lists of parameters to plot
corner_figs = []
for key in plot_params:
	#create corner figure
	corner_figs.append(plot_corner(post, plot_params[key], injection_params=injection_parameters_all))
	#save it
	corner_figs[-1].savefig(outdir+'/'+analysis_label+'_'+key+'.pdf')

#Runtime
print("\nRuntime: %s seconds" % (time.time() - start_runtime))

plt.show()

