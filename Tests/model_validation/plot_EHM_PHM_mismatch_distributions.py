import numpy as np
import pickle

mc_low_of_seglen  = {4: 12, 8:  8, 16: 5, 32: 3.3, 64: 2.2, 128: 1.4, 256: 0.95}
mc_high_of_seglen = {4: 20, 8: 12, 16: 8, 32:   5, 64: 3.3, 128: 2.2, 256:  1.4}

##############################################################

#Common parameter choices
seglen = 256
Nsamples = 2000
q_low  = 0.05
q_high = 1
s_high = 0.9
ecc_high = 0.4

EHM_model = 'SEOBNRv5EHM'
PHM_model = 'SEOBNRv5PHM'

#Filenames to analyze
outdir = './outdir/waveform_comparisons/'
EHM_filename = 'params_ecc_%.3g_%s_N_%s_mc_%.3g_%.3g_q_%.3g_%.3g_s1_%.3g_s2_%.3g_fmax_0.8MECO_PN_spin_8_phase_9_amp_2_HMs_5_min_p0e0l0_AplusDesign.pickle'%(ecc_high, EHM_model, Nsamples, mc_low_of_seglen[seglen], mc_high_of_seglen[seglen], q_low, q_high, s_high, s_high)
PHM_filename = 'params_prec_%s_N_%s_mc_%.3g_%.3g_q_%.3g_%.3g_s1_%.3g_s2_%.3g_fmax_0.8MECO_PN_spin_8_phase_9_amp_2_HMs_5_min_p0pS_AplusDesign.pickle'%(PHM_model, Nsamples, mc_low_of_seglen[seglen], mc_high_of_seglen[seglen], q_low, q_high, s_high, s_high)

##############################################################

#load the result samples
with open(outdir+'/'+EHM_filename, 'rb') as handle: EHM_samples = pickle.load(handle)
with open(outdir+'/'+PHM_filename, 'rb') as handle: PHM_samples = pickle.load(handle)

##############################################################

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

#make figure
fig, axs = plt.subplots(ncols=2, nrows=2, sharey=True, figsize=(12,10), constrained_layout=True)

#mismatch as a function of chi_eff for v5EHM
sc = axs[0,0].scatter(EHM_samples['chi_eff'], EHM_samples['mismatches'], c=EHM_samples['q'], vmin=q_low, vmax=q_high)
axs[0,0].set_xlim(-s_high, s_high)
axs[0,0].set_xlabel(r'$\chi_\mathrm{eff}$')
#mismatch as a function of ecc for v5EHM
sc = axs[0,1].scatter(EHM_samples['ecc'], EHM_samples['mismatches'], c=EHM_samples['q'], vmin=q_low, vmax=q_high)
axs[0,1].set_xlim(0, ecc_high)
axs[0,1].set_xlabel(r'$e_{16\,\mathrm{Hz}}^\mathrm{EOB}$')
#y-axis
axs[0,0].set_yscale('log')
axs[0,0].set_ylabel(r'$\mathtt{%s}$ $\overline{\mathcal{MM}}$'%(EHM_model))

#mismatch as a function of chi_eff for v5PHM
sc = axs[1,0].scatter(PHM_samples['chi_eff'], PHM_samples['mismatches'], c=PHM_samples['q'], vmin=q_low, vmax=q_high)
axs[1,0].set_xlim(-s_high, s_high)
axs[1,0].set_xlabel(r'$\chi_\mathrm{eff}$')
#mismatch as a function of chi_p for v5PHM
sc = axs[1,1].scatter(PHM_samples['chi_p'], PHM_samples['mismatches'], c=PHM_samples['q'], vmin=q_low, vmax=q_high)
axs[1,1].set_xlim(0, s_high)
axs[1,1].set_xlabel(r'$\chi_\mathrm{p}$')
#y-axis
axs[1,0].set_yscale('log')
axs[1,0].set_ylabel(r'$\mathtt{%s}$ $\overline{\mathcal{MM}}$'%(PHM_model))

#common things
cbar = fig.colorbar(sc, ax=axs, label=r'$q$')
plt.savefig(outdir+'/%s_%s_T_%.f_N_%.f_mismatch_distributions.pdf'%(EHM_model, PHM_model, seglen, Nsamples))


plt.show()


