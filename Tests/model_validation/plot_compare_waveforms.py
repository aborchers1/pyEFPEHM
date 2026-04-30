import matplotlib
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

import numpy as np
import pickle

#store the different comparisons
comparison_info = {}

#compare precessing SpinTaylorT4 with pn_spin_order=4 and pn_spin_order=4
comparison_info['prec_SpinTaylorT4_PN_spin_4_6'] = {
'filenames': ['params_prec_SpinTaylorT4_N_2000_mc_%.3g_%.3g_q_0.05_1_s1_0.9_s2_0.9_fmax_0.8ISCO_PN_spin_4_phase_7_amp_2_HMs_10_min_p0pS_AplusDesign.pickle',
              'params_prec_SpinTaylorT4_N_2000_mc_%.3g_%.3g_q_0.05_1_s1_0.9_s2_0.9_fmax_0.8ISCO_PN_spin_6_phase_7_amp_2_HMs_10_min_p0pS_AplusDesign.pickle'],
'labels': ['$\\mathtt{SpinTaylorT4}$\n2PN Spin', '$\\mathtt{SpinTaylorT4}$\n3PN Spin'],
'legend_kwargs': {'fontsize': 22},
}

#compare SEOBNRv5PHM with SEOBNRv5EHM
comparison_info['SEOBNRv5_EHM_PHM'] = {
'filenames': ['params_ecc_0.4_SEOBNRv5EHM_N_2000_mc_%.3g_%.3g_q_0.05_1_s1_0.9_s2_0.9_fmax_0.8MECO_PN_spin_8_phase_9_amp_2_HMs_5_min_p0e0l0_AplusDesign.pickle',
              'params_prec_SEOBNRv5PHM_N_2000_mc_%.3g_%.3g_q_0.05_1_s1_0.9_s2_0.9_fmax_0.8MECO_PN_spin_8_phase_9_amp_2_HMs_5_min_p0pS_AplusDesign.pickle',],
'labels': [r'$\mathtt{SEOBNRv5EHM}$', r'$\mathtt{SEOBNRv5PHM}$'],
'legend_kwargs': {'fontsize': 22},
}

#compare SEOBNRv5EHM with SEOBNRv6EHM
comparison_info['SEOBNR_v5_v6_EHM'] = {
'filenames': ['params_ecc_0.4_SEOBNRv5EHM_N_2000_mc_%.3g_%.3g_q_0.05_1_s1_0.9_s2_0.9_fmax_0.8MECO_PN_spin_8_phase_9_amp_2_HMs_5_min_p0e0l0_AplusDesign.pickle',
              'params_ecc_0.4_SEOBNRv6EHM_N_2000_mc_%.3g_%.3g_q_0.05_1_s1_0.9_s2_0.9_fmax_0.8MECO_PN_spin_8_phase_9_amp_2_HMs_5_min_p0e0l0_AplusDesign.pickle',],
'labels': [r'$\mathtt{SEOBNRv5EHM}$', r'$\mathtt{SEOBNRv6EHM}$'],
}

#compare SEOBNRv6EHM with SEOBNRv6EPHM
comparison_info['SEOBNRv6_EHM_EPHM'] = {
'filenames': ['params_ecc_0.4_SEOBNRv6EHM_N_2000_mc_%.3g_%.3g_q_0.05_1_s1_0.9_s2_0.9_fmax_0.8MECO_PN_spin_8_phase_9_amp_2_HMs_5_min_p0e0l0_AplusDesign.pickle',
              'params_prec_ecc_0.4_SEOBNRv6EPHM_N_2000_mc_%.3g_%.3g_q_0.05_1_s1_0.9_s2_0.9_fmax_0.8MECO_PN_spin_8_phase_9_amp_2_HMs_5_min_p0pSe0l0_AplusDesign.pickle',],
'labels': [r'$\mathtt{SEOBNRv6EHM}$', r'$\mathtt{SEOBNRv6EPHM}$'],
}

#compare precessing eccentric TEOBResumS-Dali
comparison_info['TEOBResumS_EPHM_EHM'] = {
'filenames': ['params_prec_ecc_0.4_TEOBResumS_N_2000_mc_%.3g_%.3g_q_0.05_1_s1_0.9_s2_0.9_fmax_0.8MECO_PN_spin_8_phase_9_amp_2_HMs_7_min_p0pSe0l0_AplusDesign.pickle',
              'params_ecc_0.4_TEOBResumS_N_2000_mc_%.3g_%.3g_q_0.05_1_s1_0.9_s2_0.9_fmax_0.8MECO_PN_spin_8_phase_9_amp_2_HMs_7_min_p0e0l0_AplusDesign.pickle'],
'labels': ['$\\mathtt{TEOBResumS-Dali}$\nEPHM', '$\\mathtt{TEOBResumS-Dali}$\nEHM'],
'legend_kwargs': {'fontsize': 20},
}


##########################################################################################

#output directory
outdir = './outdir/waveform_comparisons/'

#comparison to plot
plot_label = 'SEOBNRv5_EHM_PHM'

#chirp masses
mc_lows  = [12,  8, 5, 3.3, 2.2, 1.4, 0.95]
mc_highs = [20, 12, 8, 5, 3.3, 2.2, 1.40]
seglens =  [4, 8, 16, 32, 64, 128, 256]

#quantiles to show in plots
quantiles = [.05, .5, .95]

#colors
colors = ['C0', 'C1']

#remove the samples with f_max <= f_min + ndf_min*delta_f
remove_below_fmax = True
f_min = 20
ndf_min = 1

#choose to show also chirp mass range
show_chirp_mass_range = False

##########################################################################################

# load the data
mismatches, all_quantiles, xlabels = [], [], []
for i in range(len(mc_lows)):
	
	#load the result dictionary
	try:
		results = list()
		for filename in comparison_info[plot_label]['filenames']:
			with open(outdir+'/'+filename%(mc_lows[i], mc_highs[i]), 'rb') as handle: results.append(pickle.load(handle))
			#make sure the seglen is correct
			assert results[-1]['seglen'] == seglens[i]
	except:
		continue
	
	#append the mismatches and quantiles to use
	mismatches.append([np.log10(np.maximum(np.abs(result['mismatches']), 1e-17)) for result in results])

	#check if there are samples below the specified fmin
	if remove_below_fmax:
		for ir, result in enumerate(results):
			#samples below f_min
			i_remove = (result['f_max'] < (f_min + (1+ndf_min)/seglens[i]))
			#if there are any, do not display this case in plots
			if np.any(i_remove):
				print("For mc in [%s , %s] there are %s samples with f_max <= f_min + %s*delta_f"%(mc_lows[i], mc_highs[i], np.sum(i_remove), ndf_min+1))
				mismatches[-1][ir] = [np.nan, np.nan]
		
	all_quantiles.append(quantiles)
	
	#append a label
	if show_chirp_mass_range: xlabels.append(r'%s\,s\n$[%s , %s]$'%(seglens[i], mc_lows[i], mc_highs[i]))
	else:                     xlabels.append(r'%s\,s'%(seglens[i]))
	
#transpose mismatches
mismatches = list(map(list, zip(*mismatches)))

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Make violin plot
violins = list()
for color, mismatch, side in zip(colors, mismatches, ['low', 'high']):

	#plot this mismatch on its side of the violin
	violins.append(ax.violinplot([MM for MM in mismatch], widths=0.8, quantiles=all_quantiles, showextrema=False, side=side))

	#set colors
	for body in violins[-1]['bodies']:
		body.set_facecolor(color)
		body.set_edgecolor(color)
		body.set_alpha(1)
	violins[-1]['cquantiles'].set_color('k')

# Add labels etc
ax.set_xticks(1 + np.arange(len(xlabels)), xlabels)

if show_chirp_mass_range: ax.set_xlabel(r'Duration $T$ / Chirp mass range $\\left[\\mathcal{M}_{c,\\mathrm{min}}, \\mathcal{M}_{c,\\mathrm{max}} \\right]$ $[M_\\odot]$')
else:                     ax.set_xlabel(r'Duration $T$')

ax.tick_params(axis='x')
ax.set_xlim(0.5, len(all_quantiles)+0.5)
ax.set_ylim(top=0)
ax.set_ylabel(r'$\overline{\mathcal{MM}}$')
if 'legend_kwargs' not in comparison_info[plot_label].keys(): comparison_info[plot_label]['legend_kwargs'] = {}
plt.legend([violins[0]['bodies'][0], violins[1]['bodies'][0], violins[0]['cquantiles']], [*comparison_info[plot_label]['labels'], '%s\nquantiles'%(quantiles)], ncols=3, loc=(0,1), framealpha=0, **comparison_info[plot_label]['legend_kwargs'])

#get the current limits of the y-axis
ymin_plot, ymax_plot = ax.get_ylim()
ymin, ymax = np.floor(ymin_plot), np.ceil(ymax_plot)
#find locations of major and minor ticks
major_ticks = np.arange(ymin, ymax+1)
minor_ticks = np.concatenate([y0 + np.log10(np.arange(2, 10)) for y0 in major_ticks[:-1]])
#set ticks and labels
ax.set_yticks(major_ticks, minor=False)
ax.set_yticks(minor_ticks, minor=True)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: "$10^{%.f}$"%(y)))
#Put the y-axis limits that were there initially
ax.set_ylim(ymin_plot, ymax_plot)

plt.tight_layout()
plt.savefig(outdir+'/MM_violins_'+plot_label+'.pdf')


plt.show()

