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

#compare non-precessing TaylorF2 and SpinTaylorT4
comparison_info["equalized_pyEFPEHM_vs_pyEFPE"] = dict(
labels = ['pyEFPEHM', 'pyEFPE'],
pretty_labels = [r'\texttt{pyEFPEHM}', r'\texttt{pyEFPE}'],
)

comparison_info["default_pyEFPEHM_vs_pyEFPE"] = dict(
labels = ['pyEFPEHM', 'pyEFPE'],
pretty_labels = [r'\texttt{pyEFPEHM}', r'\texttt{pyEFPE}'],
)

comparison_info["Amps_0PN_vs_Amps_1PN"] = dict(
labels = ['Amps_0PN', 'Amps_1PN'],
pretty_labels = ['0PN Amplitudes', '1PN Amplitudes'],
)

##########################################################################################

#type of comparison
comparison_name = "equalized_pyEFPEHM_vs_pyEFPE"

#psd used
psd_name = 'AplusDesign'

#number of points
N_tests = 2000

#chirp masses
mc_lows  = [12,  8, 5, 3.3, 2.2, 1.4, 0.95]
mc_highs = [20, 12, 8, 5, 3.3, 2.2, 1.40]
seglens =  [4, 8, 16, 32, 64, 128, 256]

#output directory
outdir = './outdir/pyEFPE_version_comparisons/'

#quantiles to show in plots
quantiles = [.05, .5, .95]

#colors
color_1 = 'C0'
color_2 = 'C1'

#plot a fit to the timing
fit_timing = False
show_chirp_mass_range = False

##########################################################################################

#pyEFPE version labels
pyEFPE_labels = comparison_info[comparison_name]['labels']
pretty_labels = comparison_info[comparison_name]['pretty_labels']


# load the data
mismatches, speedups, all_quantiles, runtime_1, runtime_2, xlabels = [], [], [], [], [], []
for i in range(len(mc_lows)):
	
	#name of the result
	string_id = '_compare_pyEFPE_%s_N_%s_Mc_%.4g_%.4g_seglen_%s'%(comparison_name, N_tests, mc_lows[i], mc_highs[i], seglens[i])

	#if required, add the ASD to the name
	if type(psd_name)==str: string_id += '_psd_'+psd_name

	#load the result dictionary
	try:
		with open(outdir+'/result'+string_id+'.pickle', 'rb') as handle: result = pickle.load(handle)
	except:
		continue
	
	#append the mismatches, speedups, runtimes and quantiles to use
	mismatches.append(np.log10(np.maximum(np.abs(result['MM']), 1e-17)))
	speedups.append(result['run_times'][:,5]/result['run_times'][:,2])
	runtime_1.append(result['run_times'][:,2])
	runtime_2.append(result['run_times'][:,5])
	all_quantiles.append(quantiles)
	
	#append a label
	if show_chirp_mass_range: xlabels.append(r'%s\,s\n$[%s , %s]$'%(seglens[i], mc_lows[i], mc_highs[i]))
	else:                     xlabels.append(r'%s\,s'%(seglens[i]))

	#print the runtimes
	if len(runtime_1)==1: print("T \t %s Runtimes [s] \t\t %s Runtimes [s]"%(pyEFPE_labels[0], pyEFPE_labels[1]))
	print(seglens[i], np.quantile(runtime_1[-1], quantiles), np.quantile(runtime_2[-1], quantiles))


#compute the quantiles of the runtimes
q_fit = [.16, .5, .84]
quantiles_1 = np.quantile(runtime_1, q_fit, axis=1)
quantiles_2 = np.quantile(runtime_2, q_fit,axis=1)

if (len(all_quantiles) == len(seglens)):

	if fit_timing:
		#fit a line a + b*T to this
		x_plot = np.array(seglens)#/(0.5*(np.array(mc_lows) + np.array(mc_highs)))
		b_1, a_1 = np.polyfit(x_plot, quantiles_1[1], 1, w=1/(quantiles_1[2] - quantiles_1[0]))
		b_2, a_2 = np.polyfit(x_plot, quantiles_2[1], 1, w=1/(quantiles_2[2] - quantiles_2[0]))
		fit_label = r'$%.4f\mathrm{s} + %.4f T/\overline{\mathcal{M}}_c$'

# Create the plot comparing mismatches and speedups of pyEFPE versions
fig, ax1 = plt.subplots(figsize=(12, 8))
# Plot mismatch on the left side of the violins
violins_1 = ax1.violinplot(mismatches, widths=0.8, quantiles=all_quantiles, showextrema=False, side='low')
#set colors
for body in violins_1['bodies']:
	body.set_facecolor(color_1)
	body.set_edgecolor(color_1)
	body.set_alpha(1)
violins_1['cquantiles'].set_color('k')
ax1.set_ylabel(r'$\mathcal{MM}$', color=color_1)
ax1.tick_params(axis='y', which='both', colors=color_1)
#get the current limits of the y-axis
ymin_plot, ymax_plot = ax1.get_ylim()
ymin, ymax = np.floor(ymin_plot), np.ceil(ymax_plot)
#find locations of major and minor ticks
major_ticks = np.arange(ymin, ymax+1)
minor_ticks = np.concatenate([y0 + np.log10(np.arange(2, 10)) for y0 in major_ticks[:-1]])
#set ticks and labels
ax1.set_yticks(major_ticks, minor=False)
ax1.set_yticks(minor_ticks, minor=True)
ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: "$10^{%.f}$"%(y)))
#Put the y-axis limits that were there initially
ax1.set_ylim(ymin_plot, ymax_plot)
# Add a secondary y-axis for the speedup
ax2 = ax1.twinx()
# Plot speedup on the right side of the violins
violins_2 = ax2.violinplot(speedups, widths=0.8, quantiles=all_quantiles, showextrema=False, side='high')
for body in violins_2['bodies']:
	body.set_facecolor(color_2)
	body.set_edgecolor(color_2)
	body.set_alpha(1)
violins_2['cquantiles'].set_color('k')
ax2.set_ylabel(r'Speedup $\left( \frac{\text{%s Runtime}}{\text{%s Runtime}} \right)$'%(pretty_labels[1], pretty_labels[0]), color=color_2)
ax2.tick_params(axis='y', which='both', colors=color_2)
ax2.spines['right'].set_color(color_2)
ax2.spines['left'].set_color(color_1)
#make a line indicating speedup = 1
ax2.axhline(y=1, linewidth=1, linestyle='--', color=color_2)
# Add x-axis labels and title
ax1.set_xticks(1 + np.arange(len(xlabels)), xlabels)
if show_chirp_mass_range: ax1.set_xlabel(r'Duration $T$ / Chirp mass range $\\left[\\mathcal{M}_{c,\\mathrm{min}}, \\mathcal{M}_{c,\\mathrm{max}} \\right]$ $[M_\\odot]$')
else:                     ax1.set_xlabel(r'Duration $T$')
ax1.tick_params(axis='x', labelsize=22)
ax1.set_xlim(0.5, len(all_quantiles)+0.5)
plt.legend([violins_1['bodies'][0], violins_2['bodies'][0], violins_1['cquantiles']], [r'$\mathcal{MM}$', 'Speedup', '%s\nquantiles'%(quantiles)], ncols=3, loc=(0,1), framealpha=0, fontsize=22)
plt.tight_layout()
plt.savefig(outdir+'/'+comparison_name+'_MM_Speedup_violins.pdf')

# Create a plot comparing timings of pyEFPE versions
fig, ax = plt.subplots(figsize=(12, 8))
# Plot mismatch on the left side of the violins
violins_1 = ax.violinplot(runtime_1, widths=0.8, quantiles=all_quantiles, showextrema=False, side='low')
#set colors
for body in violins_1['bodies']:
	body.set_facecolor(color_1)
	body.set_edgecolor(color_1)
violins_1['cquantiles'].set_color('k')
# Plot speedup on the right side of the violins
violins_2 = ax.violinplot(runtime_2, widths=0.8, quantiles=all_quantiles, showextrema=False, side='high')
for body in violins_2['bodies']:
	body.set_facecolor(color_2)
	body.set_edgecolor(color_2)
	
violins_2['cquantiles'].set_color('k')
# Add labels and titles
ax.set_xticks(1 + np.arange(len(xlabels)), xlabels)
if show_chirp_mass_range: ax.set_xlabel(r'Duration $T$ / Chirp mass range $\\left[\\mathcal{M}_{c,\\mathrm{min}}, \\mathcal{M}_{c,\\mathrm{max}} \\right]$ $[M_\\odot]$')
else:                     ax.set_xlabel(r'Duration $T$')
ax.tick_params(axis='x', labelsize=22)
ax.set_xlim(0.5, len(all_quantiles)+0.5)
ax.set_ylabel(r'Runtime [s]')
ax.set_yscale('log')
#plot lines with fit to timings
if (len(all_quantiles) == len(seglens)) and fit_timing:
	t_plot = 1+np.arange(len(seglens))
	p_1 = ax.plot(t_plot, a_1 + b_1*x_plot, color_1)
	p_2 = ax.plot(t_plot, a_2 + b_2*x_plot, color_2)
	for body in violins_1['bodies']: body.set_alpha(0.6)
	for body in violins_2['bodies']: body.set_alpha(0.6)
	plt.legend([violins_1['bodies'][0], violins_2['bodies'][0], violins_1['cquantiles'], p_1[0], p_2[0]], [pretty_labels[0], pretty_labels[1], '%s\nquantiles'%(quantiles), fit_label%(a_1, b_1), fit_label%(a_2, b_2)], loc='upper left', ncols=2, fontsize=20, framealpha=0)
else:
	for body in violins_1['bodies']: body.set_alpha(1)
	for body in violins_2['bodies']: body.set_alpha(1)
	plt.legend([violins_1['bodies'][0], violins_2['bodies'][0], violins_1['cquantiles']], [pretty_labels[0], pretty_labels[1], '%s\nquantiles'%(quantiles)], loc='upper left')
	
plt.tight_layout()
plt.savefig(outdir+'/'+comparison_name+'_runtime_violins.pdf')
plt.show()

