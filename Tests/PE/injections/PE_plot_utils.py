import numpy as np
import h5py
from bilby.gw.conversion import generate_all_bbh_parameters
from matplotlib import pyplot as plt

#parameter names and labels dictionary
parameter_names = {'mass_1': r'$m_1 \, [M_\odot]$',
                   'mass_2': r'$m_2 \, [M_\odot]$',
                   'chirp_mass': r'$\mathcal{M}_c \, [M_\odot]$',
                   'mass_ratio': r'$q$',
                   'eccentricity': r'$e_0$',
                   'mean_anomaly': r'$\ell_0$',
                   'chi_eff': r'$\chi_\mathrm{eff}$',
                   'chi_p': r'$\chi_\mathrm{p}$',
                   'luminosity_distance': r'$d_L \, \left[ \mathrm{Mpc} \right]$',
                   'iota': r'$\iota_0$',
                   'geocent_time': r'$t_\mathrm{geocent} \, [\mathrm{s}]$',
                   'dt_inj_ms': r'$t_\mathrm{c} - t_\mathrm{c}^{\mathrm{inj}}  \, [\mathrm{ms}]$',
                   'eccentric_chirp_mass': r'$\mathcal{M}_c^\mathrm{ecc} \, [M_\odot]$'
                   }

#function to compute the eccentric chirp mass from Eq.(1.1) of 2108.05861
def compute_McEcc(Mc, e):
	return (1 + (157/40)*e*e)*Mc #Mc/((1 - (157/24)*e*e)**0.6)

#function to create titles
def median_and_error_text(samples, quantiles=[0.05, 0.95], error_figures=1):

	#compute the value of lower quantile, median and upper quantile
	qts = np.percentile(samples, 100*np.sort([quantiles[0], 0.5, quantiles[-1]]))
	
	#compute the lower and upper errors
	errors = np.array([-1,1])*(qts[::2]-qts[1])

	#compute order of magnitude of error
	error_mag = int(np.floor(np.log10(np.amin(errors))))

	#compute the decimal figures to show
	decimal_figs = error_figures - 1 - error_mag
	factor = 10**decimal_figs

	#round the errors and median to the desired precision
	show_median = np.round(qts[1]*factor)/factor
	show_errs = np.round(errors*factor)/factor

	#number of digits to show
	digits = max(0, decimal_figs)
	fmt = '%.'+'%s'%(digits)+'f'

	return (r'$%s^{+%s}_{-%s}$'%(fmt, fmt,fmt))%(show_median, show_errs[1], show_errs[0])
	
#make density plot
def plot_density(posterior, plot_key, injection_params=None, bins=25, density=True, cumulative=False, histtype='step', color='C0', tick_fontsize=14, label_fontsize=16, show_title=True, quantiles=[0.05, 0.5, 0.95], error_figures=2):
	
	from matplotlib import pyplot as plt

	#obtain parameter names
	label = parameter_names[plot_key] if plot_key in parameter_names.keys() else plot_key

	#make plot
	fig, ax = plt.subplots()
	ax.hist(posterior[plot_key], bins=bins, density=density, cumulative=cumulative, histtype=histtype, color=color, label='Posterior')
	ax.set_ylabel('Posterior Probability Density', fontsize=label_fontsize)
	ax.set_xlabel(label, fontsize=label_fontsize)
	ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

	#obtain and show injection value
	if injection_params is not None:
		inj_val = injection_params[plot_key] if plot_key in injection_params.keys() else 0
		ax.axvline(x=inj_val, color='k', label='Injected')
		ax.legend(fontsize=label_fontsize)
	
	#if required compute and plot quantiles
	if quantiles is not None:
		#compute the value of the quantiles
		qs = np.percentile(posterior[plot_key], 100*np.array(quantiles))
		#plot them
		for q in qs: ax.axvline(x=q, color=color, linestyle='--')
	
	#if required, show title
	if show_title:
		title_label = label + ' = ' + median_and_error_text(posterior[plot_key], quantiles=quantiles, error_figures=error_figures)
		ax.set_title(title_label, fontsize=label_fontsize)

	plt.tight_layout()
	return fig


#corner plot subroutine
def plot_corner(posterior, plot_keys, injection_params=None, plot_ranges=None, **kwargs):

	import corner
	
	corner_kwargs = dict(bins=25, smooth=0.9, color='C0', truth_color='k', hist_bin_factor=3,
	                     quantiles=[0.05, 0.5, 0.95], levels=(0.5, 0.9), show_titles=True,
	                     plot_density=False, plot_datapoints=False, fill_contours=True, max_n_ticks=4, hist_kwargs=dict(density=True),
	                     label_kwargs=dict(fontsize=20), title_kwargs=dict(fontsize=20), tick_fontsize=18)

	#update default_kwargs with input kwargs
	corner_kwargs.update(kwargs)

	#obtain the number of samples in posterior
	n_samples = max([len(posterior.get(key, [0])) for key in plot_keys])

	#loop over plot keys to make samples to plot and plot ranges
	samples = []
	if plot_ranges is None: plot_ranges = [None for _ in plot_keys]
	for ik, key in enumerate(plot_keys):
		#if posterior exists and makes sense, just add it to the list
		if key in posterior.keys() and len(posterior[key])==n_samples and (np.amin(posterior[key])<np.amax(posterior[key])):
			samples.append(posterior[key])
		#if not, we have to make up some dummy data
		else:
			#if no plot range is given for this variable, set it to [0,1]
			if plot_ranges[ik] is None: plot_ranges[ik] = [0, 1]
			#extract upper and lower limit of plot
			xmin, xmax = plot_ranges[ik]
			#distribute dummy data uniformly in (xmin-10*width, xmin-8*with)
			samples.append(np.linspace(xmin - 10*(xmax - xmin), xmin - 8*(xmax - xmin), n_samples))

	#create numpy array with the samples to plot
	samples = np.transpose(samples)

	#make the labels
	labels = [parameter_names[key] if key in parameter_names.keys() else key for key in plot_keys]

	#extract the truths from injection parameters
	if injection_params is not None:
		truths = []
		for ik, key in enumerate(plot_keys):
			#if the key is in injection_params, add it to the list, otherwise, put truth outside plot
			if key in injection_params.keys(): truths.append(injection_params[key])
			else:                              truths.append(np.amin(samples[:,ik]) - np.amax(samples[:,ik]))
	else:   truths = None

	#if required, create titles manually
	if corner_kwargs['show_titles'] and ('titles' not in corner_kwargs.keys()):
		titles = list()
		for label, key in zip(labels, plot_keys):
			if key in posterior.keys():
				titles.append(label + ' = ' + median_and_error_text(posterior[key], quantiles=corner_kwargs['quantiles']))
			else:
				titles.append('')
		corner_kwargs['titles'] = titles
		corner_kwargs['title_fmt'] = None	

	#make the corner plot
	fig = corner.corner(samples, truths=truths, labels=labels, **corner_kwargs)

	#set the size of the x-ticks to the desired size
	for ax in fig.axes:
		ax.tick_params(axis="both", which="major", labelsize=corner_kwargs['tick_fontsize'])
	
	#set the correct plot ranges
	for ik1, k1 in enumerate(plot_keys):
		for ik2, k2 in enumerate(plot_keys):
			iax = ik1 + ik2*len(plot_keys)
			if ik1==ik2:
				if plot_ranges[ik1] is not None: fig.axes[iax].set_xlim(plot_ranges[ik1])
			else:
				if plot_ranges[ik1] is not None: fig.axes[iax].set_xlim(plot_ranges[ik1])
				if plot_ranges[ik2] is not None: fig.axes[iax].set_ylim(plot_ranges[ik2])

	return fig

#make multiple cornerplots in the same figure
def plot_multiple_in_corner(posteriors, plot_keys, plot_runs=None, injection_params=None, **kwargs):
	
	from matplotlib.lines import Line2D
	
	#if plot_runs is not given, plot all
	if plot_runs is None: plot_runs = posteriors.keys()

	extra_corner_kwargs = dict(hist_bin_factor=2, show_titles=False, fill_contours=False, quantiles=[0.05, 0.95], legend_fontsize=22)
	extra_corner_kwargs.update(kwargs)
	
	#make the ranges to plot all the variables tight
	plot_ranges = []
	for key in plot_keys:
		xmins = [np.amin(posteriors[run][key]) for run in plot_runs if key in posteriors[run]]
		xmaxs = [np.amax(posteriors[run][key]) for run in plot_runs if key in posteriors[run]]
		if xmins and xmaxs:
			min_x, max_x = np.amin(xmins), np.amax(xmaxs)
			if min_x<max_x: plot_ranges.append([min_x, max_x])
			else          : plot_ranges.append([min_x-0.5, min_x+0.5])
		else: plot_ranges.append([0., 1.])

	#loop over runs
	fig = None
	handles = []
	for i_run, run in enumerate(plot_runs):
		#add this run to the corner
		fig = plot_corner(posteriors[run], plot_keys, injection_params=injection_params[run], plot_ranges=plot_ranges, fig=fig, color='C%s'%(i_run), **extra_corner_kwargs)
		#create handles for legend
		handles.append(Line2D([], [], color='C%s'%(i_run), linestyle='-', label=run))
	
	fig.legend(handles=handles, fontsize=extra_corner_kwargs['legend_fontsize'], loc='upper right')
	
	return fig

