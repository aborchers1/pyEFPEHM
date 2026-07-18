import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

try:    import pyEFPE
except: print("Warning: could not import pyEFPE")

try:    import pyEFPEHM
except: print("Warning: could not import pyEFPEHM")

import numpy as np
import pickle
from utils_compute_mismatches import *
from scipy.interpolate import CubicSpline

# pybop downloaded from https://github.com/gmorras/pybop
from pybop import bayesian_minimization

#filter some warnings
import warnings
warnings.filterwarnings("ignore", message="Waveform has no stationary times", category=UserWarning)

import time
start_runtime = time.time()

#universal constants
t_sun_s = 4.92549094831e-6  #GMsun/c**3 [s]

#function to compute the dephasing due to horizon absortion
def horizon_absorption_dphasing(p_pyEFPE):
	
	#compute mass related things
	m1, m2 = t_sun_s*p_pyEFPE['mass1'], t_sun_s*p_pyEFPE['mass2']
	M, mu1, mu2, nu, dmu = pyEFPEHM.utils.mass_params_from_m1_m2(m1, m2)
	
	#compute reduced spins
	s1_vec = mu1*np.array([p_pyEFPE['spin1x'], p_pyEFPE['spin1y'], p_pyEFPE['spin1z']])
	s2_vec = mu2*np.array([p_pyEFPE['spin2x'], p_pyEFPE['spin2y'], p_pyEFPE['spin2z']])
	
	#compute spin scalars
	chi_eff = s1_vec[2] + s2_vec[2]
	dchi    = s1_vec[2] - s2_vec[2]
	s2_1    = np.sum(np.square(s1_vec))
	s2_2    = np.sum(np.square(s2_vec))

	#compute spin factor of horizon absortion term
	kHA = (dmu + (9./8.)*(s2_1 - s2_2))*dchi + chi_eff*(1 - 2*nu + (9./8.)*(s2_1 + s2_2) + (45./16.)*dchi*dchi + (15./16.)*chi_eff*chi_eff)

	#compute initial PN parameter
	e20 = p_pyEFPE['e_start']*p_pyEFPE['e_start']
	y0 = ((np.pi*M*p_pyEFPE['f22_start'])**(1/3))/np.sqrt(1 - e20)
	#compute final PN parameter
	if 'f22_end' in p_pyEFPE:
		yf = min((np.pi*M*p_pyEFPE['f22_end'])**(1/3), 6**-0.5)
	else:
		yf = 6**-0.5
	
	return ((5./32.)**2)*(4./5.)*kHA*np.log(yf/y0)/nu

#function to compute EFPE waveform in batches
def batched_pyEFPE_generate_waveform(EFPE_class, freqs, batch_size=100000):
	
	#if len(freqs) is below batch size, just apply waveform
	if len(freqs)<=batch_size:
		return EFPE_class.generate_waveform(freqs)
	
	#compute waveform polarizations in batches
	hphc = np.empty((2, len(freqs)), dtype=np.complex128)
	for i in range(0, len(freqs), batch_size):
		hphc[:,i:i+batch_size] = EFPE_class.generate_waveform(freqs[i:i+batch_size])
	
	return hphc

# Define function to compute log(mismatch) varying specified parameters and using batched EFPE waveforms
# We compute the logarithm of the mismatch because it will be what we actually show in plots, so the relative/absolute error of this quantity is more meaningful
def batched_logmismatch_t_ph_amp_pol_minimized_varying_params(param_vals, param_names, signal, freqs, pyEFPE_params, EFPE, asd, return_optimum_params=False, batch_size=100000, fft_padding_fact=4):

	#update required pyEFPE parameters
	p_pyEFPE = update_pyEFPE_params(pyEFPE_params, param_vals, param_names)

	#compute waveform
	try:    hp_pyEFPE, hc_pyEFPE = batched_pyEFPE_generate_waveform(EFPE.pyEFPE(p_pyEFPE), freqs, batch_size=batch_size)
	except: raise Exception("Waveform generation failed with \nparams=%s"%(p_pyEFPE))

	#return the log mismatch analytically minimized over polarization
	mm_result = mismatch_t_ph_amp_pol_minimized(signal, hp_pyEFPE, hc_pyEFPE, freqs, fft_padding_fact=fft_padding_fact, return_optimum_params=return_optimum_params, asd=asd)
	if return_optimum_params:
		min_mm, dt_shift_sh, ang_sh, amp_sh, pol_h = mm_result
		return np.log(min_mm), dt_shift_sh, ang_sh, amp_sh, pol_h
	else:
		return np.log(mm_result)

#function to minimize the mismatch over a specified list of minimize_parameters with certain specified parameter ranges using bayesian optimization and batched EFPE waveforms
def batched_bo_minimize_mismatch(signal, freqs, pyEFPE_params, minimize_parameters, EFPE=pyEFPEHM, rtol_e=0.3, rtol_p=0.1, max_e=0.85, asd=None, disp=True, n_initial=16, n_calls=100, batch_size=100000, fft_padding_fact=4):

	#use differential_evolution only if there are parameters to minimize over
	if len(minimize_parameters)>=1:

		#generate parameter bounds
		parameter_bounds = determine_pyEFPE_bounds(minimize_parameters, pyEFPE_params, rtol_e=rtol_e, rtol_p=rtol_p, max_e=max_e)

		#find the dimensions along which problem is periodic and their periods
		periodic_dims = list()
		periods = list()
		for ip, pname in enumerate(minimize_parameters):
			if pname in ['phi_start', 'phase_s', 'phase_s1', 'phase_s2', 'mean_anomaly_start', 'inclination', 'pol']:
				periodic_dims.append(ip)
				if pname in ['pol']: periods.append(  np.pi)
				else :               periods.append(2*np.pi)

		#minimize log mismatch using bayesian optimization
		minimize_result = bayesian_minimization(batched_logmismatch_t_ph_amp_pol_minimized_varying_params, parameter_bounds,
		                                        args=(minimize_parameters, signal, freqs, pyEFPE_params, EFPE, asd),
		                                        kwargs=dict(return_optimum_params=False, batch_size=batch_size, fft_padding_fact=fft_padding_fact),
		                                        periodic_dims=periodic_dims, periods=periods,
		                                        n_initial=n_initial, n_calls=n_calls, verbose=disp)

		#extract information from minimization
		params_min = minimize_result['x_best']

	else:
		params_min = []

	#reconstruct the parameters that were analytically maximized
	log_min_mm, dt_shift_sh, ang_sh, amp_sh, pol_h = batched_logmismatch_t_ph_amp_pol_minimized_varying_params(params_min, minimize_parameters, signal, freqs, pyEFPE_params, EFPE, asd, return_optimum_params=True, batch_size=batch_size, fft_padding_fact=fft_padding_fact)

	#return the minimum mismatch and the quantities required to reconstruct the best fitting waveform
	return np.exp(log_min_mm), dt_shift_sh, ang_sh, amp_sh, pol_h, params_min

###########################################################################

#store the different comparisons
comparison_info = {}

comparison_info["HorizonAbsorption"] = dict(
pyEFPE_1 = pyEFPEHM,
pyEFPE_2 = pyEFPEHM,
params_pyEFPE_1 = {'horizon_absorption': True,},
update_params_pyEFPE_2 = {'horizon_absorption': False,},
minimize_parameters = ['phi_start', 'phase_s', 'mean_anomaly_start'],
base_pyEFPE_params = dict(spin1x=-0.2, spin1y=0.3, spin1z=0.9, spin2x=-0.4, spin2y=-0.4, spin2z=0.8, mean_anomaly_start=4.3, inclination=np.pi/3, phi_start=1.2, distance=100.),
pol = 0.5*np.pi,
qs = np.geomspace(0.05, 1, 20),
es = np.linspace(0, 0.7, 15),
norm_dphase_function = horizon_absorption_dphasing,
normed_dphase_label  = r'$\Delta \lambda/\Delta \lambda^\mathrm{QC}_\mathrm{H}$',
)

#store the information about the different interferometers
ifo_info = {}

ifo_info['LIGO_Aplus'] = dict(
mcs = np.geomspace(1, 10, 4),
f22_start = 10,
f_min = 10,
f_max = 2048,
max_seglen = 2**12,
max_duration = None,
psd_name = 'AplusDesign',
fancy_name = 'LIGO A+',
)

ifo_info['ET10kmHFLF'] = dict(
mcs = np.geomspace(1, 100, 4),
f22_start = 1,
f_min = 1,
f_max = 2048,
max_seglen = 2**12,
max_duration = None,
psd_name = 'ET10kmHFLF',
fancy_name = 'ET (10km HF+LF)',
)

ifo_info['LISA4yr'] = dict(
mcs = np.geomspace(10, 100000, 4),
f22_start = 1e-4,
f_min = 1e-4,
f_max = 1,
max_seglen   = 2**21,
max_duration = 4*365.25*24*3600,
psd_name = 'LISA4yr1803.01944',
fancy_name = 'LISA (4yr)',
)

############################################################################

#type of comparison
comparison_name = 'HorizonAbsorption'

#interferometer to consider
ifo = 'LISA4yr'

#output directory
outdir = './outdir/pyEFPE_version_comparisons/'
plotdir = outdir

#fft padding factor for mismatch time minimization
fft_padding_fact=2
batch_size = 100000

#Number of points to use in bayesian optimization
n_initial = 16
n_calls   = 50

############################################################################

#extract interferometer information
mcs          = ifo_info[ifo]['mcs']
f_min        = ifo_info[ifo]['f_min']
f_max        = ifo_info[ifo]['f_max']
f22_start    = ifo_info[ifo]['f22_start']
max_seglen   = ifo_info[ifo]['max_seglen']
max_duration = ifo_info[ifo]['max_duration']
psd_name     = ifo_info[ifo]['psd_name']

#default parameters for pyEFPE
params_pyEFPE_1 = comparison_info[comparison_name]['params_pyEFPE_1']
params_pyEFPE_1.update(comparison_info[comparison_name]['base_pyEFPE_params'])

#parameters to update in pyEFPE_2
update_params_pyEFPE_2 = comparison_info[comparison_name]['update_params_pyEFPE_2']

#parameters to update
minimize_parameters = comparison_info[comparison_name]['minimize_parameters']

#extract the different pyEFPEs
pyEFPE_1 = comparison_info[comparison_name]['pyEFPE_1']
pyEFPE_2 = comparison_info[comparison_name]['pyEFPE_2']

#extract qs, es and pol
qs  = comparison_info[comparison_name]['qs']
es  = comparison_info[comparison_name]['es']
pol = comparison_info[comparison_name]['pol']

#function to compute dphasing
norm_dphase_function = comparison_info[comparison_name].get('norm_dphase_function', None)
normed_dphase_label  = comparison_info[comparison_name].get('normed_dphase_label', None)

#if required, compute the ASD
if type(psd_name)==str:
	freqs_arr   = np.arange(f_min, f_max, 1./max_seglen)
	asd_arr     = compute_asd(1./max_seglen, f_min, f_max, psd_name=psd_name)
	asd_cspline = CubicSpline(freqs_arr, asd_arr)
else:
	asd = None

#loop over chirp masses to consider
MMs = np.zeros((len(mcs), len(qs), len(es)))
dphases = np.zeros((len(mcs), len(qs), len(es)))
for imc, mc in enumerate(mcs):

	#name of the result
	string_id = '_pyEFPE_of_q_e_compare_%s_Mc_%.4g_f_%.3g_%.3g_Tmax_%.3g'%(comparison_name, mc, f_min, f_max, max_seglen)
	if type(psd_name)==str: string_id += '_psd_'+psd_name
	result_filename = outdir+'/result'+string_id+'.pickle'

	#try to load the result
	try:
		#load the result dictionary
		with open(result_filename, 'rb') as handle: result = pickle.load(handle)
		
		#make sure it is the same
		assert np.all(result['qs']==qs)
		assert np.all(result['es']==es)

	#otherwise, create the result
	except:

		print('\nCould not load '+result_filename)

		#generate waveforms and compute their mismatches
		print('\nmc=%.3g'%(mc))
		mismatches = np.zeros((len(qs), len(es)))
		dephasings = np.zeros((len(qs), len(es)))
		for iq, q in enumerate(qs):
			for ie, e in enumerate(es):
			
				#compute component masses in solar masses
				m1 = mc*(q**-0.6)*((1 + q)**0.2)
				m2 = m1*q
			
				#put the params to pass to pyEFPE in dictionary
				params_pyEFPE_1['mass1'] = m1
				params_pyEFPE_1['mass2'] = m2
				params_pyEFPE_1['e_start'] = e
			
				if max_duration is not None:
					params_pyEFPE_1['f22_start'] = find_f22_start_for_duration(max_duration, f22_start, f_max, mc, e)
				else:
					params_pyEFPE_1['f22_start'] = f22_start
			
				#update necessary params for pyEFPE_2
				params_pyEFPE_2 = params_pyEFPE_1.copy()
				params_pyEFPE_2.update(update_params_pyEFPE_2)

				#Initialize pyEFPE waveforms
				wf1 = pyEFPE_1.pyEFPE(params_pyEFPE_1)
				wf2 = pyEFPE_2.pyEFPE(params_pyEFPE_2)

				#compute the waveform durations
				duration1 = wf1.sol.all_ts[-1] - wf1.sol.all_ts[0]
				duration2 = wf2.sol.all_ts[-1] - wf2.sol.all_ts[0]
				duration  = max(duration1, duration2)
				seglen    = min(max_seglen, 2**(np.ceil(np.log2(duration))))
				
				#initialize frequencies
				freqs = np.arange(f_min, f_max, 1./seglen)

				#compute the dephasings
				phase1_0, phase1_f = wf1.sol([wf1.sol.all_ts[0], wf1.sol.all_ts[-1]], idxs=2)
				phase2_0, phase2_f = wf2.sol([wf2.sol.all_ts[0], wf2.sol.all_ts[-1]], idxs=2)
				dephasings[iq,ie] = (phase1_f - phase1_0) - (phase2_f - phase2_0)
				del wf2

				#generate polarizations for first EFPE
				hp_1, hc_1 = batched_pyEFPE_generate_waveform(wf1, freqs)
				#compute the reference waveform
				h_1 = pol_response(hp_1, hc_1, pol)
				del wf1, hp_1, hc_1

				#compute ASD
				if type(psd_name)==str: asd = asd_cspline(freqs)

				#compute and store the minimum mismatch
				mismatches[iq,ie] = batched_bo_minimize_mismatch(h_1, freqs, params_pyEFPE_2, minimize_parameters, EFPE=pyEFPE_2, asd=asd, n_initial=n_initial, n_calls=n_calls, fft_padding_fact=fft_padding_fact)[0]

				#print information
				if max_duration is not None:
					print('q=%.3g, e=%.3g, f0=%.3g -> duration=%.3gs, seglen=%.3gs, dphase=%.4g, mm=%.4g'%(q, e, params_pyEFPE_1['f22_start'], duration, seglen, dephasings[iq,ie], mismatches[iq,ie]))
				else:
					print('q=%.3g, e=%.3g -> duration=%.3gs, seglen=%.3gs, dphase=%.4g, mm=%.4g'%(q, e, duration, seglen, dephasings[iq,ie], mismatches[iq,ie]))
			
		#make result dictionary
		result = {'MM': mismatches, 'dphase': dephasings, 'qs': qs, 'es': es,}
		
		#save the result file
		if not os.path.exists(outdir):  os.makedirs(outdir)
		with open(result_filename, 'wb') as handle: pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

	#append mismatches to full array
	MMs[imc] = result['MM']
	dphases[imc] = result['dphase']

#compute normalized dphasings
if norm_dphase_function is not None:
	normed_dphases = np.zeros_like(dphases)
	for imc, mc in enumerate(mcs):
		for iq, q in enumerate(qs):
			for ie, e in enumerate(es):

				#compute component masses in solar masses
				m1 = mc*(q**-0.6)*((1 + q)**0.2)
				m2 = m1*q

				#put the params to pass to pyEFPE in dictionary
				params_pyEFPE_1['mass1'] = m1
				params_pyEFPE_1['mass2'] = m2
				params_pyEFPE_1['e_start'] = e

				if max_duration is not None:
					params_pyEFPE_1['f22_start'] = find_f22_start_for_duration(max_duration, f22_start, f_max, mc, e)
				else:
					params_pyEFPE_1['f22_start'] = f22_start

				#compute normalized dphasing
				normed_dphases[imc,iq,ie] = dphases[imc,iq,ie]/norm_dphase_function(params_pyEFPE_1)

#########################################################################

if not os.path.exists(plotdir): os.makedirs(plotdir)

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

#function to print numbers prettily
def pretty_number_formating(x):
	
	if (x<=1000) and (x>=0.001): return r'%.3g'%(x)
	else:
		exponent = np.floor(np.log10(x))
		mantissa = x*(10**-exponent)

		if np.isclose(mantissa, 1):
			return r'10^{%.f}'%(exponent)
		else:
			return r'%.3g \cdot 10^{%.f}'%(mantissa, exponent)

#function to make ticks in log-scale plot
def log_labeler(log10x):
	x = 10**log10x
	if x>=100 or x<=0.01:
		power = np.floor(log10x)
		return r"$%.1f \cdot 10^{%.f}$"%(x*(10**-power), power)
	elif x>=  10: return r"$%.f$"%(x)
	elif x>=   1: return r"$%.1f$"%(x)
	elif x>= 0.1: return r"$%.2f$"%(x)
	elif x>=0.01: return r"$%.3f$"%(x)
	else:
		raise Exception("x=%s was not picked up by any case"%(x))


#function to make plots
def plot_x_mcs_qs_es(xplot, mcs, qs, es, nrows=None, ncols=None, plot_scale=None, zlabel=None, cmap='viridis', Nlevels=10, title=None, filename=None):
	
	#choose the number of rows/columns if not given
	if (nrows is None) or (ncols is None): nrows, ncols = 1, len(mcs)
	
	#make figure
	fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols+1, 5*nrows), sharex=True, sharey=True, constrained_layout=True, squeeze=False)

	#choose to make plot logarithmic
	if plot_scale=='log': xplot = np.log10(xplot)
	
	#levels to plot the data
	levels = np.linspace(np.amin(xplot), np.amax(xplot), Nlevels)
	
	#things for the panels
	for icol in range(ncols):
		for irow in range(nrows):
			imc = icol + irow*ncols
			cs = axes[irow,icol].contourf(qs, es, np.transpose(xplot[imc]), levels=levels, cmap=cmap)
			panel_title = r"$\mathcal{M}_c = %s M_\odot$"%(pretty_number_formating(mcs[imc]))
			if nrows==1:
				axes[irow,icol].set_title(panel_title)
			else:
				axes[irow,icol].text(0.95, 0.05, panel_title, transform=axes[irow,icol].transAxes, ha='right', va='bottom',color='white', fontsize=plt.rcParams['font.size'])
	
	#things for the x-axis
	for icol in range(ncols):
		axes[-1,icol].set_xlabel('$q$')
		axes[-1,icol].set_xscale('log')
	#things for the y-axis
	for irow in range(nrows):
		axes[irow, 0].set_ylabel('$e_0$')

	#It required, put a title
	if title is not None: fig.suptitle(title)

	#color bar
	cbar = fig.colorbar(cs, ax=axes, pad=0, label=zlabel)
	if plot_scale=='log': cbar.set_ticklabels([log_labeler(t) for t in levels])

	#save figure
	if filename is not None: fig.savefig(filename)

	return fig, axes

#name to identify plots
plot_id = '_pyEFPE_of_q_e_compare_%s_at_%s'%(comparison_name, ifo)

#make mismatch figure
plot_x_mcs_qs_es(MMs, mcs, qs, es, plot_scale='log', zlabel=r'$\overline{\mathcal{MM}}$', title=ifo_info[ifo]['fancy_name'], filename=plotdir+'/mismatches'+plot_id+'.pdf')

#make mismatch figure
plot_x_mcs_qs_es(dphases, mcs, qs, es, plot_scale='log', zlabel=r'$\Delta \lambda$')#, filename=plotdir+'/dphases'+plot_id+'.pdf')

#make normalized dephasing figure
if norm_dphase_function is not None:
	plot_x_mcs_qs_es(normed_dphases, mcs, qs, es, plot_scale='linear', zlabel=normed_dphase_label)#, filename=plotdir+'/normed_dphases'+plot_id+'.pdf')

#make a plot of all the PSDs
fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)
fmin_plot, fmax_plot = np.inf, -np.inf
for pifo in ifo_info.keys():
	pfmin = ifo_info[pifo]['f_min']
	pfmax = ifo_info[pifo]['f_max']
	fmin_plot = min(fmin_plot, pfmin)
	fmax_plot = max(fmax_plot, pfmax)
	pmaxseglen = ifo_info[pifo]['max_seglen']
	pfreqs = np.geomspace(pfmin, pfmax, 10000)
	pasd = np.interp(pfreqs,
	                 np.arange(pfmin, pfmax, 1./pmaxseglen),
	                 compute_asd(1./pmaxseglen, pfmin, pfmax, psd_name=ifo_info[pifo]['psd_name']))
	ax.plot(pfreqs, pasd, label=ifo_info[pifo]['fancy_name'], lw=2)

ax.set_xscale('log')
ax.set_xlabel(r'$f$ $[\mathrm{Hz}]$')
ax.set_xticks(10.**np.arange(-5, 5))
ax.set_xlim(fmin_plot, fmax_plot)
ax.set_yscale('log')
ax.set_ylabel(r'ASD $[\mathrm{Hz}^{-1/2}]$')
ax.legend(loc='lower left')
fig.savefig(plotdir+'/ASDs_pyEFPE_comparison_q_e.pdf')

#Runtime
print("\nRuntime: %s seconds" % (time.time() - start_runtime))

plt.show()
