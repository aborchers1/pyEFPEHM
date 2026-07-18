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
import multiprocessing as mp
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
	e20 = p_pyEFPE['eccentricity']*p_pyEFPE['eccentricity']
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

# Define function to compute mismatch varying specified parameters and using batched EFPE waveforms
def batched_mismatch_t_ph_amp_pol_minimized_varying_params(param_vals, param_names, signal, freqs, pyEFPE_params, EFPE, asd, return_optimum_params=False, batch_size=100000, fft_padding_fact=4):

	#update required pyEFPE parameters
	p_pyEFPE = update_pyEFPE_params(pyEFPE_params, param_vals, param_names)

	#compute waveform
	try:    hp_pyEFPE, hc_pyEFPE = batched_pyEFPE_generate_waveform(EFPE.pyEFPE(p_pyEFPE), freqs, batch_size=batch_size)
	except: raise Exception("Waveform generation failed with \nparams=%s"%(p_pyEFPE))

	#return the mismatch analytically minimized over polarization
	return mismatch_t_ph_amp_pol_minimized(signal, hp_pyEFPE, hc_pyEFPE, freqs, fft_padding_fact=fft_padding_fact, return_optimum_params=return_optimum_params, asd=asd)

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
			if pname in ['phase', 'phase_s', 'phase_s1', 'phase_s2', 'mean_anomaly', 'inclination', 'pol']:
				periodic_dims.append(ip)
				if pname in ['pol']: periods.append(  np.pi)
				else :               periods.append(2*np.pi)

		#minimize mismatch using bayesian optimization
		minimize_result = bayesian_minimization(batched_mismatch_t_ph_amp_pol_minimized_varying_params, parameter_bounds,
		                                        args=(minimize_parameters, signal, freqs, pyEFPE_params, EFPE, asd),
		                                        kwargs=dict(return_optimum_params=False, batch_size=batch_size, fft_padding_fact=fft_padding_fact),
		                                        periodic_dims=periodic_dims, periods=periods,
		                                        n_initial=n_initial, n_calls=n_calls, verbose=disp)

		#extract information from minimization
		params_min = minimize_result['x_best']

	else:
		params_min = []

	#reconstruct the parameters that were analytically maximized
	min_mm, dt_shift_sh, ang_sh, amp_sh, pol_h = batched_mismatch_t_ph_amp_pol_minimized_varying_params(params_min, minimize_parameters, signal, freqs, pyEFPE_params, EFPE, asd, return_optimum_params=True, batch_size=batch_size, fft_padding_fact=fft_padding_fact)

	#return the minimum mismatch and the quantities required to reconstruct the best fitting waveform
	return min_mm, dt_shift_sh, ang_sh, amp_sh, pol_h, params_min

###########################################################################

#store the different comparisons
comparison_info = {}

comparison_info["HorizonAbsorption"] = dict(
pyEFPE_1 = pyEFPEHM,
pyEFPE_2 = pyEFPEHM,
params_pyEFPE_1 = {'horizon_absorption': True,},
update_params_pyEFPE_2 = {'horizon_absorption': False,},
minimize_parameters = ['phase', 'phase_s', 'mean_anomaly'],
e_range   = (0., 0.6),
q_range   = (0.05, 1.),
s_range = (0., 1.),
dphase_function = horizon_absorption_dphasing,
dphase_label  = r'$\frac{5}{256} \frac{\tilde{\kappa}_{H,0}}{\nu} \log{\frac{y_f}{y_0}}$',
)

#store the information about the different interferometers
ifo_info = {}

ifo_info['LIGO_Aplus'] = dict(
mc_range = (1., 10.),
f22_start = 10,
f_min = 10,
f_max = 2048,
max_seglen = 2**12,
max_duration = None,
psd_name = 'AplusDesign',
fancy_name = 'LIGO A+',
)

ifo_info['ET10kmHFLF'] = dict(
mc_range = (1., 100.),
f22_start = 1,
f_min = 1,
f_max = 2048,
max_seglen = 2**12,
max_duration = None,
psd_name = 'ET10kmHFLF',
fancy_name = 'ET (10km HF+LF)',
)

ifo_info['LISA4yr'] = dict(
mc_range = (10., 1e5),
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

#number of samples
Nsamples = 1024

#number of cores to use
nworkers = 8    #max(mp.cpu_count() - 1, 1)

#interferometer to consider
ifo = 'LIGO_Aplus'

#output directory
outdir = './outdir/pyEFPE_version_comparisons/'
plotdir = outdir

#fft padding factor for mismatch time minimization
fft_padding_fact=2
batch_size = 100000

#Number of points to use in bayesian optimization
n_initial = 32
n_calls   = 100

############################################################################

#extract interferometer information
mc_range     = ifo_info[ifo]['mc_range']
f_min        = ifo_info[ifo]['f_min']
f_max        = ifo_info[ifo]['f_max']
f22_start    = ifo_info[ifo]['f22_start']
max_seglen   = ifo_info[ifo]['max_seglen']
max_duration = ifo_info[ifo]['max_duration']
psd_name     = ifo_info[ifo]['psd_name']

#default parameters for pyEFPE
params_pyEFPE_1 = comparison_info[comparison_name]['params_pyEFPE_1']

#parameters to update in pyEFPE_2
update_params_pyEFPE_2 = comparison_info[comparison_name]['update_params_pyEFPE_2']

#parameters to update
minimize_parameters = comparison_info[comparison_name]['minimize_parameters']

#extract the different pyEFPEs
pyEFPE_1 = comparison_info[comparison_name]['pyEFPE_1']
pyEFPE_2 = comparison_info[comparison_name]['pyEFPE_2']

#extract q and spin range, as well as the number of samples
e_range = comparison_info[comparison_name]['e_range']
q_range = comparison_info[comparison_name]['q_range']
s_range = comparison_info[comparison_name]['s_range']

#function to compute dphasing
dphase_function = comparison_info[comparison_name].get('dphase_function', None)
dphase_label    = comparison_info[comparison_name].get('dphase_label', None)

#if required, compute the ASD
if type(psd_name)==str:
	freqs_arr   = np.arange(f_min, f_max, 1./max_seglen)
	asd_arr     = compute_asd(1./max_seglen, f_min, f_max, psd_name=psd_name)
	asd_cspline = CubicSpline(freqs_arr, asd_arr)
else:
	asd = None

#name of the result
string_id = '_pyEFPE_comparison_paramspace_%s_N_%s_f_%.3g_%.3g_Tmax_%.3g'%(comparison_name, Nsamples, f_min, f_max, max_seglen)
if type(psd_name)==str: string_id += '_psd_'+psd_name
result_filename = outdir+'/result'+string_id+'.pickle'

#try to load the result
try:
	#load the result dictionary
	with open(result_filename, 'rb') as handle: result = pickle.load(handle)
	
	#make sure it is the same
	assert result['e_range']==e_range
	assert result['q_range']==q_range
	assert result['s_range']==s_range

except:

	print('\nCould not load '+result_filename)
	
	#generate random parameters
	params_low  = [np.log(mc_range[0]), 1./q_range[1], s_range[0], s_range[0], e_range[0], *([-1.]*3),      *([0.]*4),    0.]
	params_high = [np.log(mc_range[1]), 1./q_range[0], s_range[1], s_range[1], e_range[1], *([ 1.]*3), *([2*np.pi]*4), np.pi]
	log_mc, inv_q, s1, s2, ecc, cos_iota, cos_ths1, cos_ths2, phs1, phs2, phiref, mean_anomaly, pol = np.transpose(np.random.uniform(params_low, params_high, size=(Nsamples,len(params_low))))
	
	#undo sampling conversions
	mc   = np.exp(log_mc)
	q    = 1./inv_q
	iota, ths1, ths2 = np.arccos([cos_iota, cos_ths1, cos_ths2])

	#parameters entering the waveform
	m1 = mc*(q**-0.6)*((1 + q)**0.2)
	m2 = m1*q
	s1x, s1y, s1z = spherical_to_cartesian([s1, ths1, phs1])
	s2x, s2y, s2z = spherical_to_cartesian([s2, ths2, phs2])

	#compute the effective inspiral spin parameter
	chi_eff = (s1z + q*s2z)/(1 + q)
	
	#compute the precessing spin parameter
	s1p = np.sqrt(s1x*s1x + s1y*s1y)
	s2p = np.sqrt(s2x*s2x + s2y*s2y)
	chi_p = np.maximum(s1p, (q*(4*q + 3)/(4 + 3*q))*s2p)

	#Compute the minimum mismatch (and params/diagnostics) for sample i
	def compute_mismatch_for_sample(i):

		#reseed so each forked worker draws independent randomness in the BO
		np.random.seed()

		#put the params to pass to pyEFPE in dictionary
		params1 = params_pyEFPE_1.copy()
		params1.update({
			'mass1' : m1[i],  'mass2' : m2[i],
			'spin1x': s1x[i], 'spin1y': s1y[i], 'spin1z': s1z[i],
			'spin2x': s2x[i], 'spin2y': s2y[i], 'spin2z': s2z[i],
			'eccentricity': ecc[i], 'mean_anomaly': mean_anomaly[i],
			'inclination': iota[i], 'phase': phiref[i],
		})

		if max_duration is not None:
			params1['f22_start'] = find_f22_start_for_duration(max_duration, f22_start, f_max, mc[i], ecc[i])
		else:
			params1['f22_start'] = f22_start

		#update necessary params for pyEFPE_2
		params2 = params1.copy()
		params2.update(update_params_pyEFPE_2)

		#initialize first pyEFPE waveform and compute its duration and segment length
		wf1 = pyEFPE_1.pyEFPE(params1)
		duration = wf1.sol.all_ts[-1] - wf1.sol.all_ts[0]
		seglen   = min(max_seglen, 2**(np.ceil(np.log2(duration))))

		#initialize frequencies
		freqs = np.arange(f_min, f_max, 1./seglen)

		#generate polarizations for first EFPE
		hp_1, hc_1 = batched_pyEFPE_generate_waveform(wf1, freqs)
		#compute the reference waveform
		h_1 = pol_response(hp_1, hc_1, pol[i])

		#compute ASD
		if type(psd_name)==str: asd = asd_cspline(freqs)
		else:                   asd = None

		#compute the minimum mismatch
		mismatch = batched_bo_minimize_mismatch(h_1, freqs, params2, minimize_parameters, EFPE=pyEFPE_2, asd=asd, n_initial=n_initial, n_calls=n_calls, fft_padding_fact=fft_padding_fact, disp=False)[0]

		print('%s/%s -> MM: %.3g\nparams: %s\n'%(i+1, Nsamples, mismatch, params1))

		return mismatch, params1, params2, duration, seglen


	#compute mismatches in parallel
	with mp.Pool(nworkers) as pool:
		results = pool.map(compute_mismatch_for_sample, range(Nsamples), chunksize=1)

	#unpack results
	mismatches           = np.array([r[0] for r in results])
	list_params_pyEFPE_1 = [r[1] for r in results]
	list_params_pyEFPE_2 = [r[2] for r in results]
	durations            = np.array([r[3] for r in results])
	seglens              = np.array([r[4] for r in results])

	#make result dictionary
	result = {'e_range': e_range, 'q_range': q_range, 's_range': s_range, 'durations': durations, 'seglens': seglens, 'm1': m1, 'm2': m2, 's1x': s1x, 's1y': s1y, 's1z': s1z, 's2x': s2x, 's2y': s2y, 's2z': s2z, 'ecc': ecc, 'mean_anomaly': mean_anomaly, 'iota': iota, 'phiref': phiref, 'pol': pol, 'mc': mc, 'q': q, 'chi_eff': chi_eff, 'chi_p': chi_p, 'list_params_pyEFPE_1': list_params_pyEFPE_1, 'list_params_pyEFPE_2': list_params_pyEFPE_2, 'MM': mismatches}

	#save the result file
	if not os.path.exists(outdir):  os.makedirs(outdir)
	with open(result_filename, 'wb') as handle: pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

# if required, compute dephasings
if dphase_function is not None:
	dphases = np.zeros(Nsamples)
	for i in range(Nsamples):
		dphases[i] = dphase_function(result['list_params_pyEFPE_1'][i])

#########################################################################

# Plotting inputs
min_abs_dphase = 1e-2
marker_s = 40
n_fit = 2

if not os.path.exists(plotdir): os.makedirs(plotdir)

from matplotlib import colors
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

#make a histogram of the mismatches
bins = np.geomspace(np.nanmin(result['MM']), np.nanmax(result['MM']), 25)
plt.figure(figsize=(12,8))
plt.hist(result['MM'], bins=bins)
plt.xlabel(r'Mismatch')
plt.ylabel(r'Number of waveforms')
plt.xscale('log')
plt.xlim(bins[0], bins[-1])
plt.tight_layout()

#make a scatter plot of dephasing vs mismatch
if dphase_function is not None:
	
	# scatter plot of mismatch vs dephasing with Mc color
	plt.figure(figsize=(12,8))
	plt.scatter(dphases, result['MM'], c=result['mc'], norm=colors.LogNorm(vmin=mc_range[0], vmax=mc_range[1]))
	plt.xlabel(dphase_label)
	plt.ylabel(r'$\overline{\mathcal{MM}}$')
	plt.colorbar(label=r'$\mathcal{M}_c \; [M_\odot]$')
	plt.tight_layout()

	# scatter plot of mismatch vs |dephasing| with sign(dephasing) marker and Mc color
	min_abs_dphase = max(min_abs_dphase, np.amin(np.abs(dphases)))
	ipos = dphases>min_abs_dphase
	ineg = dphases<-min_abs_dphase
	plt.figure(figsize=(9,8))
	plt.scatter(dphases[ipos], result['MM'][ipos], c=result['mc'][ipos], s=marker_s, marker='o', norm=colors.LogNorm(vmin=mc_range[0], vmax=mc_range[1]))
	plt.scatter(np.abs(dphases[ineg]), np.abs(result['MM'][ineg]), c=result['mc'][ineg], s=marker_s, marker='x', norm=colors.LogNorm(vmin=mc_range[0], vmax=mc_range[1]))
	from matplotlib.lines import Line2D
	handles = [Line2D([], [], color='k', marker='o', linestyle='None', markersize=np.sqrt(marker_s), label='$\Delta \lambda_\mathrm{th} > 0$'),
	           Line2D([], [], color='k', marker='x', linestyle='None', markersize=np.sqrt(marker_s), label='$\Delta \lambda_\mathrm{th} < 0$')]
	# best fit of log(MM) = n*log(|dphase|) + c with the slope n fixed
	ivalid = (ipos | ineg) & np.isfinite(result['MM']) & (result['MM']>0)
	if np.any(ivalid):
		abs_dphase = np.abs(dphases[ivalid])
		c_fit = np.median(np.log(np.abs(result['MM'][ivalid])) - n_fit*np.log(abs_dphase))
		xfit = np.array([abs_dphase.min(), abs_dphase.max()])
		line_fit, = plt.plot(xfit, np.exp(c_fit)*xfit**n_fit, 'k--', lw=3, label=r'$\overline{\mathcal{MM}} = %.3g \left|\Delta \lambda_\mathrm{th} \right|^{%s}$'%(np.exp(c_fit), n_fit))
		handles.append(line_fit)
		plt.xlim(*xfit)

	plt.title(ifo_info[ifo]['fancy_name'])
	plt.xlabel(r'$\left| \Delta \lambda_\mathrm{th} \right| = \left| %s \right|$'%(dphase_label.strip('$')))
	plt.ylabel(r'$\overline{\mathcal{MM}}$')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend(handles=handles, loc='lower right', borderaxespad=0.35, fontsize=0.8*plt.rcParams['legend.fontsize'])
	plt.colorbar(label=r'$\mathcal{M}_c \; [M_\odot]$')
	plt.tight_layout()
	plt.savefig(plotdir+'/MM_absdeltalambda'+string_id+'.pdf')


#Runtime
print("\nRuntime: %s seconds" % (time.time() - start_runtime))

plt.show()
