import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

import numpy as np
import pickle

from utils_compute_mismatches import *

try:    import lalsimulation as lalsim
except: print("Warning: could not import lalsimulation")

try:    import lal
except: print("Warning: could not import lal")

try:    import EOBRun_module
except: print("Warning: could not import EOBRun_module (a.k.a. TEOBResumS)")

try:    import pyseobnr
except: print("Warning: could not import pyseobnr")

try:    import pyEFPE
except: print("Warning: could not import pyEFPE")

try:    import pyEFPEHM
except: print("Warning: could not import pyEFPEHM")

import time
import multiprocessing
import resource

#use fork so the child inherits the loaded modules and the (unpicklable) lal waveform dictionaries
mp_ctx = multiprocessing.get_context('fork')

#universal constants
t_sun_s = 4.92549094831e-6  #GMsun/c**3 [s]

#total physical memory of this machine in bytes
total_memory_bytes = os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')

#time a function call Ntries times in a child process, so that segfaults/OOM kills of the
#waveform generator only kill the child and show up as NaN timings instead of crashing the script
def time_function_in_subprocess(function, args, kwargs, Ntries, max_eval_time=None, max_memory_bytes=None):

	#compute the total timeout from the maximum time per waveform evaluation, if given
	if max_eval_time is not None: timeout = Ntries*max_eval_time
	else:                         timeout = None

	#shared array holding the timings, NaN until the corresponding try succeeds
	ts_shared = mp_ctx.Array('d', [np.nan]*Ntries)

	#worker writing each timing into shared memory as soon as it is measured,
	#so timings of tries completed before a crash are preserved
	def timing_worker():
		try:
			#limit the address space of the child, so that runaway allocations fail
			#inside it (MemoryError) instead of thrashing/OOM-killing the whole machine
			if max_memory_bytes is not None:
				resource.setrlimit(resource.RLIMIT_AS, (int(max_memory_bytes), int(max_memory_bytes)))
			for i in range(Ntries):
				t_start = time.perf_counter()
				function(*args, **kwargs)
				ts_shared[i] = time.perf_counter() - t_start
		except Exception as e:
			print('Warning: waveform call raised %r'%(e,))

	#run the worker in a child process and wait for it (at most timeout seconds, if given)
	p = mp_ctx.Process(target=timing_worker)
	p.start()
	p.join(timeout)

	#if the child is still running after timeout, kill it
	if p.is_alive():
		p.terminate()
		p.join(10)
		if p.is_alive(): p.kill(); p.join()
		print('Warning: waveform call timed out after %ss and was killed'%(timeout,))
	#a negative exitcode means the child was killed by a signal (e.g. SIGSEGV or the OOM killer)
	elif p.exitcode != 0:
		print('Warning: waveform call died with exit code %s'%(p.exitcode,))

	return np.array(ts_shared[:])

#function to get waveform in frequency domain from LAL
def lalsim_fd(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, dL_Mpc, iota, phiref, long_asc_nodes, ecc, mean_ano, delta_f, f_min, f_max, f_ref, delta_t, waveform_dictionary, approx_string, f_min_gen_fact=1):

	#compute the minimum frequency to generate the waveform from
	f_min_gen = f_min_gen_fact*f_min

	#find lalsim approximant
	approximant = lalsim.SimInspiralGetApproximantFromString(approx_string)

	#compute the polarizations as lal frequency series
	hp_LAL, hc_LAL = lalsim.SimInspiralChooseFDWaveform(lal.lal.MSUN_SI*m1, lal.lal.MSUN_SI*m2, s1x, s1y, s1z, s2x, s2y, s2z, dL_Mpc*lal.lal.PC_SI*1.0e6, iota, phiref, long_asc_nodes, ecc, mean_ano, delta_f, f_min_gen, f_max, f_ref, waveform_dictionary, approximant)
	hp_LAL = np.array(hp_LAL.data.data)
	hc_LAL = np.array(hc_LAL.data.data)

#function to get waveform in time domain from LAL
def lalsim_td(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, dL_Mpc, iota, phiref, long_asc_nodes, ecc, mean_ano, delta_f, f_min, f_max, f_ref, delta_t, waveform_dictionary, approx_string, f_min_gen_fact=1):

	#compute the minimum frequency to generate the waveform from
	f_min_gen = f_min_gen_fact*f_min

	#find lalsim approximant
	approximant = lalsim.SimInspiralGetApproximantFromString(approx_string)

	#compute the polarizations as lal time series
	hp_LAL, hc_LAL = lalsim.SimInspiralChooseTDWaveform(lal.lal.MSUN_SI*m1, lal.lal.MSUN_SI*m2, s1x, s1y, s1z, s2x, s2y, s2z, dL_Mpc*lal.lal.PC_SI*1.0e6, iota, phiref, long_asc_nodes, ecc, mean_ano, delta_t, f_min_gen, f_ref, waveform_dictionary, approximant)
	hp_LAL = np.array(hp_LAL.data.data)
	hc_LAL = np.array(hc_LAL.data.data)

#function to get time domain pySEOBNR waveform
def pyseobnr_td(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, dL_Mpc, iota, phiref, long_asc_nodes, ecc, mean_ano, delta_f, f_min, f_max, f_ref, delta_t, waveform_dictionary, approx_string, f_min_gen_fact=1):

	#compute the minimum frequency to generate the waveform from
	f_min_gen = f_min_gen_fact*f_min

	#initialize pySEOBNR parameter dictionary with our default options
	SEOBNR_params_dict = {"EccIC": 1,  # EccIC = 0 for instantaneous initial orbital frequency, and EccIC = 1 for orbit-averaged initial orbital frequency
	                      "lmax_nyquist": 1, # Disable check of ringdown being below nyquist freq.
	}

	#update pySEOBNR parameter dictionary with inputs
	SEOBNR_params_dict.update(waveform_dictionary)
	SEOBNR_params_dict.update({
	    "mass1": m1,
	    "mass2": m2,
	    "spin1x": s1x,
	    "spin1y": s1y,
	    "spin1z": s1z,
	    "spin2x": s2x,
	    "spin2y": s2y,
	    "spin2z": s2z,
	    "deltaT": delta_t,
	    "f22_start": f_min_gen,
	    "f_ref": f_ref,
	    "distance": dL_Mpc,
	    "inclination": iota,
	    "f_max": 0.5/delta_t,
	    "eccentricity": ecc,
	    "rel_anomaly": mean_ano,
	    "approximant": approx_string,
	})

	#generate time-domain polarizations (results are LAL REAL8TimeSeries)
	SEOB_wfm_gen = pyseobnr.generate_waveform.GenerateWaveform(SEOBNR_params_dict)
	hp_SEOB_td, hc_SEOB_td = SEOB_wfm_gen.generate_td_polarizations()
	hp_SEOB_td = np.array(hp_SEOB_td.data.data)
	hc_SEOB_td = np.array(hc_SEOB_td.data.data)

#function to get time domain TEOBResum waveform
def teobresum_td(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, dL_Mpc, iota, phiref, long_asc_nodes, ecc, mean_ano, delta_f, f_min, f_max, f_ref, delta_t, waveform_dictionary, approx_string, f_min_gen_fact=1):
	
	#compute the minimum frequency to generate the waveform from
	f_min_gen = f_min_gen_fact*f_min

	#initialize TEOB parameter dictionary with our default options
	#For details, check teobresums/src/Dali/C/src/TEOBResumSPars.c and teobresums/src/Dali/C/src/TEOBResumS.h
	TEOB_params = {
	'domain'             : 0,         # Time domain. EOBSPA is not available for eccentric waveforms!
	'use_geometric_units': "no",      # output quantities in geometric units. Default = 1
	'ecc_freq'           : 3,         # Use periastron (0), average between r+ and r- (1), apastron (2),  orbit-averaged frequency (3)
	'interp_uniform_grid': "yes",     # interpolate mode by mode on a uniform grid. Default = 0 (no interpolation)
	'use_mode_lm'        : [0, 1, 2, 3, 4, 6, 8],  # List of modes to use/output through EOBRunPy. By default we consider [[2,1],[2,2],[3,1],[3,2],[3,3],[4,2],[4,4]], but this can be overriden by passing waveform_dictionary with use_mode_lm
	}

	#update TEOB parameter dictionary with inputs
	TEOB_params.update(waveform_dictionary)
	TEOB_params.update({
	# System parametes
	'M'                  : float(m1+m2),
	'q'                  : float(m1/m2), # Mass ratio m1/m2 > 1
	'chi1x'              : float(s1x),   # x component of chi1
	'chi1y'              : float(s1y),   # y component of chi1
	'chi1z'              : float(s1z),   # z component of chi1
	'chi2x'              : float(s2x),   # x component of chi2
	'chi2y'              : float(s2y),   # y component of chi2
	'chi2z'              : float(s2z),   # z component of chi2
	'chi1'               : float(s1z),   # z component of chi1
	'chi2'               : float(s2z),   # z component of chi2
	'ecc'                : float(ecc),           # Eccentricity. Default = 0.
	'anomaly'            : float(mean_ano),      # True anomaly. Default = Pi
	'distance'           : float(dL_Mpc),
	'inclination'        : float(iota),
	'coalescence_angle'  : float(phiref),
	# Initial conditions and output time grid
	'initial_frequency'  : float(f_min_gen), # in Hz if use_geometric_units = 0, else in geometric units
	'srate_interp'       : float(1./delta_t),
	})
	
	#generate time-domain polarizations
	times, hp_TEOB_td, hc_TEOB_td = EOBRun_module.EOBRunPy(TEOB_params)
	hp_TEOB_td = np.asarray(hp_TEOB_td)
	hc_TEOB_td = np.asarray(hc_TEOB_td)

#function to get time domain EFPE waveform
def EFPE_td(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, dL_Mpc, iota, phiref, long_asc_nodes, ecc, mean_ano, delta_f, f_min, f_max, f_ref, delta_t, waveform_dictionary, approx_string, f_min_gen_fact=1):
	
	#compute the minimum frequency to generate the waveform from
	f_min_gen = f_min_gen_fact*f_min

	#compute the pyEFPE dictionary
	p_pyEFPE = waveform_dictionary.copy()
	p_pyEFPE.update({
		"mass1": m1,
		"mass2": m2,
		"spin1x": s1x,
		"spin1y": s1y,
		"spin1z": s1z,
		"spin2x": s2x,
		"spin2y": s2y,
		"spin2z": s2z,
		"eccentricity": ecc,
		"distance": dL_Mpc,
		"inclination": iota,
		"mean_anomaly": mean_ano,
		"f22_start": f_min_gen,
	})

	#choose approximant, translating to the legacy *_start keys for the old pyEFPE package
	if approx_string=="pyEFPE":
		EFPE = pyEFPE.pyEFPE
		p_pyEFPE = to_legacy_pyEFPE_params(p_pyEFPE)
	elif approx_string=="pyEFPEHM":
		EFPE = pyEFPEHM.pyEFPE
	
	#compute EFPE polarization
	EFPE(p_pyEFPE).generate_tdomain_waveform(delta_t=delta_t)

#function to get frequency domain EFPE waveform
def EFPE_fd(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, dL_Mpc, iota, phiref, long_asc_nodes, ecc, mean_ano, delta_f, f_min, f_max, f_ref, delta_t, waveform_dictionary, approx_string, f_min_gen_fact=1):

	#compute the minimum frequency to generate the waveform from
	f_min_gen = f_min_gen_fact*f_min

	#compute the pyEFPE dictionary
	p_pyEFPE = waveform_dictionary.copy()
	p_pyEFPE.update({
		"mass1": m1,
		"mass2": m2,
		"spin1x": s1x,
		"spin1y": s1y,
		"spin1z": s1z,
		"spin2x": s2x,
		"spin2y": s2y,
		"spin2z": s2z,
		"eccentricity": ecc,
		"distance": dL_Mpc,
		"inclination": iota,
		"mean_anomaly": mean_ano,
		"f22_start": f_min_gen,
	})

	#choose approximant, translating to the legacy *_start keys for the old pyEFPE package
	if approx_string=="pyEFPE":
		EFPE = pyEFPE.pyEFPE
		p_pyEFPE = to_legacy_pyEFPE_params(p_pyEFPE)
	elif approx_string=="pyEFPEHM":
		EFPE = pyEFPEHM.pyEFPE
	
	#frequency array
	freqs = np.arange(f_min, f_max, delta_f)
	
	#compute EFPE polarization
	EFPE(p_pyEFPE).generate_waveform(freqs)

#function to compute eccentric inspiral duration duration at 0PN
def compute_duration_of_Mc_f0_e0(Mc, f0, e0):
	return (5./256.)*((t_sun_s*Mc)**(-5./3.))*((np.pi*f0)**(-8./3.))*((1-e0*e0)**3.5)*pyEFPEHM.functions.F_tLO_series(e0*e0)


########################## General info on waveform comparissons ##################################

#Initialize base waveform dictionaries
EFPE_dict = {'Amplitude_tol': 1e-4, 'pn_amplitude_order': 2}

#LAL dictionaries
LAL_dict = lal.CreateDict()
if 'pn_spin_order' in EFPE_dict:      lalsim.SimInspiralWaveformParamsInsertPNSpinOrder(LAL_dict, int(EFPE_dict['pn_spin_order']))
if 'pn_phase_order' in EFPE_dict:     lalsim.SimInspiralWaveformParamsInsertPNPhaseOrder(LAL_dict, int(EFPE_dict['pn_phase_order']))
if 'pn_amplitude_order' in EFPE_dict: lalsim.SimInspiralWaveformParamsInsertPNAmplitudeOrder(LAL_dict, int(EFPE_dict['pn_amplitude_order']))
if 'mode_array' in EFPE_dict:
	mode_array_lal = lalsim.SimInspiralCreateModeArray()
	for mode in EFPE_dict['mode_array']:
		lalsim.SimInspiralModeArrayActivateMode(mode_array_lal, mode[0], mode[1])
	lalsim.SimInspiralWaveformParamsInsertModeArray(LAL_dict, mode_array_lal)

#SEOBNR dictionaries
SEOB_dict = {}

#TEOB dictionaries
TEOB_dict = {}

########################## Comparissons to consider ##################################

# Comparissons
comparisons = {'td': {}, 'fd': {}}

# Comparissons of time-domain models
comparisons['td']['pyEFPEHM'] = dict(
waveform_dictionary = EFPE_dict,
function            = EFPE_td,
eccentric           = True,
precessing          = True,
)
comparisons['td']['pyEFPE'] = dict(
waveform_dictionary = EFPE_dict,
function            = EFPE_td,
eccentric           = True,
precessing          = True,
)
comparisons['td']['SEOBNRv5PHM'] = dict(
waveform_dictionary = SEOB_dict,
function            = pyseobnr_td,
eccentric           = False,
precessing          = True,
)
comparisons['td']['SEOBNRv5EHM'] = dict(
waveform_dictionary = SEOB_dict,
function            = pyseobnr_td,
eccentric           = True,
precessing          = False,
)
comparisons['td']['TEOBResumS-Dali'] = dict(
waveform_dictionary = TEOB_dict,
function            = teobresum_td,
eccentric           = True,
precessing          = True,
)
comparisons['td']['SpinTaylorT4'] = dict(
waveform_dictionary = LAL_dict,
function            = lalsim_td,
eccentric           = False,
precessing          = True,
)
comparisons['td']['IMRPhenomTPHM'] = dict(
waveform_dictionary = LAL_dict,
function            = lalsim_td,
eccentric           = False,
precessing          = True,
)

# Comparissons of frequency-domain models
comparisons['fd']['pyEFPEHM'] = dict(
waveform_dictionary = EFPE_dict,
function            = EFPE_fd,
eccentric           = True,
precessing          = True,
)
comparisons['fd']['pyEFPE'] = dict(
waveform_dictionary = EFPE_dict,
function            = EFPE_fd,
eccentric           = True,
precessing          = True,
)
comparisons['fd']['IMRPhenomXPNR'] = dict(
waveform_dictionary = LAL_dict,
function            = lalsim_fd,
eccentric           = False,
precessing          = True,
)

####################################### Systems to consider #######################################

#Constant binary parameters to consider
m1 = 20.0
m2 = 10.0
s1x = -0.44
s1y = -0.26
s1z = 0.48
s2x = -0.31
s2y = 0.01
s2z = -0.84
mean_ano = 0.
dL_Mpc = 1000.0
iota = np.pi/3
phiref = 0.
long_asc_nodes=0.

#Range of minimum frequencies to consider
f_mins = np.geomspace(0.5, 20., 20)

#Time array specifications
f_max   = 256.
delta_t = 0.5/f_max
f_min_gen_fact = 1

#number of repeats for timings
Ntries_timings = 5

#maximum seconds allowed per waveform evaluation before killing it (None = no limit)
max_eval_time_timings = None

#maximum memory each waveform is allowed to allocate, slightly below the physical RAM (None = no limit)
max_memory_timings = 0.9*total_memory_bytes

#eccentricities to consider
eccs = np.linspace(0, 0.6, 3)

#output directory
outdir = './outdir/waveform_comparisons/'
plotdir = outdir

################################ Do timings ##########################################

string_id = '_wf_timings_N_%s'%(Ntries_timings)
result_filename = outdir+'/result'+string_id+'.pickle'

#put all input params in a dictionary
input_params = {
    'm1': m1, 'm2': m2,
    's1x': s1x, 's1y': s1y, 's1z': s1z,
    's2x': s2x, 's2y': s2y, 's2z': s2z,
    'mean_ano': mean_ano, 'dL_Mpc': dL_Mpc,
    'iota': iota, 'phiref': phiref, 'long_asc_nodes': long_asc_nodes,
    'f_max': f_max, 'delta_t': delta_t,
    'f_min_gen_fact': f_min_gen_fact, 'Ntries_timings': Ntries_timings
}
try:

	with open(result_filename, 'rb') as handle: result = pickle.load(handle)
	
	#make sure it is the same
	assert np.all(result['eccs']==eccs)
	assert np.all(result['f_mins']==f_mins)
	assert input_params == result['input_params']

except:
	#compute chirp mass
	Mc = ((m1*m2)**0.6)*((m1+m2)**-0.2)

	wf_timings = {'td': {}, 'fd': {}, 'seglens': {}, 'durations': {}}
	for e0 in eccs:

		print('\n'+30*'-'+'e0=%s'%(e0)+30*'-')	

		#loop over minimum frequencies
		wf_timings['td'][e0] = {}
		wf_timings['fd'][e0] = {}
		wf_timings['seglens'][e0] = []
		wf_timings['durations'][e0] = []
		for f_min in f_mins:

			
			#determine the estimated duration for this event
			duration = compute_duration_of_Mc_f0_e0(Mc, f_min, e0)
			
			#compute the frequency resolution to use
			seglen   = 2**(np.ceil(np.log2(duration)))
			delta_f  = 1./seglen

			#save duration and timings
			wf_timings['seglens'][e0].append(seglen)
			wf_timings['durations'][e0].append(duration)

			print('\nf_min=%.3gHz  (seglen=%ss)'%(f_min, seglen))
			
			for domain in ['td', 'fd']:
			
				#loop over models for time domain comparisons
				for approx_string, comparison in comparisons[domain].items():

					#handle if the waveform supports precession or not
					if comparison['precessing']: s1x0, s1y0, s2x0, s2y0 = s1x, s1y, s2x, s2y
					else:                        s1x0, s1y0, s2x0, s2y0 = 0., 0., 0., 0.
					
					#handle if the waveform supports eccentricity or not
					if comparison['eccentric']: ell0 = mean_ano
					elif e0==0:                 ell0 = 0.
					else: continue
					
					#time this function Ntries_timings times in a child process, so that
					#segfaults/OOM kills only lose the child and leave the timings as NaN
					ts = time_function_in_subprocess(comparison['function'],
					                                 (m1, m2, s1x0, s1y0, s1z, s2x0, s2y0, s2z, dL_Mpc, iota, phiref, long_asc_nodes, e0, ell0, delta_f, f_min, f_max, f_min, delta_t, comparison['waveform_dictionary'], approx_string),
					                                 {'f_min_gen_fact': f_min_gen_fact},
					                                 Ntries_timings, max_eval_time=max_eval_time_timings, max_memory_bytes=max_memory_timings)

					#save the array of times
					if approx_string in wf_timings[domain][e0]:
						wf_timings[domain][e0][approx_string].append(ts)
					else:
						wf_timings[domain][e0][approx_string] = [ts]
					
					#print times
					print('For %s comparisson of %s -> tmin=%.3gs, tmed=%.3gs, tmax=%.3gs'%(domain, approx_string, np.amin(ts), np.median(ts), np.amax(ts)))

	#put timings into more convenient numpy arrays of shape (len(f_mins), Ntries_timings)
	for domain in ['td', 'fd']:
		for e0 in eccs:
			for wf in wf_timings[domain][e0].keys():
				wf_timings[domain][e0][wf] = np.vstack(wf_timings[domain][e0][wf])
	
	#put things in result dictionary
	result = {'eccs': eccs, 'f_mins': f_mins, 'input_params': input_params, 'wf_timings': wf_timings}
	
	#save the result file
	if not os.path.exists(outdir):  os.makedirs(outdir)
	with open(result_filename, 'wb') as handle: pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

################################ Plotting ##########################################

#plotting inputs
line_width = 3
ncols_legend = 2

#Color for each waveform
wf_colors = {
    'pyEFPEHM':        '#000000',  # black
    'pyEFPE':          '#0072B2',  # blue
    'TEOBResumS-Dali': '#009E73',  # green
    'SEOBNRv5EHM':     '#D55E00',  # vermillion
    'SEOBNRv5PHM':     '#CC79A7',  # pink/purple
    'SpinTaylorT4':    '#56B4E9',  # sky blue
    'IMRPhenomXPNR':   '#F0E442',  # yellow
    'IMRPhenomTPHM':   '#E69F00',  # orange
}


if not os.path.exists(plotdir): os.makedirs(plotdir)

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

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


# Do timings figure
fig, axs = plt.subplots(nrows=len(eccs), ncols=2, sharex=True, sharey='row', figsize=(9,14), constrained_layout=True, squeeze=False)
for idom, domain in enumerate(['td', 'fd']):
	for ie0, e0 in enumerate(eccs):
		for wf in result['wf_timings'][domain][e0].keys():
			axs[ie0,idom].plot(result['f_mins'], np.amin(result['wf_timings'][domain][e0][wf], axis=1), color=wf_colors[wf], lw=line_width)
		
		axs[ie0,idom].text(0.97, 0.97, rf'$e_0 = %.2g$'%(e0), transform=axs[ie0,idom].transAxes,ha='right', va='top')
		axs[ie0,idom].set_yscale('log')
		axs[ie0,idom].set_xscale('log')
		axs[ie0,idom].set_xlim(np.amin(result['f_mins']), np.amax(result['f_mins']))
		#axs[ie0,idom].margins(0)

axs[0,0].set_title('Time Domain')
axs[0,1].set_title('Frequency Domain')
for ie0, e0 in enumerate(eccs): axs[ie0,0].set_ylabel('Runtime [s]')
for idom in range(2):         axs[-1,idom].set_xlabel(r'$f_0^{\mathrm{GW},22} \; [\mathrm{Hz}]$')

#build legend handles
legend_handles = [Line2D([0], [0], color=color, lw=line_width, label=r'\texttt{%s}'%(wf)) for wf, color in wf_colors.items()]

#put legend
fig.legend(handles=legend_handles, loc='outside upper center', ncols=ncols_legend, frameon=False)

plt.savefig(plotdir+'/plot'+string_id+'.pdf')
plt.show()
        
        
        
