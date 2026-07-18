import os

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

from scipy.signal.windows import tukey
from scipy.fft import fft, rfft, irfft
from scipy.optimize import minimize_scalar, minimize, differential_evolution
import numpy as np
import pickle
import multiprocessing as mp

	
#universal constants
t_sun_s = 4.92549094831e-6  #GMsun/c**3 [s]

#dictionary of variable names as pyEFPE keys
pyEFPE_keys = {'mass1': 'm1', 'mass2': 'm2', 'eccentricity': 'ecc', 'mean_anomaly':'mean_anomaly', 'spin1x': 's1x', 'spin1y': 's1y', 'spin1z': 's1z', 'spin2x': 's2x', 'spin2y': 's2y', 'spin2z': 's2z', 'inclination': 'iota', 'phase': 'phiref'}

#dictionary mapping the new parameter keys to the legacy *_start keys understood by the old pyEFPE package
legacy_pyEFPE_keys = {'eccentricity': 'e_start', 'phase': 'phi_start', 'mean_anomaly': 'mean_anomaly_start'}

#function to translate a pyEFPEHM parameter dictionary to the legacy keys of the old pyEFPE package
#the old package silently ignores unknown keys, so the new keys have to be renamed for it
def to_legacy_pyEFPE_params(p_pyEFPE):
	#the old package has no notion of a reference frequency, so a set f22_ref cannot be translated
	#fail loudly instead of silently comparing waveforms anchored at different frequencies
	if p_pyEFPE.get('f22_ref') is not None:
		raise ValueError("f22_ref=%s cannot be translated to the legacy pyEFPE package, which only supports parameters defined at f22_start"%(p_pyEFPE['f22_ref']))
	p_legacy = dict(p_pyEFPE)
	p_legacy.pop('f22_ref', None)
	for new_key, old_key in legacy_pyEFPE_keys.items():
		if new_key in p_legacy: p_legacy[old_key] = p_legacy.pop(new_key)
	return p_legacy

#dictionary with CBC variables for evaluation
param_keys = ['m1', 'm2', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z', 'iota', 'phiref', 'pol', 'f_max', 'ecc', 'mean_anomaly']

#names of parameters to minimize over for stringID
minimize_parameter_names = {'phase':'p0', 'phase_s': 'pS', 'phase_s1': 'pS1', 'phase_s2': 'pS2', 'eccentricity': 'e0', 'mean_anomaly': 'l0',}

#function to compute the time to ISCO given an initial frequency and component masses (in solar masses)
def t_to_ISCO_0PN(f0, m1, m2):
	
	#compute the symmetric mass ratio and the total mass in solar masses
	M = m1+m2
	nu = m1*m2/(M*M)
	
	#convert total mass to seconds
	M = t_sun_s*M
	
	#compute pi*M*f_ISCO
	piMff = 6**-1.5
	#compute pi*M*f0
	piMf0 = np.pi*M*f0

	#return the time to ISCO from Eq.(4.19) of Maggiore
	return (5/256)*(M/nu)*(piMf0**(-8/3) - piMff**(-8/3))

#function to compute the XPHM Minimum Energy Circular Orbit (MECO) frequency
def compute_fMECO(M, nu, spin_1z, spin_2z):
	
	#compute the Mf at the MECO with the lalsimulation function
	MfMECO = np.array([lalsim.SimIMRPhenomXfMECO(nu[i], spin_1z[i], spin_2z[i]) for i in range(len(nu))])
	
	#use the total mass to give the right units and return
	return MfMECO/(t_sun_s*M)

#function to get waveform in frequency domain from frequency domain LAL waveform
def fd_from_lalsim_fd(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, dL_Mpc, iota, phiref, long_asc_nodes, ecc, mean_ano, delta_f, f_min, f_max, f_ref, waveform_dictionary, approximant, **kwargs):

		#compute the polarizations as lal frequency series
		hp_LAL, hc_LAL = lalsim.SimInspiralChooseFDWaveform(lal.lal.MSUN_SI*m1, lal.lal.MSUN_SI*m2, s1x, s1y, s1z, s2x, s2y, s2z, dL_Mpc*lal.lal.PC_SI*1.0e6, iota, phiref, long_asc_nodes, ecc, mean_ano, delta_f, f_min, f_max, f_ref, waveform_dictionary, approximant)
		
		#return the frequencies we want as numpy arrays
		return np.array(hp_LAL.data.data)[int(f_min/delta_f):int(f_max/delta_f)], np.array(hc_LAL.data.data)[int(f_min/delta_f):int(f_max/delta_f)]

#function to convert time domain signals to the frequency domain
def convert_td_to_fd(h_td, delta_t, delta_f, f_min, f_max, max_alpha=0.05, apply_tukey=True, max_rolloff_lead_time=0, df_factor=2):

	#compute the size of the array needed for the specified delta_f and delta_t
	df_factor = max(1, int(df_factor))
	N_tsamples = int(np.ceil(df_factor/(delta_f*delta_t)))

	#compute indexes of target frequencies
	i_start = round(df_factor*f_min/delta_f)
	i_end = round(df_factor*f_max/delta_f)

	#if we generated too much stuff at the beginning, remove it
	if len(h_td)>N_tsamples:
		print("Warning: len(h_td)=%s > N_tsamples=%s, removing %s points at the beginning of the signal"%(len(h_td), N_tsamples, len(h_td) - N_tsamples))
		h_td = h_td[-N_tsamples:]

	#if required, apply Tukey window to reduce spectral leakage
	if apply_tukey:

		#compute the roll off from the f_min requirement
		roll_off = 2/f_min
		#We want the start of the tukey roll off at t = tf - max_rolloff_lead_time
		n_extra = max(0, int(np.ceil((roll_off - max_rolloff_lead_time)/delta_t)))
		#compute the value of alpha to use in Tukey window
		alpha = min(max_alpha, 2*roll_off/(delta_t*(len(h_td) + n_extra)))
		#apply the tukey window, removing the samples after the length of h_td
		h_td = h_td*tukey(len(h_td)+ n_extra, alpha=alpha)[:len(h_td)]

	#if required, prepend with zeros to get the desired frequency resolution
	if len(h_td)<N_tsamples:
		h_td = np.concatenate((np.zeros(N_tsamples - len(h_td)), h_td))

	#compute fast fourier transform at the required frequencies
	return delta_t*rfft(h_td)[i_start:i_end:df_factor]


#function to get waveform in frequency domain from time domain LAL waveform
def fd_from_lalsim_td(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, dL_Mpc, iota, phiref, long_asc_nodes, ecc, mean_ano, delta_f, f_min, f_max, f_ref, waveform_dictionary, approximant, f_min_gen_fact=0.8, dt_factor=4, delta_t_max=1/4096, max_alpha=0.05, apply_tukey=True, df_factor=2):

	#compute the required delta_t
	delta_t = min(delta_t_max, 1/(2*f_max*max(1, dt_factor)))
	#make this a power of two
	delta_t = 2**np.floor(np.log2(delta_t))
	
	#compute the minimum frequency to generate the waveform from
	f_min_gen = f_min_gen_fact*f_min
	
	#compute lal waveform
	hp_LAL, hc_LAL = lalsim.SimInspiralChooseTDWaveform(lal.lal.MSUN_SI*m1, lal.lal.MSUN_SI*m2, s1x, s1y, s1z, s2x, s2y, s2z, dL_Mpc*lal.lal.PC_SI*1.0e6, iota, phiref, long_asc_nodes, ecc, mean_ano, delta_t, f_min_gen, f_ref, waveform_dictionary, approximant)

	#loop over polarizations
	hs_fd = []
	for h_LAL in [hp_LAL, hc_LAL]:

		#convert time domain polarizations to the frequency domain
		hs_fd.append(convert_td_to_fd(np.array(h_LAL.data.data), delta_t, delta_f, f_min, f_max, max_alpha=max_alpha, apply_tukey=apply_tukey, max_rolloff_lead_time=t_to_ISCO_0PN(f_max, m1, m2), df_factor=df_factor))

	return hs_fd

#function to get waveform in the frequency domain from time domain pySEOBNR waveform
def fd_from_pyseobnr_td(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, dL_Mpc, iota, phiref, long_asc_nodes, ecc, mean_ano, delta_f, f_min, f_max, f_ref, waveform_dictionary, approximant, f_min_gen_fact=0.8, dt_factor=4, delta_t_max=1/4096, max_alpha=0.05, apply_tukey=True, df_factor=2):

	#compute the required delta_t
	delta_t = min(delta_t_max, 1/(2*f_max*max(1, dt_factor)))
	#make this a power of two
	delta_t = 2**np.floor(np.log2(delta_t))

	#compute the minimum frequency to generate the waveform from
	f_min_gen = f_min_gen_fact*f_min

	#if f_min_gen and f_ref are very similar, have them be exactly equal
	if (abs(f_min_gen - f_ref) < 1e-12*abs(f_min_gen + f_ref)):
		f_min_gen = f_ref

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
	    "approximant": approximant,
	})

	#generate time-domain polarizations (results are LAL REAL8TimeSeries)
	SEOB_wfm_gen = pyseobnr.generate_waveform.GenerateWaveform(SEOBNR_params_dict)
	hp_SEOB_td, hc_SEOB_td = SEOB_wfm_gen.generate_td_polarizations()

	#loop over polarizations
	hs_fd = []
	for h_SEOB in [hp_SEOB_td, hc_SEOB_td]:

		#convert time domain polarizations to the frequency domain
		hs_fd.append(convert_td_to_fd(np.array(h_SEOB.data.data), delta_t, delta_f, f_min, f_max, max_alpha=max_alpha, apply_tukey=apply_tukey, max_rolloff_lead_time=0, df_factor=df_factor))

	return hs_fd

#function to convert (l, m) modes to TEOBResum linear index k
def modes_to_teob_k(modes):
    return sorted([int(l*(l-1)/2 + m - 2) for (l, m) in modes])

#function to get waveform in the frequency domain from time domain TEOBResumS waveform
def fd_from_teobresum_td(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, dL_Mpc, iota, phiref, long_asc_nodes, ecc, mean_ano, delta_f, f_min, f_max, f_ref, waveform_dictionary, approximant, f_min_gen_fact=0.8, dt_factor=4, delta_t_max=1/4096, max_alpha=0.05, apply_tukey=True, df_factor=2):
	
	#compute the required delta_t
	delta_t = min(delta_t_max, 1/(2*f_max*max(1, dt_factor)))
	#make this a power of two
	delta_t = 2**np.floor(np.log2(delta_t))

	#compute the minimum frequency to generate the waveform from
	f_min_gen = f_min_gen_fact*f_min

	#Since in TEOB we can not choose f_ref, we have to check that f_min_gen==f_ref
	if (abs(f_min_gen - f_ref) > 1e-9*abs(f_min_gen + f_ref)):
		print("Warning: f_ref=%sHz is not equal to f_min_gen=%sHz. TEOB will actually use f_min_gen as its reference frequency."%(f_ref, f_min_gen))

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

	#loop over polarizations
	hs_fd = []
	for h_TEOB in [hp_TEOB_td, hc_TEOB_td]:

		#convert time domain polarizations to the frequency domain
		hs_fd.append(convert_td_to_fd(np.asarray(h_TEOB), delta_t, delta_f, f_min, f_max, max_alpha=max_alpha, apply_tukey=apply_tukey, max_rolloff_lead_time=0, df_factor=df_factor))

	return hs_fd


#function to obtain PSD at certain frequencies from lal
def compute_asd(delta_f, f_min, f_max, psd_name='aLIGOO3LowT1800545', asd_folder='./ASDs'):

	#frequencies at which to return the ASD
	freqs = np.arange(f_min, f_max, delta_f)

	#first try to load asd from a file
	try:
		asd_raw = np.loadtxt(asd_folder+'/'+psd_name+'.txt')

		#interpolate the tabulated ASD onto freqs
		asd = np.interp(freqs, asd_raw[:,0], asd_raw[:,1])

	#otherwise, try computing it using lalsimulation
	except:
		#create lalseries, add one frequency point because in the last freq, psd gives 0
		lalseries = lal.CreateREAL8FrequencySeries('', lal.LIGOTimeGPS(0), 0, delta_f, lal.DimensionlessUnit, int(f_max/delta_f)+1)

		#guess the PSD function from name
		func = lalsim.__dict__['SimNoisePSD' + psd_name]

		#put the PSD on lalseries
		func(lalseries, f_min)

		#interpolate the ASD (sqrt of the PSD) onto freqs
		lal_freqs = delta_f*np.arange(len(lalseries.data.data))
		asd = np.interp(freqs, lal_freqs, np.sqrt(np.array(lalseries.data.data)))

	#return the ASD as the sqrt of the PSD
	return asd

#function to convert spherical coordinates into cartesian coordinates (inspired in the pyroq function)
def spherical_to_cartesian(sph):
	x = sph[0] * np.sin(sph[1]) * np.cos(sph[2])
	y = sph[0] * np.sin(sph[1]) * np.sin(sph[2])
	z = sph[0] * np.cos(sph[1])
	return x, y, z

#function to compute polarization response
def pol_response(hp, hc, pol):
	return np.cos(2*pol)*hp + np.sin(2*pol)*hc

#function to compute random parameters
def generate_rand_params(Nsamples, mc_low=1.4, mc_high=2.6, q_low=0.25, q_high=1, s1_low=None, s1_high=1, s2_low=None, s2_high=1, ecc_low=0, ecc_high=0, precessing=True):
	
	#generate random input parameters
	params_low  = [ mc_low,  q_low,       0,       0,  ecc_low,-1,       0,       0,     0]
	params_high = [mc_high, q_high, s1_high, s2_high, ecc_high, 1, 2*np.pi, 2*np.pi, np.pi]
	mc, q, s1, s2, ecc, cos_iota, phiref, mean_anomaly, pol = np.transpose(np.random.uniform(params_low, params_high, size=(Nsamples,len(params_low))))

	#if q is larger than 1, flip it to make it smaller
	q = np.where(q>1, 1/q, q)
	
	#compute m1, m2 from mc, q
	m1 = mc*(q**-0.6)*((1 + q)**0.2)
	m2 = m1*q
	
	#compute inclination
	iota = np.arccos(cos_iota)

	#generate spin tilt angles
	ths1, ths2 = np.arccos(np.random.uniform(-1, 1, size=(2,Nsamples)))
	#generate spin phases
	phs1, phs2 = np.random.uniform(0, 2*np.pi, size=(2,Nsamples))
	#compute component spins
	s1x, s1y, s1z = spherical_to_cartesian([s1, ths1, phs1])
	s2x, s2y, s2z = spherical_to_cartesian([s2, ths2, phs2])

	#in the non-precessing case, set perpendicular spins to 0
	if not precessing:
		s1x, s1y, s2x, s2y = np.zeros_like(s1), np.zeros_like(s1), np.zeros_like(s2), np.zeros_like(s2)

	#in the non-eccentric case, set mean anomaly to 0
	if ecc_high==0: mean_anomaly *= 0

	#compute derived mass parameters
	M = m1 + m2
	nu = m1*m2/(M*M)

	#compute ISCO frequency
	f_ISCO =  1/((6**1.5)*np.pi*t_sun_s*M)
	
	#compute MECO frequency
	f_MECO = compute_fMECO(M, nu, s1z, s2z)

	#force the MECO to be below the ISCO
	f_MECO = np.minimum(f_ISCO, f_MECO)

	#compute the effective inspiral spin parameter
	chi_eff = (s1z + q*s2z)/(1 + q)
	
	#compute the precessing spin parameter
	s1p = np.sqrt(s1x*s1x + s1y*s1y)
	s2p = np.sqrt(s2x*s2x + s2y*s2y)
	chi_p = np.maximum(s1p, (q*(4*q + 3)/(4 + 3*q))*s2p)

	#make a dictionary with all the samples
	return {'m1': m1, 'm2': m2, 's1x': s1x, 's1y': s1y, 's1z': s1z, 's2x': s2x, 's2y': s2y, 's2z': s2z, 'iota': iota, 'phiref': phiref, 'pol': pol, 'f_ISCO': f_ISCO, 'f_MECO': f_MECO, 'mc': mc, 'q': q, 'chi_eff': chi_eff, 'chi_p': chi_p, 'ecc': ecc, 'mean_anomaly': mean_anomaly}

#function to normalize waveforms and weigh them by asd=sqrt(psd)
def normalize_h(h, asd=None):

	#normalize by psd
	if asd is not None: h = h/asd

	#compute the norm of h
	h_norm = np.linalg.norm(h)
	
	return h/h_norm, h_norm

#function to compute the fft of an overlap using padding
def fft_overlap(overlap, padding_fact=4):
	
	#compute the new len(overlap) to be a power of 2
	new_len = 2**int(np.ceil(np.log2(padding_fact*len(overlap))))
	
	#return the fft of the padded overlap
	return fft(np.append(overlap, np.zeros(new_len - len(overlap))))

#function to compute the mismatch without any minimizations
def mismatch_amp_minimized(h1, h2, asd=None):

	#normalize the waveforms
	h1n, h1_norm = normalize_h(h1, asd=asd)
	h2n, h2_norm = normalize_h(h2, asd=asd)
	
	#return the mismatch
	return 1 - np.real(np.sum(np.conj(h1n)*h2n))

#function to minimize mismatch from initial guess from fft mismatch
def minimize_mm_dt(mm_fft, mm_func_dt, freqs, t_tol=1e-9):

	#find the index of the time that alignes them
	i_min = np.argmin(mm_fft)
	min_mm = mm_fft[i_min]
	
	#compute the time shift from the FFT, finding smallest difference
	if i_min > 0.5*len(mm_fft): di = i_min - len(mm_fft)
	else:                       di = i_min
	
	#time resotion of fourier transform
	df = freqs[1] - freqs[0]
	dT = 1/(df*len(mm_fft))

	#initial time shift guess from fft
	dt_shift = di*dT

	#try to minimize this function
	try:
		#use a bracketed search algorithm
		minimize_result = minimize_scalar(mm_func_dt, bracket=(dt_shift-dT, dt_shift, dt_shift+dT), tol=t_tol)
		if not minimize_result.success: print(minimize_result.message)
		#only replace min_mm if the result of the minimization is truly smaller
		if minimize_result.fun<min_mm:
			min_mm = minimize_result.fun
			dt_shift = minimize_result.x
	except Exception as excep:
		print('minimize_scalar failed with exception:', excep)
		print('Using fft minimization of mismatch: min_mm: %.3g'%(min_mm))

	return min_mm, dt_shift

#function to compute the mismatch minimized over time, phase and amplitude
def mismatch_t_ph_amp_minimized(h1, h2, freqs, t_tol=1e-9, asd=None, fft_padding_fact=4):
	
	#normalize the waveforms
	h1n, h1_norm = normalize_h(h1, asd=asd)
	h2n, h2_norm = normalize_h(h2, asd=asd)
	
	#compute the overlap
	overlap = np.conj(h1n)*h2n

	#compute their mismatch at all possible times
	mm_fft = 1 - np.abs(fft_overlap(overlap, padding_fact=fft_padding_fact))

	#function to compute the mismatch at any time
	mm_func_dt = lambda dt: 1 - np.abs(np.sum(np.exp(-2j*np.pi*freqs*dt)*overlap))
	
	#now minimize the mismatch
	min_mm, dt_shift = minimize_mm_dt(mm_fft, mm_func_dt, freqs, t_tol=t_tol)
	
	#compute the relative phases of the waveforms
	ang_h2mh1 = np.angle(np.sum(np.exp(-2j*np.pi*freqs*dt_shift)*overlap))
	
	#return maximized mismatch, time shift, and relative phase and amplitude
	return min_mm, dt_shift, ang_h2mh1, h2_norm/h1_norm

#function to compute compute mismatch in terms of the rhos from Eq.(27) of 1603.02444
def pol_minimized_mismatch_rhos(rho_p, rho_c, Ipc, return_optimum_u=False):

	#compute gamma from Eq.(22)
	gamma = np.real(rho_p*np.conj(rho_c))

	#compute coefficients of quadratic Eq.(26)
	rho_p2 = np.abs(rho_p)**2
	rho_c2 = np.abs(rho_c)**2
	A = Ipc*rho_p2 - gamma
	B = rho_p2-rho_c2
	C = gamma - Ipc*rho_c2
	
	#use it to compute it Eq.(27)
	sqrt_part = np.sqrt(B*B - 4*A*C)
	num = rho_p2 - 2*Ipc*gamma + rho_c2 + sqrt_part
	den = 1 - Ipc**2
	
	#use that the mismatch is mm = 1 - sqrt(2*lambda)
	mm = 1 - np.sqrt(num/(2*den))
	
	#if required, return optimum polarization solving Eq.(26)
	if return_optimum_u:
		#use that kappa=arctan(1/u) and that u_{-} is the one that gives Eq.(27)
		unum = -sqrt_part - B
		uden = 2*A
		#return mismatch and numerator/denominator of u (to avoid divisions)
		return mm, unum, uden
	
	#return only the mismatch
	else:
		return mm

#function to compute the mismatch minimized over time, phase, amplitude and polarization (following 1603.02444)
def mismatch_t_ph_amp_pol_minimized(signal, hp, hc, freqs, t_tol=1e-9, asd=None, fft_padding_fact=4, return_optimum_params=False):
	
	#normalize the waveforms
	sn, s_norm   = normalize_h(signal, asd=asd)
	hpn, hp_norm = normalize_h(hp, asd=asd)
	hcn, hc_norm = normalize_h(hc, asd=asd)
	
	#compute the overlaps
	overlap_sp = np.conj(sn)*hpn
	overlap_sc = np.conj(sn)*hcn

	#compute I_{+x} from Eqs.(23), since it does not depend on time
	Ipc = np.real(np.sum(np.conj(hpn)*hcn))
	
	#compute \rho_+ and \rho_x as functions of time using fft
	rho_p = fft_overlap(overlap_sp, padding_fact=fft_padding_fact)
	rho_c = fft_overlap(overlap_sc, padding_fact=fft_padding_fact)

	#compute the mismatch as a function of time with these rhos
	mm_fft = pol_minimized_mismatch_rhos(rho_p, rho_c, Ipc)

	#function to compute the mismatch as a function of time
	rho_p_func_dt = lambda dt: np.sum(np.exp(-2j*np.pi*freqs*dt)*overlap_sp)
	rho_c_func_dt = lambda dt: np.sum(np.exp(-2j*np.pi*freqs*dt)*overlap_sc)
	mm_func_dt = lambda dt: pol_minimized_mismatch_rhos(rho_p_func_dt(dt), rho_c_func_dt(dt), Ipc)

	#now minimize the mismatch
	min_mm, dt_shift = minimize_mm_dt(mm_fft, mm_func_dt, freqs, t_tol=t_tol)
	
	#if required, return parameters that have been analytically minimized
	if return_optimum_params:
	
		#compute rho_p and rho_c for optimum time shift
		rho_p_opt, rho_c_opt = rho_p_func_dt(dt_shift), rho_c_func_dt(dt_shift)
		
		#compute optimum polarization
		mm, unum, uden = pol_minimized_mismatch_rhos(rho_p_opt, rho_c_opt, Ipc, return_optimum_u=True)
		
		#compute polarization from Eq.(24), where kappa = 2*pol = arctan((hp_norm/hc_norm)/u)
		pol = 0.5*np.arctan2(uden*hp_norm, unum*hc_norm)
		
		#compute the sqrt(<h|h>) for this polarization
		fact_p, fact_c = hp_norm*np.cos(2*pol), hc_norm*np.sin(2*pol)
		h_norm = np.sqrt(fact_p**2 + fact_c**2 + 2*fact_p*fact_c*Ipc)
		
		#compute angle(<s|h>) for this polarization
		ang_sh = np.angle(fact_p*rho_p_opt + fact_c*rho_c_opt)

		#return the parameters we have maximized over
		return min_mm, dt_shift, ang_sh, h_norm/s_norm, pol
	else:
		#return only the mismatch
		return min_mm

#function to compute 2d rotation matrix
def rotation_matrix_2D(phi):
	sph, cph = np.sin(phi), np.cos(phi)
	return np.array([[cph, -sph], [sph, cph]])

#function to update required parameters in pyEFPE dictionary
def update_pyEFPE_params(pyEFPE_params, param_vals, param_names):

	#make a copy of the dictionary to update
	p_pyEFPE = pyEFPE_params.copy()

	#loop over parameters to update
	for val, pname in zip(param_vals, param_names):

		#check if the parameter is a phase to rotate the spins
		if pname in ['phase_s', 'phase_s1', 'phase_s2']:
			#compute rotation matrix
			Rot = rotation_matrix_2D(val)
			if (pname in ['phase_s', 'phase_s1']) and ('spin1x' in p_pyEFPE) and ('spin1y' in p_pyEFPE):
				p_pyEFPE['spin1x'], p_pyEFPE['spin1y'] = np.dot(Rot, np.array([p_pyEFPE['spin1x'], p_pyEFPE['spin1y']]))
			if (pname in ['phase_s', 'phase_s2']) and ('spin2x' in p_pyEFPE) and ('spin2y' in p_pyEFPE):
				p_pyEFPE['spin2x'], p_pyEFPE['spin2y'] = np.dot(Rot, np.array([p_pyEFPE['spin2x'], p_pyEFPE['spin2y']]))

		#if param is a regular parameter, just set it
		else: p_pyEFPE[pname] = val

	return p_pyEFPE

#define function to compute mismatch varying specified parameters
def mismatch_t_ph_amp_pol_minimized_varying_params(param_vals, param_names, signal, approx_string, freqs, pyEFPE_params, asd, return_optimum_params=False):

	#update required pyEFPE parameters
	p_pyEFPE = update_pyEFPE_params(pyEFPE_params, param_vals, param_names)

	#compute waveform
	try:    hp_pyEFPE, hc_pyEFPE = pyEFPE_like_generator(approx_string, freqs, p_pyEFPE)
	except: raise Exception("%s waveform generation failed with \nparams=%s"%(approx_string, p_pyEFPE))

	#return the mismatch analytically minimized over polarization
	return mismatch_t_ph_amp_pol_minimized(signal, hp_pyEFPE, hc_pyEFPE, freqs, return_optimum_params=return_optimum_params, asd=asd)

#function to minimize the mismatch over a specified list of minimize_parameters with certain specified parameter ranges using differential_evolution
def numerically_minimize_mismatch(signal, freqs, pyEFPE_params, minimize_parameters, approx_string="pyEFPEHM", rtol_e=0.3, rtol_p=0.1, max_e=0.85, asd=None, disp=True, workers=1, maxiter=100, popsize=30):

	#use differential_evolution only if there are parameters to minimize over
	if len(minimize_parameters)>=1:

		#generate parameter bounds
		parameter_bounds = determine_pyEFPE_bounds(minimize_parameters, pyEFPE_params, rtol_e=rtol_e, rtol_p=rtol_p, max_e=max_e)

		#minimize mismatch using differential evolution
		minimize_result = differential_evolution(mismatch_t_ph_amp_pol_minimized_varying_params, parameter_bounds, args=(minimize_parameters, signal, approx_string, freqs, pyEFPE_params, asd), disp=disp, workers=workers, maxiter=maxiter, popsize=popsize)
		
		#extract information from minimization
		params_min = minimize_result.x
	
	else:
		params_min = []

	#reconstruct the parameters that were analytically maximized
	min_mm, dt_shift_sh, ang_sh, amp_sh, pol_h = mismatch_t_ph_amp_pol_minimized_varying_params(params_min, minimize_parameters, signal, approx_string, freqs, pyEFPE_params, asd, return_optimum_params=True)
	
	#return the minimum mismatch and the quantities required to reconstruct the best fitting waveform
	return min_mm, dt_shift_sh, ang_sh, amp_sh, pol_h, params_min

#function to automatically create bounds for minimization of pyEFPE mismatches
def determine_pyEFPE_bounds(parameter_names, pyEFPE_params, rtol_e=0.3, rtol_p=0.1, max_e=0.85):

	#loop over minimize parameters and append the bound of each
	bounds = []
	for pname in parameter_names:

		#for angles we can consider their full range
		#some of them span [0, 2\pi]
		if pname in ['phase', 'phase_s', 'phase_s1', 'phase_s2', 'mean_anomaly']:
			bounds.append((0, 2*np.pi))
		#some [0, \pi]
		elif pname in ['inclination', 'pol']:
			bounds.append((0, np.pi))
		#for eccentricity consider the range specified by the relative tolerance
		elif pname == 'eccentricity':
			e = pyEFPE_params[pname]
			emin = max(e*(1 - rtol_e),    0.)
			emax = min(e*(1 + rtol_e), max_e)
			bounds.append((emin, emax))
		#for other parameters, set a wiggle room given by a relative tolerance
		else:
			#extract parameters
			p = pyEFPE_params[pname]
			pmin = max(0, 1 - rtol_p)*p
			pmax =       (1 + rtol_p)*p
			bounds.append((pmin, pmax))

	#return bounds we have constructed
	return bounds

#function to initialize a pyEFPE-like frequency domain waveform generator
def pyEFPE_like_generator(approx_string, freqs, params_pyEFPE, f_min_gen_fact=None):

	#make a local copy of parameters
	p_pyEFPE = params_pyEFPE.copy()

	#if f22_start is not in p_pyEFPE, set it to minimum frequency (times f_min_gen_fact when relevant)
	#and when f_min_gen_fact is not defined, have it such that f_min_gen = f22_start
	if 'f22_start' not in p_pyEFPE:
		if f_min_gen_fact is None:
			f_min_gen_fact = 1
			p_pyEFPE['f22_start'] = np.amin(freqs)
		else:
			p_pyEFPE['f22_start'] = f_min_gen_fact*np.amin(freqs)
	elif f_min_gen_fact is None:
		f_min_gen_fact = p_pyEFPE['f22_start']/np.amin(freqs)
	
	#cases where the approximant is pyEFPE or pyEFPEHM
	if approx_string=="pyEFPE":
		return pyEFPE.pyEFPE(to_legacy_pyEFPE_params(p_pyEFPE)).generate_waveform(freqs)
	elif approx_string=="pyEFPEHM":
		return pyEFPEHM.pyEFPE(p_pyEFPE).generate_waveform(freqs)
	#cases where approximant is not in pyEFPE family
	else:
		#check if approximant is in pyseobnr
		if approx_string[:8] in ['SEOBNRv5', 'SEOBNRv6']:
	
			#the approximant in pySEOBNR is just the string
			approximant = approx_string
			
			#initialize dictionary for extra arguments
			waveform_dictionary = {}
			
			#If there is a mode_array in p_pyEFPE, set the same array in SEOBNR
			if 'mode_array' in p_pyEFPE:
				mode_array = [(mode[0], mode[1]) for mode in p_pyEFPE['mode_array']]
				waveform_dictionary.update({"return_modes": mode_array, "ModeArray": mode_array,})
			
			#set the approximant function
			approx_func = fd_from_pyseobnr_td

		#check if approximant is TEOBResumS
		elif approx_string=='TEOBResumS':

			#the approximant will be ignored by fd_from_teobresum_td
			approximant = approx_string

			#initialize dictionary for extra arguments
			waveform_dictionary = {}
			
			#If there is a mode_array in p_pyEFPE, set the same array in TEOB
			if 'mode_array' in p_pyEFPE:
				waveform_dictionary.update({'use_mode_lm': modes_to_teob_k(p_pyEFPE['mode_array']),})
			
			#set the approximant function
			approx_func = fd_from_teobresum_td

		#otherwise, assume it is in lal
		else:
			#waveflags for LAL
			waveform_dictionary = lal.CreateDict()
			if 'pn_spin_order' in p_pyEFPE:      lalsim.SimInspiralWaveformParamsInsertPNSpinOrder(waveform_dictionary, int(p_pyEFPE['pn_spin_order']))
			if 'pn_phase_order' in p_pyEFPE:     lalsim.SimInspiralWaveformParamsInsertPNPhaseOrder(waveform_dictionary, int(p_pyEFPE['pn_phase_order']))
			if 'pn_amplitude_order' in p_pyEFPE: lalsim.SimInspiralWaveformParamsInsertPNAmplitudeOrder(waveform_dictionary, int(p_pyEFPE['pn_amplitude_order']))
			
			#If there is a mode_array in p_pyEFPE, set the mode array
			if 'mode_array' in p_pyEFPE:
				mode_array_lal = lalsim.SimInspiralCreateModeArray()
				for mode in p_pyEFPE['mode_array']:
					lalsim.SimInspiralModeArrayActivateMode(mode_array_lal, mode[0], mode[1])
				lalsim.SimInspiralWaveformParamsInsertModeArray(waveform_dictionary, mode_array_lal)

			#get approximant from string
			approximant = lalsim.SimInspiralGetApproximantFromString(approx_string)
			
			#consider the cases where the LAL approximant is time-domain or frequency-domain
			if approx_string in ['SpinTaylorT4', 'SEOBNRv4P', 'IMRPhenomT', 'IMRPhenomTP']:
				approx_func = fd_from_lalsim_td
			else:
				approx_func = fd_from_lalsim_fd

		#Make sure frequencies are sorted and equally spaced
		dfs = np.diff(freqs)
		df = np.mean(dfs)
		if not np.all(dfs>0): raise ValueError("Frequencies must be strictly increasing")
		if not np.allclose(dfs, df, rtol=1e-10): raise ValueError("Frequencies must be equally spaced")

		#return polarizations
		return approx_func(p_pyEFPE['mass1'], p_pyEFPE['mass2'],
		                   p_pyEFPE['spin1x'], p_pyEFPE['spin1y'], p_pyEFPE['spin1z'],
		                   p_pyEFPE['spin2x'], p_pyEFPE['spin2y'], p_pyEFPE['spin2z'],
		                   p_pyEFPE['distance'], p_pyEFPE['inclination'], p_pyEFPE['phase'], 0,
		                   p_pyEFPE['eccentricity'], p_pyEFPE['mean_anomaly'],
		                   df, freqs[0], freqs[-1]+df, p_pyEFPE['f22_start'], waveform_dictionary, approximant, f_min_gen_fact=f_min_gen_fact)

#function to initialize signal waveform, return function that take a dictionary p with keys ['m1', 'm2', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z', 'iota', 'phiref', 'pol', 'f_max'] and outputs a waveform
def initialize_signal_generator(f_min, delta_f, params_pyEFPE, approx_string, pn_spin_order, pn_phase_order, pn_amplitude_order, f_min_gen_fact=0.8):

	#make a copy of params_pyEFPE
	p_pyEFPE = params_pyEFPE.copy()
	
	#overwrite pn orders
	if pn_spin_order is not None:      p_pyEFPE['pn_spin_order']       = pn_spin_order
	if pn_phase_order is not None:     p_pyEFPE['pn_phase_order']      = pn_phase_order
	if pn_amplitude_order is not None: p_pyEFPE['pn_amplitude_order']  = pn_amplitude_order
	
	#signal generator function that we will use
	def signal_generator(p):

		#compute frequencies
		freqs = np.arange(f_min, p['f_max'], delta_f)

		#fill pyEFPE keys with LAL keys
		for pyEFPE_key, LAL_key in pyEFPE_keys.items(): p_pyEFPE[pyEFPE_key] = p[LAL_key]

		#return polarizations and frequencies from signal generator
		return pyEFPE_like_generator(approx_string, freqs, p_pyEFPE, f_min_gen_fact=f_min_gen_fact), freqs
	
	#return the function we have created
	return signal_generator

#function to create string ID from relevant parameters
def generate_string_ID(Nsamples, precessing, approximant, psd_name, mc_low, mc_high, q_low, q_high, s1_high, s2_high, flim_type, pn_spin_order, pn_phase_order, pn_amplitude_order, params_pyEFPE, minimize_parameters=[], ecc_high=0, fmax_flim=1, string_start='_'):
	
	#if the system is precessing, add a prefix to indicate it
	string_ID = string_start
	if precessing: string_ID += 'prec_'
	
	#if the system is eccentric, add a prefix to indicate it
	if ecc_high>0: string_ID += 'ecc_%.3g_'%(ecc_high)
	
	#generate generic string ID, with approximant and parameter space used
	string_ID += approximant+'_N_%s_mc_%.3g_%.3g_q_%.3g_%.3g_s1_%.3g_s2_%.3g_fmax_%.3g%s_PN_spin_%s_phase_%s_amp_%s'%(Nsamples, mc_low, mc_high, q_low, q_high, s1_high, s2_high, fmax_flim, flim_type, pn_spin_order, pn_phase_order, pn_amplitude_order)
	
	#add it to string different kwargs in params_pyEFPE
	if 'SUA_kmax' in params_pyEFPE.keys(): string_ID += '_SUA_%s'%(params_pyEFPE['SUA_kmax'])
	if 'Amplitude_tol'in params_pyEFPE.keys(): string_ID += '_Atol_%.3g'%(params_pyEFPE['Amplitude_tol'])
	if 'mode_array' not in params_pyEFPE.keys(): string_ID += '_HMs_10'
	else: string_ID += '_HMs_%s'%(len(params_pyEFPE['mode_array']))
	
	#if we numerically minimize over any parameters, keep track of which ones
	if len(minimize_parameters)>=1:
		string_ID += '_min_'
		for min_p in minimize_parameters:
			string_ID += '%s'%(minimize_parameter_names[min_p])
	
	#if a PSD is given, add it also to the string ID
	if type(psd_name)==str: string_ID += '_'+psd_name

	return string_ID

#function to compute mismatches
def random_mismatches_with_EFPE(Nsamples, approx_string, seglen=128, f_min=20, flim_type='ISCO', fmax_flim=1, psd_name=None, distance_Mpc=10, mc_low=1.4, mc_high=2.6, q_low=1, q_high=4, s1_low=None, s1_high=1, s2_low=None, s2_high=1, ecc_high=0, precessing=True, params_pyEFPE={}, pn_spin_order=None, pn_phase_order=None, pn_amplitude_order=None, f_min_gen_fact=0.8, minimize_parameters=[], outdir='./outdir', nworkers=-1, print_progress=True, rtol_e=0.1, rtol_p=0.1, maxiter=100, popsize=15):
	
	#complete params_pyEFPE with things that dont change
	if 'f22_start' not in params_pyEFPE: params_pyEFPE['f22_start'] = f_min_gen_fact*f_min
	if 'pn_spin_order' not in params_pyEFPE: params_pyEFPE['pn_spin_order'] = pn_spin_order
	if 'pn_phase_order' not in params_pyEFPE: params_pyEFPE['pn_phase_order'] = pn_phase_order
	if 'pn_amplitude_order' not in params_pyEFPE: params_pyEFPE['pn_amplitude_order'] = pn_amplitude_order
	#overwrite distance even if given in params_pyEFPE
	params_pyEFPE['distance'] = distance_Mpc

	#compute random parameters
	params = generate_rand_params(Nsamples, mc_low=mc_low, mc_high=mc_high, q_low=q_low, q_high=q_high, s1_low=s1_low, s1_high=s1_high, s2_low=s2_low, s2_high=s2_high, ecc_high=ecc_high, precessing=precessing)

	#compute delta_f from seglen
	delta_f = 1/seglen

	#add to params the necessary things to reconstruct results
	params.update({'approx_string': approx_string, 'psd_name': psd_name, 'dL': distance_Mpc, 'seglen': seglen, 'f_min': f_min, 'flim_type': flim_type, 'fmax_flim': fmax_flim, 'delta_f': delta_f, 'precessing': precessing, 'params_pyEFPE': params_pyEFPE, 'pn_spin_order': pn_spin_order, 'pn_phase_order': pn_phase_order, 'pn_amplitude_order': pn_amplitude_order, 'minimize_parameters': minimize_parameters})

	#compute f_max to be a fraction of the ISCO
	params['f_max'] = np.floor(np.maximum(fmax_flim*params['f_'+flim_type], f_min + 10*delta_f)*seglen)*delta_f

	#generate a list with dictionaries of parameters needed for evaluation
	params['param_keys'] = param_keys
	param_dicts = [{key: params[key][i] for key in params['param_keys']} for i in range(Nsamples)]

	#if we want to print progress, number each sample
	if print_progress:
		for ip in range(Nsamples): param_dicts[ip]['ip'] = ip
	
	#initialize the signal waveform generator
	signal_generator = initialize_signal_generator(f_min, delta_f, params_pyEFPE, approx_string, pn_spin_order, pn_phase_order, pn_amplitude_order, f_min_gen_fact=f_min_gen_fact)

	#compute the asd up to a maximum frequency
	if type(psd_name)==str:
		params['freqs_asd'] = np.arange(f_min, np.amax(params['f_max'])+delta_f, delta_f)
		params['asd'] = compute_asd(delta_f, f_min, np.amax(params['f_max'])+delta_f, psd_name=psd_name)

	#global function to compute the mismatch with EFPE for a given parameter p
	global mismatch_with_EFPE
	def mismatch_with_EFPE(p):

		#compute signal waveform and frequencies
		(hp_signal, hc_signal), freqs = signal_generator(p)
		#combine polarizations
		h_signal = pol_response(hp_signal, hc_signal, p['pol'])
		
		#compute the psd
		if type(psd_name)==str: asd = np.interp(freqs, params['freqs_asd'], params['asd'])
		else:                   asd = None

		#create dictionary with appropriate pyEFPE parameters
		p_pyEFPE = params_pyEFPE.copy()
		for pyEFPE_key, LAL_key in pyEFPE_keys.items(): p_pyEFPE[pyEFPE_key] = p[LAL_key]

		#compute the minimum mismatches and the parameters for which it is minimized for each case
		output_mismatch = numerically_minimize_mismatch(h_signal, freqs, p_pyEFPE, minimize_parameters, asd=asd, rtol_e=rtol_e, rtol_p=rtol_p, maxiter=maxiter, popsize=popsize, disp=False, workers=1)
			
		#if required print information
		if print_progress: print('%s/%s -> mc=%.2g, q=%.2g, chi_eff=%.2g, chi_p=%.2g, e=%.2g -> MM: %.3g'%(p['ip']+1, Nsamples, params['mc'][p['ip']], params['q'][p['ip']], params['chi_eff'][p['ip']], params['chi_p'][p['ip']], params['ecc'][p['ip']], output_mismatch[0]))
		
		#return the output of mismatch calculation
		return output_mismatch
	
	#if the number of workers is -1, set it to the maximum
	if nworkers==-1: nworkers = max(mp.cpu_count()-1, 1)

	#Compute mismatches in parallel
	with mp.Pool(nworkers) as pool:
		output_mismatches = pool.map(mismatch_with_EFPE, param_dicts)

	#add the mismatches and parameters with which they are obtained into to params
	for i_outkey, outkey in enumerate(['mismatches', 'dt_shift_sh', 'ang_sh', 'amp_sh', 'pol_h', 'params_min']):
		params[outkey] = np.array([output[i_outkey] for ip, output in enumerate(output_mismatches)])

	#Create a dictionary with pyEFPE inputs
	all_params_pyEFPE = []
	for ip, (p, params_min) in enumerate(zip(param_dicts, params['params_min'])):
		p_pyEFPE = params_pyEFPE.copy()
		for pyEFPE_key, LAL_key in pyEFPE_keys.items(): p_pyEFPE[pyEFPE_key] = p[LAL_key]
		all_params_pyEFPE.append(update_pyEFPE_params(p_pyEFPE, params_min, minimize_parameters))
	params['all_params_pyEFPE'] = all_params_pyEFPE

	#if the directory to dump params does not exist, create it
	if not os.path.exists(outdir): os.makedirs(outdir)
	
	#create an identifiable name for the file
	string_id = generate_string_ID(Nsamples, precessing, approx_string, psd_name, mc_low, mc_high, q_low, q_high, s1_high, s2_high, flim_type, pn_spin_order, pn_phase_order, pn_amplitude_order, params_pyEFPE, minimize_parameters=minimize_parameters, ecc_high=ecc_high, fmax_flim=fmax_flim)
	
	#dump it
	with open(outdir+'/params'+string_id+'.pickle', 'wb') as handle: pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	return params

#function to plot waveforms from params
def compare_waveforms_from_params(params, plot_order='maximum mismatch'):

	#initialize the signal waveform generator
	signal_generator = initialize_signal_generator(params['f_min'], params['delta_f'], params['params_pyEFPE'], params['approx_string'], params['pn_spin_order'], params['pn_phase_order'], params['pn_amplitude_order'])
	
	#generate a list with dictionaries of parameters needed for evaluation
	param_dicts = [{key: params[key][i] for key in params['param_keys']} for i in range(len(params['mismatches']))]

	#if required, sort mismatches from largest to smaller
	if plot_order == 'maximum mismatch':   iplot = np.argsort(-params['mismatches'])
	elif plot_order == 'minimum mismatch': iplot = np.argsort(params['mismatches'])
	else:                                  iplot = np.arange(len(params['mismatches']))
	
	#now loop over parameters
	for ip in iplot:
		
		#parameters to use
		p = param_dicts[ip]

		#compute signal waveform and frequencies
		(hp_signal, hc_signal), freqs = signal_generator(p)
		h_signal = pol_response(hp_signal, hc_signal, p['pol'])
		
		#compute the psd
		if type(params['psd_name'])==str: asd = np.interp(freqs, params['freqs_asd'], params['asd'])
		else:                             asd = None

		#compute pyEFPE waveform
		wf = pyEFPEHM.pyEFPE(params['all_params_pyEFPE'][ip])
		hp_pyEFPE, hc_pyEFPE = wf.generate_waveform(freqs)
		h_pyEFPE = pol_response(hp_pyEFPE, hc_pyEFPE, params['pol_h'][ip])
		
		#adjust the amplitude, phase and reference time of pyEFPE waveform
		h_pyEFPE = (h_pyEFPE/params['amp_sh'][ip])*np.exp(-1j*(2*np.pi*freqs*params['dt_shift_sh'][ip] + params['ang_sh'][ip]))
		
		#compute mismatch expected from adjusted waveform
		hn_signal, h_norm_signal = normalize_h(h_signal, asd=asd)
		hn_pyEFPE, h_norm_pyEFPE = normalize_h(h_pyEFPE, asd=asd)
		mm_direct = 1 - np.real(np.sum(np.conj(hn_signal)*hn_pyEFPE))
		
		#compute title string
		title_str = r'$\mathcal{M}_c = %.3g M_\odot$, $q=%.2f$, $\chi_\mathrm{eff}=%.2g$, $\chi_\mathrm{p}=%.2g$, $e=%.2g$ $\rightarrow$ MM: %.3g'%(params['mc'][ip], params['q'][ip], params['chi_eff'][ip], params['chi_p'][ip], params['ecc'][ip], params['mismatches'][ip])
		
		#print informations
		print('\nmc=%.3g, q=%.2f, chi_eff=%.2g, chi_p=%.2g, e=%.2g -> MM: %s  MM_direct: %s'%(params['mc'][ip], params['q'][ip], params['chi_eff'][ip], params['chi_p'][ip], params['ecc'][ip], params['mismatches'][ip], mm_direct))
		
		#make a plot of h_signal vs h_pyEFPE
		fig = plot_h1_h2(h_signal, h_pyEFPE, freqs, asd=asd, label_1=params['approx_string'], label_2 ='pyEFPE', title_str=title_str)


# function to compute the iFFT of the signals
def compute_htd_from_hfd(hfd, freqs, pad_factor=2):
	
	#put the low frequencies back (they are 0)
	delta_f = np.mean(np.diff(freqs))
	
	#make sure we have an equispaced array of frequencies
	assert np.std(np.diff(freqs)) < 1e-8*delta_f
	
	#Add the frequencies from 0 to fmin
	low_freqs = np.arange(0, freqs[0], delta_f)
	#add the required high frequencies
	high_freqs = delta_f + np.arange(freqs[-1], max(1,pad_factor)*freqs[-1], delta_f)
	
	all_freqs = np.concatenate((low_freqs, freqs, high_freqs))
	
	#pad the strain with zeros at these low freqs
	all_hfd = np.concatenate((np.zeros(len(low_freqs), dtype=hfd.dtype), hfd, np.zeros(len(high_freqs), dtype=hfd.dtype)))

	#perform the inverse FFT, taking into account that h is the FFT of a real signal
	htd = delta_f*irfft(all_hfd, norm='forward')
	#compute corresponding array of times
	times = np.arange(len(htd))/(len(htd)*delta_f)

	return htd, times

#function to plot time and frequency domain waveforms
def plot_h1_h2(h1, h2, freqs, asd=None, label_1=r'1', label_2=r'2', show=True, figsize_fd=(12,12), figsize_td=(16,5), title_str=None, alpha=0.6):

	from matplotlib import pyplot as plt
	fig, axs = plt.subplots(2, 1, sharex=True, figsize=figsize_fd)

	############################# Frequency Domain #############################

	#plot amplitudes
	abs_h1, abs_h2 = np.abs(h1), np.abs(h2)
	axs[0].plot(freqs, abs_h1, label=r'$|\tilde{h}_\mathrm{%s}|$'%(label_1), alpha=alpha)
	axs[0].plot(freqs, abs_h2, label=r'$|\tilde{h}_\mathrm{%s}|$'%(label_2), alpha=alpha)
	if asd is not None: axs[0].plot(freqs, asd, label=r'ASD', color='k')
	axs[0].set_yscale('log')
	axs[0].set_ylabel(r'$|\tilde{h}|$ [Hz]')
	axs[0].legend()
	
	#plot proxy of phases
	norm_prod = np.conj(h1)*h2/(abs_h1*abs_h2)
	axs[1].plot(freqs, np.unwrap(np.angle(norm_prod), period=2*np.pi)/(2*np.pi), label=r'$\frac{\varphi_\mathrm{%s} - \varphi_\mathrm{%s}}{2 \pi}$'%(label_1, label_2))
	axs[1].legend()
	
	#set x-axis related stuff
	axs[1].set_xlim(freqs[0], freqs[-1])
	axs[1].set_xlabel(r'$f$ [Hz]')
	axs[1].set_xscale('log')

	#add title
	if title_str is not None:
		fig.suptitle(title_str, fontsize=22)

	#remove spaces between subplots
	plt.subplots_adjust(wspace=0, hspace=0)

	############################# Time Domain #############################

	#reconstruct whitened time domain waveforms
	if asd is None:
		h1_td, t1 = compute_htd_from_hfd(h1, freqs)
		h2_td, t2 = compute_htd_from_hfd(h2, freqs)
	else:
		h1_td, t1 = compute_htd_from_hfd(h1/asd, freqs)
		h2_td, t2 = compute_htd_from_hfd(h2/asd, freqs)

	#plot time domain waveform
	plt.figure(figsize=(16,5))
	plt.plot(t1, h1_td, label=r"$h_\mathrm{%s}$"%(label_1), alpha=alpha)
	plt.plot(t2, h2_td, label=r"$h_\mathrm{%s}$"%(label_2), alpha=alpha)
	plt.xlabel("Time [s]")
	if asd is None: plt.ylabel("Strain")
	else:           plt.ylabel("Whitened Strain")
	plt.xlim(t1[0],t2[-1])
	plt.title(title_str)
	plt.legend()
	plt.tight_layout()

	#show plot if required
	if show: plt.show()

#function to compute morlet wavelet in time domain
def td_morlet_wavelet(t, t0=0., f0=1.0, Q=8):
	#Use quality factor to compute sigma of the wavelet
	sigma = Q/(2*np.pi*f0)
	#return time domain morlet wavelet
	return np.exp(2j*np.pi*f0*(t - t0) - 0.5*np.square((t-t0)/sigma))/(np.sqrt(2*np.pi)*sigma)

#function to compute morlet wavelet in frequency domain
def fd_morlet_wavelet(f, t0=0., f0=1.0, Q=8):
	#Use quality factor to compute sigma of the wavelet
	sigma = Q/(2*np.pi*f0)
	#return frequency domain morlet wavelet
	return np.exp(-2j*np.pi*f*t0 - 2*np.square(np.pi*sigma*(f - f0)))

#function to compute the continuous wavelet transform using morlet wavelets of constant quality factor
def morlet_cwt_constantQ(signal, delta_t, fmin, fmax, Q=8, Nfreqs_fact=4):
	
	#compute the signal in the frequency domain
	fft_signal = np.fft.fft(signal)
	fft_freqs  = np.fft.fftfreq(len(signal), d=delta_t)
	
	#based on the frequency resolution, compute frequencies for wavelet transform
	Nfreqs = int(np.ceil(Nfreqs_fact*np.log2(fmax/fmin)*Q))
	freqs = np.geomspace(fmin, fmax, Nfreqs)

	#do an array to store the wavelet transform
	W = np.zeros((len(freqs), len(signal)), dtype=np.complex128)

	#loop over the central frequencies
	for i, fc in enumerate(freqs):

		#compute the wavelet transform using the ifft
		W[i] = np.fft.ifft(fft_signal*np.conj(fd_morlet_wavelet(fft_freqs, f0=fc, Q=Q)))

	#return the frequency array and the wavelet transform
	return freqs, W

#function to compute the time-domain amplitude and phase difference between a test signal h_test and a reference h_ref using wavelets
def compute_td_dA_dphi_wavelets(h_test, h_ref, dt, fmin, fmax, Q=8, Nfreqs_fact=4):
	
	#compute the continuous wavelet transform of both signals
	freqs, W_test = morlet_cwt_constantQ(h_test, dt, fmin, fmax, Q=Q, Nfreqs_fact=Nfreqs_fact)
	freqs, W_ref  = morlet_cwt_constantQ(h_ref , dt, fmin, fmax, Q=Q, Nfreqs_fact=Nfreqs_fact)

	#deal with numpy trapz -> trapezoid renaming
	try:    trapezoid = np.trapezoid
	except: trapezoid = np.trapz
	
	#compute the integrals over frequency, weigh them by f^{-1} to correct for morlet transform being done at constant Q
	weight = 1/freqs[:,None]
	rho_ref  = np.sqrt(trapezoid(weight*np.abs(W_ref)**2 , x=freqs, axis=0))
	rho_test = np.sqrt(trapezoid(weight*np.abs(W_test)**2, x=freqs, axis=0))
	WrefWtest = trapezoid(weight*W_ref*np.conj(W_test), x=freqs, axis=0)
	
	#look for places where the amplitude of the reference signal does not vanish
	ivalid = (rho_ref>0) & (rho_ref>1e-12*np.mean(rho_ref))

	#compute amplitude difference
	dA = np.zeros_like(rho_ref)
	dA[ivalid] = np.sqrt(rho_test[ivalid]/rho_ref[ivalid]) - 1
	
	#compute phase difference
	dphi = np.unwrap(np.where(ivalid, np.angle(WrefWtest), 0), period=2*np.pi)
	#substract required constant number of cycles to have a median that it closer to 0
	dphi = dphi - 2*np.pi*np.round(np.median(dphi)/(2*np.pi))
	
	return dA, dphi

#function to compute phase difference of two time domain waveforms using hilbert transform
def compute_td_dA_dphi_hilbert(h_test, h_ref):
	
	#use hilbert transform to convert A(t)\cos(\phi(t)) into A(t)\exp(i\phi(t))
	from scipy.signal import hilbert
	z_test = hilbert(h_test)
	z_ref = hilbert(h_ref)

	#look for places where the amplitude of the reference signal does not vanish
	abs_zref = np.abs(z_ref)
	ivalid = (abs_zref>0) & (abs_zref>1e-12*np.mean(abs_zref))

	#compute amplitude difference
	dA = np.zeros_like(abs_zref)
	dA[ivalid] = np.abs(z_test[ivalid]/z_ref[ivalid]) - 1
	
	#compute phase difference
	dphi = np.unwrap(np.where(ivalid, np.angle(z_test*np.conj(z_ref)), 0), period=2*np.pi)
	#substract required constant number of cycles to have a median that it closer to 0
	dphi = dphi - 2*np.pi*np.round(np.median(dphi)/(2*np.pi))

	return dA, dphi

#function to compute EFPE duration
def compute_duration_of_Mc_f0_e0(Mc, f0, e0):
	return (5./256.)*((t_sun_s*Mc)**(-5./3.))*((np.pi*f0)**(-8./3.))*((1-e0*e0)**3.5)*pyEFPEHM.functions.F_tLO_series(e0*e0)

#function to find the frequency from which the waveform has a certain duration using the bisection method
def find_f22_start_for_duration(duration, f22_start_min, f22_start_max, Mc, e0, max_iter=100, rtol=1e-10):
	
	#compute duration from minimum and maximum frequencies
	durL = compute_duration_of_Mc_f0_e0(Mc, f22_start_min, e0)
	durR = compute_duration_of_Mc_f0_e0(Mc, f22_start_max, e0)
	
	#check if maximum duration is smaller than duration
	if durL<duration:
		return f22_start_min
	#check if minimum duration is greater than duration
	if durR>duration:
		print("Warning: From f22_start_max=%s, expected duration is %ss > duration=%ss"%(f22_start_max, durR, duration))
		return f22_start_max

	#do bisection algorithm to find frequency from which the waveform has a certain duration
	fL, fR = f22_start_min, f22_start_max
	for i in range(max_iter):
		
		#compute middle frequency and duration
		f_mid = 0.5*(fL + fR)
		dur_mid = compute_duration_of_Mc_f0_e0(Mc, f_mid, e0)

		#Update segment f22_start is in using that duration is monotonously decreasing
		if   dur_mid>duration : fL, durL = f_mid, dur_mid
		elif dur_mid<duration : fR, durR = f_mid, dur_mid
		elif dur_mid==duration: return f_mid
	
		#check if desired tolerance has been achieved
		if (abs(fR - fL)<rtol*abs(fR + fL)) or (abs(durL - durR)<rtol*abs(durL + durR)):
			break

	#return final middle frequency
	return 0.5*(fL + fR)

