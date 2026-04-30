import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

import numpy as np
import warnings
import time
from scipy.signal.windows import tukey
from scipy.fft import fft, rfft, irfft
from scipy.optimize import minimize_scalar, minimize, differential_evolution
from utils_compute_mismatches import *

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")  # silence LAL warnings
from pyseobnr.generate_waveform import GenerateWaveform, generate_modes_opt

#function to get waveform in frequency domain from time domain waveform
def td_to_fd(hp_td, hc_td, f_min, f_max, max_alpha=0.05, apply_tukey=True, asym_tukey=True, df_factor=2):

	#compute the duration of the waveform
	duration = 2**np.ceil(np.log2(hp_td.deltaT*hp_td.data.length))

	#compute the size of the array needed for the specified duration
	df_factor = max(1, int(df_factor))
	N_tsamples = int(df_factor*duration/hp_td.deltaT)

	#compute indexes of target frequencies
	i_start = int(df_factor*f_min*duration)
	i_end = int(df_factor*f_max*duration)
	
	#loop over polarizations
	hpc_fd = []
	for h_LAL in [hp_td, hc_td]:

		#compute arrays with data
		h = np.array(h_LAL.data.data)

		#if we generated too much stuff at the beginning, remove it
		if len(h)>N_tsamples:
			h = h[-N_tsamples:]

		#if required, apply Tukey window to reduce spectral leakage
		if apply_tukey:
			
			#compute the roll of from the f_min requirement
			roll_off = 2/f_min
			#compute Tukey window
			window = tukey(len(h), alpha=min(max_alpha, 2*roll_off/(h_LAL.deltaT*len(h))))
			#if requested, apply tukey only to beginninf of signal (end might be regularized by ringdown)
			if asym_tukey: window[(len(h)//2):] = 1
			#apply the tukey window
			h *= window

		#if required, prepend with zeros to get the desired frequency resolution
		if len(h)<N_tsamples:
			h = np.append(np.zeros(N_tsamples - len(h)), h)

		#compute fast fourier transform at the required frequencies
		hpc_fd.append(h_LAL.deltaT*rfft(h)[i_start:i_end:df_factor])

	#compute frequency array
	freqs = np.arange(f_min, f_max, 1/duration)

	#return frequency-domain polarizations and frequencies
	return hpc_fd, freqs

#define function to compute mismatch varying initial eccentricity and mean anomaly
def mismatch_t_ph_amp_pol_minimized_varying_ecc_ell(ecc_ell, signal, freqs, p_pyEFPE, asd, return_optimum_params=False):
	
	#update relevant pyEFPE parameters
	p_pyEFPE['e_start'] = ecc_ell[0]
	p_pyEFPE['mean_anomaly_start'] = ecc_ell[1]
	
	#Initialize pyEFPE waveform
	wf = pyEFPEHM.pyEFPE(p_pyEFPE)
	#compute pyEFPE waveform
	hp_pyEFPE, hc_pyEFPE = wf.generate_waveform(freqs)

	#compute the mismatch maximized over polarization
	return mismatch_t_ph_amp_pol_minimized(signal, hp_pyEFPE, hc_pyEFPE, freqs, return_optimum_params=return_optimum_params, asd=asd)


#function to minimize the mismatch over eccentricity and mean anomaly
def minimize_eccentric_mismatch_pyEFPE(signal, freqs, pyEFPE_params, asd=None, disp=True, Delta_e=0.05, workers=1):

	#make a local dictionary to update
	p_pyEFPE = pyEFPE_params.copy()
	
	#compute grid of eccentricity and mean anomaly to minimize over
	emin, emax = np.maximum(0, pyEFPE_params['e_start'] + Delta_e*np.array([-1, 1]))
	ellmin, ellmax = 0, 2*np.pi

	minimize_result = differential_evolution(mismatch_t_ph_amp_pol_minimized_varying_ecc_ell, [(emin, emax), (ellmin, ellmax)], args=(signal, freqs, p_pyEFPE, asd), disp=disp, workers=workers)

	min_mm = minimize_result.fun
	min_ecc, min_ell = minimize_result.x
	
	print('mm=%.4g -> ecc = %.4g, ell = %.4g (Ncalls=%s)'%(min_mm, min_ecc, min_ell, minimize_result.nfev))
	
	#compute the parameters that have been analytically maximized
	min_mm_2, dt_shift, ang_sh, amp_sh, pol_h, = mismatch_t_ph_amp_pol_minimized_varying_ecc_ell(minimize_result.x, signal, freqs, p_pyEFPE, asd, return_optimum_params=True)
		
	return min_mm, dt_shift, ang_sh, amp_sh, pol_h, min_ecc, min_ell

#function to compute phase difference of 2 time domain waveforms
def compute_tdomain_dphase_damp(h_1, h_2):
	
	#use hilbert transform to convert A(t)\cos(\phi(t)) into A(t)\exp(i\phi(t))
	from scipy.signal import hilbert
	z_1 = hilbert(h_1)
	z_2 = hilbert(h_2)
	
	#compute dephasing by unwrapping the phase of A_1(t)*A_2(t)*exp(i(\phi_1 (t) - \phi_2(t)))
	dphase = np.unwrap(np.angle(z_1*np.conj(z_2)))
	
	#compute symetric relatice error of amplitudes
	damp = 2*(np.abs(z_1) - np.abs(z_2))/(np.abs(z_1) + np.abs(z_2))
	
	return dphase, damp
	
####################################### Inputs #######################################

m1 = 4.0
m2 = 4.0
s1x = 0.0
s1y = 0.0
s1z = 0.2
s2x = 0.0
s2y = 0.0
s2z = 0.1
rel_anomaly = np.pi
deltaT = 1.0 / 32768.0
f22_start = 16
distance = 1000.0
inclination = 0
pol = 0

#frequency range to consider
fmin = 20
fmaxmax = 2048

#eccentricities and fmaxs to consider
eccentricities = np.linspace(0, 0.6, 5)
fmaxs = np.geomspace(40, 500, 5).astype(int)

#pn orders to consider in the different parts of the waveform
pn_phase_order     = 9
pn_spin_order      = 8
pn_amplitude_order = 2

#mode array to consider
mode_array = [(2,2)] #[(2,1),(2,2),(3,2),(3,3),(4,3),(4,4)]

#output directory
outdir='./outdir/waveform_comparisons'

#file to save results in
string_ID = '_m1_%.2g_m2_%.2g_s1z_%.2g_s2z_%.2g_ecc_%.2g_%.2g_%s_fmax_%s_%s_%s_pn_%s_%s_%s_HMs_%s'%(m1, m2, s1z, s2z, eccentricities[0], eccentricities[-1], len(eccentricities), fmaxs[0], fmaxs[-1], len(fmaxs), pn_phase_order, pn_spin_order, pn_amplitude_order, len(mode_array))
result_file = 'SEOBNRv5EHM_vs_pyEFPE_min_mm_results'+string_ID+'.npy'

#number of workers for paralelization of differential evolution
workers = 4

#choose if you want to plot only the highest frequency
plot_only_highest_freq = True

######################################################################################

#create base SEOBNR dictionary
base_SEOBNR_params_dict = {
    "mass1": m1,
    "mass2": m2,
    "spin1x": s1x,
    "spin1y": s1y,
    "spin1z": s1z,
    "spin2x": s2x,
    "spin2y": s2y,
    "spin2z": s2z,
    "deltaT": deltaT,
    "f22_start": f22_start,
    "distance": distance,
    "inclination": inclination,
    "f_max": 0.5/deltaT,
    "return_modes": mode_array,
    "ModeArray": mode_array,
    "approximant": "SEOBNRv5EHM",
    "rel_anomaly": rel_anomaly,
    "EccIC": 1,  # EccIC = 0 for instantaneous initial orbital frequency, and EccIC = 1 for orbit-averaged initial orbital frequency
    "lmax_nyquist": 1, # Disable check of ringdown being below nyquist freq. 
}

#create base pyEFPE dictionary
base_pyEFPE_params_dict = {
    "mass1": m1,
    "mass2": m2,
    "spin1x": s1x,
    "spin1y": s1y,
    "spin1z": s1z,
    "spin2x": s2x,
    "spin2y": s2y,
    "spin2z": s2z,
    "f22_start": f22_start,
    "distance": distance,
    "inclination": inclination,
    "mean_anomaly_start": rel_anomaly,
    "pn_phase_order": pn_phase_order,
    "pn_spin_order": pn_spin_order,
    "pn_amplitude_order": pn_amplitude_order,
    "mode_array": mode_array,
}

######################################################################################

#try to load min_mm_results from result file
try:
	min_mm_results = np.load(outdir+'/'+result_file)

except:

	#loop over eccentricities
	min_mm_results = list()
	for eccentricity in eccentricities:

		# Create dictionary with SEOBNR parameters
		SEOBNR_params_dict = base_SEOBNR_params_dict.copy()
		SEOBNR_params_dict["eccentricity"] = eccentricity

		# Create dictionary with pyEFPE parameters
		pyEFPE_params_dict = base_pyEFPE_params_dict.copy()
		pyEFPE_params_dict["e_start"] = eccentricity

		#time waveform generation
		start_evaltime = time.time()

		# Generate time-domain polarizations - As LAL REAL8TimeSeries
		SEOB_wfm_gen = GenerateWaveform(SEOBNR_params_dict)
		hp_SEOB_td, hc_SEOB_td = SEOB_wfm_gen.generate_td_polarizations()

		print("\nWaveform evaluation time: %s seconds" % (time.time() - start_evaltime))

		#compute frequency domain polarizations
		(hp_SEOB_fd, hc_SEOB_fd), freqs_all = td_to_fd(hp_SEOB_td, hc_SEOB_td, fmin, fmaxmax)

		#compute the projection
		h_SEOB_fd_all = pol_response(hp_SEOB_fd, hc_SEOB_fd, pol)

		#loop over fmax
		for fmax in fmaxs:
			
			print('\nfmax=%sHz, ecc=%s'%(fmax, eccentricity))

			#compute relevant frequencies
			idxs_sel = (freqs_all<fmax)
			freqs = freqs_all[idxs_sel]
			h_SEOB_fd = h_SEOB_fd_all[idxs_sel]

			#compute mismatch minimized over eccentricity and mean anomaly
			min_mm_results.append(minimize_eccentric_mismatch_pyEFPE(h_SEOB_fd, freqs, pyEFPE_params_dict, asd=None, disp=True, workers=workers))

	#convert min_mm_results to numpy array
	min_mm_results = np.array(min_mm_results)

	#if directory to save data does not exist, create it
	if not os.path.exists(outdir): os.makedirs(outdir)

	#save the result array
	np.save(outdir+'/'+result_file, min_mm_results)

####################################### Plots #######################################

#if the directory for plots does not exist, create it
plots_dir = outdir+'/Plots/'
if not os.path.exists(plots_dir): os.makedirs(plots_dir)

import matplotlib.pyplot as plt
plt.rcParams.update({
	'lines.linewidth': 2.5,
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
        'patch.force_edgecolor':True,
})

fig, ax = plt.subplots(figsize=(12,12), nrows=2, sharex=True, sharey=True)
ax[0].set_yscale('log')
ax[1].set_yscale('log')
ax[0].set_title('$m_1 = %.2g M_\odot$, $m_2 = %.2g M_\odot$, $f_\mathrm{min} = %s$Hz'%(m1, m2, fmin))
pmesh0 = ax[0].pcolormesh(eccentricities, fmaxs, np.log10(np.transpose(min_mm_results[:,0].reshape(len(eccentricities),len(fmaxs)))), antialiased=True, edgecolors='face')
cbar0 = fig.colorbar(pmesh0, ax=ax[0], label=r'$\log_{10}{\mathcal{MM}}$')
pmesh1 = ax[1].pcolormesh(eccentricities, fmaxs, np.transpose(eccentricities[:,None] - min_mm_results[:,5].reshape(len(eccentricities),len(fmaxs))), antialiased=True, edgecolors='face')
cbar1 = fig.colorbar(pmesh1, ax=ax[1], label=r'$e^\mathrm{v5EHM}_{%s\mathrm{Hz}} - e^\mathrm{pyEFPE}_{%s\mathrm{Hz}}$'%(f22_start, f22_start))
ax[0].set_ylabel(r'$f_\mathrm{max}$ [Hz]')
ax[0].set_ylim(fmaxs[0], fmaxs[-1])
ax[1].set_ylabel(r'$f_\mathrm{max}$ [Hz]')
ax[1].set_ylim(fmaxs[0], fmaxs[-1])
ax[1].set_xlabel(r'$e^\mathrm{v5EHM}_{%s\mathrm{Hz}}$'%(f22_start))
ax[1].set_xlim(eccentricities[0], eccentricities[-1])
plt.tight_layout()
plt.savefig(plots_dir+'/v5EHM_vs_pyEFPE'+string_ID+'.pdf')

plt.show()

###########################################################################################################

iplot = 0
for eccentricity in eccentricities:

	# Create dictionary with SEOBNR parameters
	SEOBNR_params_dict = base_SEOBNR_params_dict.copy()
	SEOBNR_params_dict["eccentricity"] = eccentricity

	# Create dictionary with pyEFPE parameters
	pyEFPE_params_dict = base_pyEFPE_params_dict.copy()
	pyEFPE_params_dict["e_start"] = eccentricity

	#time waveform generation
	start_evaltime = time.time()

	# Generate time-domain polarizations - As LAL REAL8TimeSeries
	SEOB_wfm_gen = GenerateWaveform(SEOBNR_params_dict)
	hp_SEOB_td, hc_SEOB_td = SEOB_wfm_gen.generate_td_polarizations()

	print("\nWaveform evaluation time: %s seconds" % (time.time() - start_evaltime))

	#compute frequency domain polarizations
	(hp_SEOB_fd, hc_SEOB_fd), freqs_all = td_to_fd(hp_SEOB_td, hc_SEOB_td, fmin, fmaxmax)

	#compute the projection
	h_SEOB_fd_all = pol_response(hp_SEOB_fd, hc_SEOB_fd, pol)

	#loop over fmax
	for fmax in fmaxs:

		if plot_only_highest_freq and (fmax == fmaxs[-1]):
			print('\nfmax=%sHz, ecc=%s'%(fmax, eccentricity))

			#compute relevant frequencies
			idxs_sel = (freqs_all<fmax)
			freqs = freqs_all[idxs_sel]
			h_SEOB_fd = h_SEOB_fd_all[idxs_sel]

			#extract parameters
			min_mm, dt_shift, ang_sh, amp_sh, pol_h, min_ecc, min_ell = min_mm_results[iplot]
			print('min_mm=%s, min_ecc=%s'%(min_mm, min_ecc))
			print('dt_shift=%.3g, ang_sh=%.3g, amp_sh=%.3g'%(dt_shift, ang_sh, amp_sh))
			#update relevant pyEFPE parameters
			pyEFPE_params_dict['e_start'] = min_ecc
			pyEFPE_params_dict['mean_anomaly_start'] = min_ell
			
			#Initialize pyEFPE waveform
			wf = pyEFPEHM.pyEFPE(pyEFPE_params_dict)
			#compute pyEFPE waveform
			hp_pyEFPE, hc_pyEFPE = wf.generate_waveform(freqs)

			#correct it with the different things
			h_pyEFPE = (pol_response(hp_pyEFPE, hc_pyEFPE, pol_h)/amp_sh)*np.exp(-1j*(2*np.pi*freqs*dt_shift + ang_sh))

			#plot frequency domain waveform
			plot_h1_h2(h_SEOB_fd, h_pyEFPE, freqs, label_1='EOB', label_2='EFPE', title_str=r'$f_\mathrm{max}=%s$Hz, $e_\mathrm{EOB}=%.3g$, $e_\mathrm{EFPE}=%.3g$ $\rightarrow$ $\mathcal{MM} = %.3g$'%(fmax, eccentricity, min_ecc, min_mm), show=False)

			#Plot waveforms in the actual time-domain
			#compute time-array from pySEOBNR
			t_SEOB = float(hp_SEOB_td.epoch) + hp_SEOB_td.deltaT*np.arange(len(hp_SEOB_td.data.data))
			h_SEOB_td = pol_response(hp_SEOB_td.data.data, hc_SEOB_td.data.data, pol)

			#compute time-domain time-shift (when we computed Fourier transform of EOB, we did t_EOB[0]=0=t_EOB[-1], due to periodicity of FFT)
			dt_shift_td = dt_shift + t_SEOB[0]
			#compute time-domain pyEFPE waveform
			hp_pyEFPE_td, hc_pyEFPE_td = wf.generate_tdomain_waveform(t_SEOB)
			#compute frequency domain waveform to apply the phase corrections
			hp_pyEFPE_fft, hc_pyEFPE_fft = rfft(hp_pyEFPE_td), rfft(hc_pyEFPE_td)
			#compute correction due to overal phase and time shift
			freqs_fft = np.arange(len(hp_pyEFPE_fft))/(hp_SEOB_td.deltaT*len(t_SEOB))
			h_pyEFPE_fft = (pol_response(hp_pyEFPE_fft, hc_pyEFPE_fft, pol_h)/amp_sh)*np.exp(-1j*(2*np.pi*freqs_fft*dt_shift_td + ang_sh))
			#re-compute time domain waveform
			h_pyEFPE_td = irfft(h_pyEFPE_fft, n=len(t_SEOB))

			plt.figure(figsize=(16,5))
			plt.plot(t_SEOB, h_SEOB_td,   label=r"$h_\mathrm{EOB}$" , alpha=0.6)
			plt.plot(t_SEOB, h_pyEFPE_td, label=r"$h_\mathrm{EFPE}$", alpha=0.6)
			plt.xlabel("Time [s]")
			plt.ylabel("Strain")
			plt.xlim(t_SEOB[0],t_SEOB[-1])
			plt.title(r'$f_\mathrm{max}=%s$Hz, $e_\mathrm{EOB}=%.3g$, $e_\mathrm{EFPE}=%.3g$ $\rightarrow$ $\mathcal{MM} = %.3g$'%(fmax, eccentricity, min_ecc, min_mm))
			plt.legend(loc='upper left')
			plt.tight_layout()

			#compute time domain dephasing and relative amplitude error
			dphi_td, damp_td = compute_tdomain_dphase_damp(h_SEOB_td, h_pyEFPE_td)

			plt.figure(figsize=(12,8))
			plt.plot(t_SEOB, dphi_td, label=r"$\phi_\mathrm{EOB} - \phi_\mathrm{EFPE}$")
			plt.plot(t_SEOB, damp_td, label=r"$2\frac{A_\mathrm{EOB} -A_\mathrm{EFPE}}{A_\mathrm{EOB} + A_\mathrm{EFPE}}$")
			plt.xlabel("Time [s]")
			plt.xlim(t_SEOB[0],t_SEOB[-1])
			plt.yscale('symlog', linthresh=1e-3)
			plt.title(r'$f_\mathrm{max}=%s$Hz, $e_\mathrm{EOB}=%.3g$, $e_\mathrm{EFPE}=%.3g$ $\rightarrow$ $\mathcal{MM} = %.3g$'%(fmax, eccentricity, min_ecc, min_mm))
			plt.legend(loc='upper left')
			plt.tight_layout()

			plt.show()

		#add to the counter
		iplot += 1
