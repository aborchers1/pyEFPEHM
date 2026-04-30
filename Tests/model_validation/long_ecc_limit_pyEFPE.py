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

import time
start_runtime = time.time()

####################################### Inputs #######################################

#model to consider
approx_string = "SEOBNRv5EHM"

#simulation parameters
q = 1.0    #0.5
s1x = 0.0
s1y = 0.0
s1z = 0.0  #0.3
s2x = 0.0
s2y = 0.0
s2z = 0.0  #0.2
rel_anomaly = np.pi
distance = 1000.0
inclination = 0
phiref = 0
pol = 0

#chirp masses and eccentricities to consider
chirp_masses = np.geomspace(0.6, 10, 25)
eccentricities = np.linspace(0, 0.6, 12)

#asd to consider in mismatches (set to None for no asd)
psd_name = 'AplusDesign' #None

#parameters to optimize over
minimize_parameters = ['phi_start', 'e_start', 'mean_anomaly_start']
#Relative wigle room to allow eccentricity
rtol_e  = 0.2
rtol_p = 0.1

#frequency range to consider
f_min = 20
f_min_gen_fact = 0.8
fmax_fISCO = 0.6

#minimum segment length (in s)
seglen_min = 4

#pn orders to consider in the different parts of the waveform
pn_phase_order     = 9
pn_spin_order      = 8
pn_amplitude_order = 2
#mode array to consider
mode_array = [(2,2)] #[(2,1),(2,2),(3,2),(3,3),(4,3),(4,4)]

#extra pyEFPE params
params_pyEFPE = {}

#kwargs for differential evolution
workers = 8
popsize = 16
maxiter = 100

#output directory
outdir='./outdir/waveform_comparisons'

#arguments for plt.contourf
levels = 20

######################################################################################

#file to save results in
string_ID = '_q_%.2g_s1z_%.2g_s2z_%.2g_mc_%.2g_%.2g_%s_ecc_%.2g_%.2g_%s_fmax_%.2gISCO_pn_%s_%s_%s_HMs_%s_%s'%(q, s1z, s2z, chirp_masses[0], chirp_masses[-1], len(chirp_masses), eccentricities[0], eccentricities[-1], len(eccentricities), fmax_fISCO, pn_phase_order, pn_spin_order, pn_amplitude_order, len(mode_array), psd_name)
result_file = outdir+'/'+approx_string+'_vs_pyEFPE_min_mm_results'+string_ID+'.pickle'

#create base pyEFPE dictionary
params_pyEFPE.update({
    "spin1x": s1x,
    "spin1y": s1y,
    "spin1z": s1z,
    "spin2x": s2x,
    "spin2y": s2y,
    "spin2z": s2z,
    "f22_start": f_min_gen_fact*f_min,
    "distance": distance,
    "inclination": inclination,
    "phi_start": phiref,
    "mean_anomaly_start": rel_anomaly,
    "pn_phase_order": pn_phase_order,
    "pn_spin_order": pn_spin_order,
    "pn_amplitude_order": pn_amplitude_order,
    "mode_array": mode_array,
})

#try to load result dictionary
try:
	with open(result_file, 'rb') as handle: result = pickle.load(handle)
except:
	print('Could not load', result_file)

	#loop over chirp masses
	seglens = np.zeros((len(chirp_masses), len(eccentricities)), dtype=int)
	f_maxs, min_mm, dt_shift_sh, ang_sh, amp_sh, pol_h = [np.zeros((len(chirp_masses), len(eccentricities)), dtype=np.float64) for _ in range(6)]
	min_pyEFPE_params = []
	for imc, mc in enumerate(chirp_masses):

		#loop over eccentricity
		min_pyEFPE_params.append([])
		for iecc, ecc in enumerate(eccentricities):

			#compute m1, m2 from mc, q
			m1 = mc*(q**-0.6)*((1 + q)**0.2)
			m2 = m1*q
			M = m1 + m2
			
			#estimate the waveform duration
			PN_duration = -pyEFPEHM.functions.tLO_func((np.pi*f_min_gen_fact*f_min*M*t_sun_s)**(1./3.)/np.sqrt(1 - ecc*ecc), ecc*ecc, m1*t_sun_s, m2*t_sun_s) 
			
			#estimate the segment length
			seglens[imc,iecc] = max(2**np.ceil(np.log2(PN_duration)), seglen_min)
			delta_f = 1./seglens[imc,iecc]
			
			#compute the ISCO frequency
			fISCO = 1/((6**1.5)*np.pi*t_sun_s*M)
			
			#compute the maximum frequency to analyze
			f_maxs[imc,iecc] = np.floor(np.maximum(fmax_fISCO*fISCO, f_min + 10*delta_f)*seglens[imc,iecc])*delta_f

			#compute the asd up to a maximum frequency
			if type(psd_name)==str:
				asd = compute_asd(delta_f, f_min, f_maxs[imc,iecc], psd_name=psd_name)
			else:   asd = None

			#initialize the signal waveform generator
			signal_generator = initialize_signal_generator(f_min, delta_f, params_pyEFPE, approx_string, pn_spin_order, pn_phase_order, pn_amplitude_order, f_min_gen_fact=f_min_gen_fact)

			#compute pyEFPE dictionary
			p_pyEFPE = params_pyEFPE.copy()
			p_pyEFPE.update({'mass1': m1, 'mass2': m2, 'e_start': ecc})

			#compute signal waveform and frequencies
			p = {LAL_key: p_pyEFPE[pyEFPE_key] for pyEFPE_key, LAL_key in pyEFPE_keys.items()}
			p["f_max"] = f_maxs[imc,iecc]
			(hp_signal, hc_signal), freqs = signal_generator(p)
			#combine polarizations
			h_signal = pol_response(hp_signal, hc_signal, pol)

			print('\nmc=%.2gMsun, ecc=%.2g, seglen=%.fs, f_max=%.fHz'%(mc, ecc, seglens[imc,iecc], f_maxs[imc,iecc]))

			#compute the minimum mismatches and the parameters for which it is minimized for each case
			min_mm[imc,iecc], dt_shift_sh[imc,iecc], ang_sh[imc,iecc], amp_sh[imc,iecc], pol_h[imc,iecc], params_min = numerically_minimize_mismatch(h_signal, freqs, p_pyEFPE, minimize_parameters, asd=asd, rtol_e=rtol_e, rtol_p=rtol_p, maxiter=maxiter, popsize=popsize, disp=True, workers=workers)

			print('mm =', min_mm[imc,iecc])
			print('Minimum pyEFPE params: '+''.join(['%s = %.3g, '%(name, val) for name, val in zip(minimize_parameters, params_min)])[:-2])

			#compute best fit pyEFPE waveform
			min_pyEFPE_params[-1].append(update_pyEFPE_params(p_pyEFPE, params_min, minimize_parameters))
			
		#put everything in a dictionary and save it
		result = {'f_min': f_min, 'f_maxs': f_maxs, 'seglens': seglens,
			  'minimize_parameters': minimize_parameters, 'rtol_e': rtol_e, 'rtol_p': rtol_p, 'maxiter': maxiter, 'popsize': popsize,
			  'min_mm': min_mm, 'dt_shift_sh':dt_shift_sh, 'ang_sh':ang_sh, 'amp_sh':amp_sh, 'pol_h':pol_h, 'min_pyEFPE_params':min_pyEFPE_params,
			  }

		#save result dictionary
		if not os.path.exists(outdir): os.makedirs(outdir)
		with open(result_file, 'wb') as handle: pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

#make a plot of the mismatch as a function of Mc and ecc
plt.figure(figsize=(12,8))
plt.contourf(chirp_masses, eccentricities, np.log10(np.transpose(result['min_mm'])), levels=levels)
plt.colorbar(label=r'$\log_{10}{\overline{\mathcal{MM}}}$')
plt.ylabel(r'$e_{%.3g \mathrm{Hz}}$'%(params_pyEFPE['f22_start']))
plt.ylim(eccentricities[0], eccentricities[-1])
plt.xlabel(r'$\mathcal{M}_c$ [$M_\odot$]')
plt.xlim(chirp_masses[0], chirp_masses[-1])
plt.xscale('log')
plt.title(r'$\mathtt{%s}$ ($q=%.2g$, $s_{1z}=%.2g$, $s_{2z}=%.2g$)'%(approx_string, q, s1z, s2z))
plt.tight_layout()

plt.savefig(plots_dir+'/mismatch_Mc_ecc_'+approx_string+string_ID+'.pdf')

#Runtime
print("\nRuntime: %s seconds" % (time.time() - start_runtime))

plt.show()
