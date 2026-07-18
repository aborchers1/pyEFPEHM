import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

import sxs
import numpy as np
from scipy.fft import rfft, irfft, rfftfreq
from scipy.optimize import minimize_scalar, minimize, differential_evolution
from utils_compute_mismatches import *

#universal constants
t_sun_s = 4.92549094831e-6  #GMsun/c**3 [s]
Mpc_s = 1.02927125054339e14 #Mpc/c in [s]

#######################################################################################

#function to load sxs simulation
def load_sxs_sim(sxs_id):

	# Obtain and load the strain data
	sim = sxs.load(sxs_id)
	hraw = sim.h

	#obtain the parameters of this simulation from the catalog (contains more things than sim.metadata)
	sim_params = sxs.load("dataframe", tag="3.0.0").loc[sxs_id].to_dict()

	return hraw, sim_params

#function to preprocess sxs waveform in time-domain
def prepare_sxs_h(hraw_sxs, theta, phi, pol):

	#preprocess waveform
	hprep = hraw_sxs.preprocess(evaluate_directions=(theta, phi))

	#compute projected time domain waveform
	ht = np.cos(2*pol)*hprep.real + np.sin(2*pol)*hprep.imag

	#compute the array of times
	times = hprep.t
	
	#return them
	return ht, times

#function to obtain frequency domain strain from sxs
def compute_hfft_freqs(ht, times, fmin=0, fmax=np.inf):

	#compute time-step
	delta_t = np.mean(np.diff(times))

	#compute frequency domain waveform
	hfft = delta_t*rfft(ht)

	#compute frequencies
	freqs = rfftfreq(len(ht), d=delta_t)

	#find the frequencies that are requested
	isel = ((freqs >= fmin) & (freqs <= fmax))
	
	#return frequency domain strain and frequencies
	return hfft[isel], freqs[isel]

#function to set the total mass (in Msun) and distance in (Mpc) of the frequency domain waveform
def hfft_set_M_dL_BBH(hfft_sim, freqs_sim, fmin=0, fmax=np.inf, M=60, dL=100):
	
	#convert mass and distance to seconds
	M_s  =  M*t_sun_s
	dL_s = dL*Mpc_s
	
	#stretch frequecies
	freqs_phys = freqs_sim/M_s
	
	#dilute strain (extra factor of M comes from converting dt of fft)
	hfft_phys  = M_s*(M_s/dL_s)*hfft_sim

	#find the frequencies that are requested
	isel = ((freqs_phys >= fmin) & (freqs_phys <= fmax))

	#return them
	return hfft_phys[isel], freqs_phys[isel]

#function to compute average frequency from instantaneous frequency at leading PN order, eccentricity, e, and mean anomaly, l
def compute_favg_from_finst_0PN(finst, e, l, atol=1e-10, maxiter=10):
	
	#make sure mean anomaly is [-pi, pi]
	l = np.mod(l + np.pi, 2*np.pi) - np.pi
	
	#initial guess of the true anomaly
	u = l + e*np.sin(l)

	#apply Newton-Rhapson method to solve Kepler Eq. f(u) = u - e*sin(u) - l = 0
	for i in range(maxiter):
		#update u to u - f(u)/f'(u)
		du = (u - e*np.sin(u) - l)/(1 - e*np.cos(u))
		u -= du
		#check if it has converged
		if np.abs(du)<atol: break

	#return average frequency computed using leading PN expression
	return finst*np.square(1 - e*np.cos(u))/np.sqrt(1 - e*e)

#function to compute a matrix that rotates v to the z axis
def rotation_to_z(v):

	#make sure v is normalized
	v = np.asarray(v)/np.linalg.norm(v)

	#define unit vector in the z-direction
	ez = np.array([0., 0., 1.])

	#the rotation axis is perpendicular to both matrices
	k = np.cross(v, ez)

	#compute the cosine of the required rotation angle
	c = np.dot(v, ez)

	#consider the case where v is already aligned with z
	if np.linalg.norm(k) < 1e-12:
		if c > 0: return np.eye(3)
		else:     return np.diag([1, 1, -1])

	#skew-symmetric matrix for Rodrigues' formula
	K = np.array([[    0, -k[2],  k[1]],
		      [ k[2],     0, -k[0]],
		      [-k[1],  k[0],     0]])

	#return rotation matrix from Rodrigues' formula
	return np.eye(3) + K + np.matmul(K, K)/(1 + c)

#compute dictionary of pyEFPE parameters from simulation paramters
def compute_pyEFPE_params_from_sim(sim_params, M=1, theta=0, phi=0, dL=10, eccentric=True, precessing=True):
	
	#extract the mass ratio
	q = sim_params['reference_mass_ratio']

	#compute the reference average orbital frequency
	reference_frequency = sim_params["reference_orbital_frequency"]
	reference_frequency_mag = np.linalg.norm(reference_frequency)
	avg_forb_ref = compute_favg_from_finst_0PN(reference_frequency_mag, sim_params['reference_eccentricity'], sim_params["reference_mean_anomaly"])

	#compute reference inclination as the angle between reference orbital frequency and viewing vector
	lN_vec = reference_frequency/reference_frequency_mag
	viewing_vec = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
	inclination = np.arccos(np.dot(viewing_vec, lN_vec))

	#compute rotation between inertial frame and L-frame, this would transform lN into z
	R_inertial_to_L = rotation_to_z(lN_vec)
	
	#compute spins in the L-frame
	s1_vec = np.dot(R_inertial_to_L, sim_params['reference_dimensionless_spin1'])
	s2_vec = np.dot(R_inertial_to_L, sim_params['reference_dimensionless_spin2'])
	
	#extract eccentricity and mean anomaly
	eccentricity = sim_params['reference_eccentricity']
	mean_anomaly = sim_params['reference_mean_anomaly']

	#When required, force parameters to be exactly quasi-circular and/or spin-aligned
	if not eccentric:
		if eccentricity>0.01: print("Warning: System is eccentric (ecc=%.4g) but, as requested, eccentricity will be set to 0."%(eccentricity))
		eccentricity, mean_anomaly = 0., 0.
	if not precessing:
		s1_perp, s2_perp = np.linalg.norm(s1_vec[:2]), np.linalg.norm(s2_vec[:2])
		if (s1_perp>1e-4) or (s2_perp>1e-4): print("Warning: System is precessing (s1_perp=%.4g, s2_perp=%.4g) but, as requested, perpendicular spins will be set to 0."%(s1_perp, s2_perp))
		s1_vec[:2], s2_vec[:2] = 0., 0.

	#return dictionary with pyEFPE parameters
	return {
	'mass1': q*M/(1+q),
	'mass2': M/(1+q),
	'eccentricity': eccentricity,
	'spin1x': s1_vec[0],
	'spin1y': s1_vec[1],
	'spin1z': s1_vec[2],
	'spin2x': s2_vec[0],
	'spin2y': s2_vec[1],
	'spin2z': s2_vec[2],
	'inclination': inclination,
	'f22_start': avg_forb_ref/(np.pi*M*t_sun_s),
	'phase': 0.,
	'mean_anomaly': mean_anomaly,
	'distance': dL,
	}

#function to choose the parameters to minimize over
def choose_minimize_parameters(eccentric=True, precessing=True, minimize_f22_start=True):

	#parameters to always minimize over
	minimize_parameters = ['phase']

	#add parameters to minimize over for eccentric and or precessing systems
	if precessing: minimize_parameters += ['phase_s']
	if eccentric:  minimize_parameters += ['eccentricity', 'mean_anomaly']
	#when eccentric or when required, minimize over f22_start
	if eccentric or minimize_f22_start: minimize_parameters += ['f22_start']

	return minimize_parameters

#function to compute monte carlo integral and it error
def integrate_with_MC(fs, V=1):
	
	#compute integral as the mean of f(x) times the volume
	integral = V*np.mean(fs)
	
	#compute error from sttandar deviation of f
	error = V*np.std(fs, ddof=1)/np.sqrt(len(fs))
	
	return integral, error

#function to compute the SNR weighted mismatch
def compute_avg_MM_SNR(mismatches, SNRs):
	
	#compute faithfulness
	Fs = 1 - mismatches
	
	#compute the average of (F*SNR)^3 and SNR^3
	num_integrand = (Fs*SNRs)**3
	den_integrand = SNRs**3
	num = np.mean(num_integrand)
	den = np.mean(den_integrand)
	#compute the average faithfullness
	F_avg_SNR = (num/den)**(1/3.)
	
	#Numerator and denominator are correlated, we compute error with covariance
	numden_cov = np.cov([num_integrand, den_integrand], ddof=1)
	num_var_normed = numden_cov[0,0]/(num*num)
	den_var_normed = numden_cov[1,1]/(den*den)
	numden_cov_normed = numden_cov[0,1]/(num*den)
	#compute the error on the faithfullness estimate
	F_avg_SNR_err = F_avg_SNR*np.sqrt((num_var_normed + den_var_normed - 2*numden_cov_normed)/len(Fs))/3.

	#return the average mismatch and its error
	return 1 - F_avg_SNR, F_avg_SNR_err

#######################################################################################

#configuration for the different approximants
config_dict = {
"pyEFPEHM":    {"eccentric":  True, "precessing":  True, "max_e": 0.85},
"pyEFPE":      {"eccentric":  True, "precessing":  True, "max_e": 0.85},
"TEOBResumS":  {"eccentric":  True, "precessing":  True, "max_e": 0.85},
"SEOBNRv5EHM": {"eccentric":  True, "precessing": False, "max_e": 0.70},
"SEOBNRv5PHM": {"eccentric": False, "precessing":  True, "max_e": 0.00}
}

#######################################################################################

#number of sxs simulation
sxs_num = '0088' #0088 2538 2549 2561 2619 2621 3951 4286 4290

#number of configurations (th, ph, pol) to consider
Nconf = 100
seed = 150914
distance = 100 #Mpc

#asd to consider in mismatches (set to None for no asd)
psd_name = 'AplusDesign' #None

#approximant to compare with NR
approx_string="pyEFPEHM"

#Relative wigle room to allow eccentricity
rtol_e = 0.4
#Relative wigle room to allow for general parameter
rtol_p = 0.1

#minimum frequency for the mismatch (in Hz)
minimum_frequency = 20
#fraction of fISCO that fmax represents
fmax_fISCO = 0.8

#maximum reference frequency to consider (in Hz)
max_ref_freq = 19

#parameters for numerical mismatch minimization
workers = 8
maxiter = 1000
popsize = 32

#output file for result
outdir = 'outdir/NR_comparisons/'

#name to identify this run
name_ID = '%s_vs_SXSBBH%s_Nconf_%s_fref_%.3g_fmin_%.3g_fmax_%.3gISCO_De_%.3g_Dp_%.3g_%s'%(approx_string, sxs_num, Nconf, max_ref_freq, minimum_frequency, fmax_fISCO, rtol_e, rtol_p, psd_name)

#ploting parameters
make_plots = True
plot_order = 'maximum mismatch'
flowHz_to_remove_memory = 2. # set to 0 to not remove memory
thinning_factor = 8 #to reduce the number of samples in time domain plots
t_end_after_t_peak_in_M = 100
t_break_b4_t_peak_in_M = 1000
color_NR   = 'C0'
color_EFPE = 'C1'
color_dphi = 'C2'
color_dA   = 'C4'

#######################################################################################

if __name__ == "__main__":

	#determine sxs id
	sxs_id = "SXS:BBH:%s"%(sxs_num)

	#load sxs simulation
	hraw_sxs, sim_params = load_sxs_sim(sxs_id)

	#compute the reference average orbital frequency
	avg_forb_ref = compute_favg_from_finst_0PN(sim_params["reference_orbital_frequency_mag"], sim_params['reference_eccentricity'], sim_params["reference_mean_anomaly"])

	#use simulation parameters and the requested reference frequency to compute the mass
	M = avg_forb_ref/(max_ref_freq*np.pi*t_sun_s)
	print('\nAnalyzing', sxs_id, 'vs', approx_string)
	print('M = %.3g Msun, Avg freq: %.4g Hz, Instantaneous freq: %.4g Hz'%(M, avg_forb_ref/(np.pi*M*t_sun_s), sim_params["reference_orbital_frequency_mag"]/(np.pi*M*t_sun_s)))

	#compute the maximum frequency to compare to
	maximum_frequency = fmax_fISCO*(6**-1.5)/(np.pi*M*t_sun_s)
	print('minimum_frequency: %.4g Hz, maximum_frequency: %.4g Hz'%(minimum_frequency, maximum_frequency))

	#compute and print more information about the simulation
	lN_vec = sim_params['reference_orbital_frequency']/np.linalg.norm(sim_params['reference_orbital_frequency'])
	s1z = np.dot(sim_params['reference_dimensionless_spin1'], lN_vec)
	s2z = np.dot(sim_params['reference_dimensionless_spin2'], lN_vec)
	s1p = np.linalg.norm(np.cross(sim_params['reference_dimensionless_spin1'], lN_vec))
	s2p = np.linalg.norm(np.cross(sim_params['reference_dimensionless_spin2'], lN_vec))
	q   = 1/sim_params['reference_mass_ratio']
	print('q=%.3g, chi_eff=%.3g, chi_p=%.3g, e=%.3g'%(q, (s1z + q*s2z)/(1 + q), np.maximum(s1p, (q*(4*q + 3)/(4 + 3*q))*s2p), sim_params['reference_eccentricity']))

	#compute parameters to minimize over
	eccentric, precessing, max_e = (config_dict[approx_string][key] for key in ['eccentric', 'precessing', 'max_e'])
	minimize_parameters = choose_minimize_parameters(eccentric=eccentric, precessing=precessing)
	print("Minimize parameters:", minimize_parameters)

	#try to load result dictionary
	try:
		with open(outdir+'/result_'+name_ID+'.pickle', 'rb') as handle: result = pickle.load(handle)
		
		#make sure it corresponds to the same type of file
		assert result['maxiter'] == maxiter
		assert result['popsize'] == popsize
		assert result['rtol_e'] == rtol_e
		assert result['rtol_p'] == rtol_p
		assert result['minimize_parameters'] == minimize_parameters
		assert result['seed'] == seed
		
	#otherwise create it
	except:

		#generate random realizations of theta, phi, psi
		rng_generator = np.random.default_rng(seed=seed)
		thetas = np.arccos(rng_generator.uniform(low=-1, high=1, size=Nconf))
		phis, pols = rng_generator.uniform(low=0, high=2*np.pi, size=(2,Nconf))

		#loop over configurations
		snr_NR = np.zeros(Nconf)
		min_mm, dt_shift_sh, ang_sh, amp_sh, pol_h = np.zeros(Nconf), np.zeros(Nconf), np.zeros(Nconf), np.zeros(Nconf), np.zeros(Nconf)
		min_pyEFPE_params = list()
		for ic, (theta, phi, pol) in enumerate(zip(thetas, phis, pols)):

			print('\n%s/%s: theta=%.3g, phi=%.3g, pol=%.3g'%(ic+1, Nconf, theta, phi, pol))
			
			#project signal and prepare it for fft
			ht_sim, t_sim = prepare_sxs_h(hraw_sxs, theta, phi, pol)

			#compute frequency domain strain in simulation units
			hfft_sim, freqs_sim = compute_hfft_freqs(ht_sim, t_sim, fmin=avg_forb_ref/np.pi)

			#convert it to physical units
			hfft_phys, freqs_phys = hfft_set_M_dL_BBH(hfft_sim, freqs_sim, M=M, dL=distance, fmin=minimum_frequency, fmax=maximum_frequency)

			#compute asd
			if type(psd_name)==str:
				delta_f = np.mean(np.diff(freqs_phys))
				asd = compute_asd(delta_f, freqs_phys[0], freqs_phys[-1] + delta_f, psd_name=psd_name)[:len(freqs_phys)]
			else: asd = None

			#compute the snr of this signal
			if asd is None: snr_NR[ic] = np.linalg.norm(hfft_phys)
			else:           snr_NR[ic] = np.linalg.norm(hfft_phys/asd)

			#compute the initial pyEFPE parameters
			pyEFPE_params = compute_pyEFPE_params_from_sim(sim_params, M=M, theta=theta, phi=phi, dL=distance, eccentric=eccentric, precessing=precessing)
			print("Initial pyEFPE parameters: ", pyEFPE_params)

			#compute the region where to minimize the mismatch
			parameter_bounds = determine_pyEFPE_bounds(minimize_parameters, pyEFPE_params, rtol_e=rtol_e, rtol_p=rtol_p, max_e=max_e)

			print("minimize_parameters:", minimize_parameters)
			print("parameter_bounds   :", parameter_bounds,'\n')

			#minimize mismatch
			min_mm[ic], dt_shift_sh[ic], ang_sh[ic], amp_sh[ic], pol_h[ic], params_min = numerically_minimize_mismatch(hfft_phys, freqs_phys, pyEFPE_params, minimize_parameters, approx_string=approx_string, rtol_e=rtol_e, rtol_p=rtol_p, max_e=max_e, asd=asd, workers=workers, maxiter=maxiter, popsize=popsize)

			print('mm=%s'%(min_mm[ic]))
			print('Minimum pyEFPE params:', {name: val for name, val in zip(minimize_parameters, params_min)})
			print('dt_shift: %.3g, ang_sh: %.3g, amp_sh: %.3g, pol_h: %.3g \n'%(dt_shift_sh[ic], ang_sh[ic], amp_sh[ic], pol_h[ic]))

			#compute best fit pyEFPE waveform
			min_pyEFPE_params.append(update_pyEFPE_params(pyEFPE_params, params_min, minimize_parameters))

		#compute the average mismatch and its error, with and without SNR weighting
		MM_avg, MM_avg_err = integrate_with_MC(min_mm)
		MM_SNR_avg, MM_SNR_avg_err = compute_avg_MM_SNR(min_mm, snr_NR)
		
		#put everything in a dictionary and save it
		result = {'sxs_num': sxs_num, 'minimum_frequency': minimum_frequency, 'max_ref_freq': max_ref_freq,
		          'fmax_fISCO': fmax_fISCO, 'maximum_frequency': maximum_frequency,
			  'minimize_parameters': minimize_parameters, 'rtol_e': rtol_e, 'rtol_p': rtol_p, 'maxiter': maxiter, 'popsize': popsize,
			  'seed': seed, 'thetas': thetas, 'phis': phis, 'pols': pols, 'M':M, 'avg_forb_ref': avg_forb_ref, 'asd': asd, 'freqs_phys': freqs_phys,
			  'snr_NR': snr_NR,'min_mm': min_mm, 'dt_shift_sh':dt_shift_sh, 'ang_sh':ang_sh,
			  'amp_sh':amp_sh, 'pol_h':pol_h, 'min_pyEFPE_params':min_pyEFPE_params,
			  'MM_avg': MM_avg, 'MM_avg_err': MM_avg_err, 'MM_SNR_avg': MM_SNR_avg, 'MM_SNR_avg_err': MM_SNR_avg_err,
			  }

		#save result dictionary
		if not os.path.exists(outdir): os.makedirs(outdir)
		with open(outdir+'/result_'+name_ID+'.pickle', 'wb') as handle: pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

	#######################################################################################

	#print the result for the average SNR
	print('MM_avg =',result['MM_avg'],'+-',result['MM_avg_err'])
	print('MM_snr =',result['MM_SNR_avg'],'+-', result['MM_SNR_avg_err'] )
	
	if make_plots:
	
		import matplotlib.pyplot as plt
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

		#make a histogram with the mismatches
		bins = np.geomspace(np.nanmin(result['min_mm']), np.nanmax(result['min_mm']), 25)
		plt.figure(figsize=(12,8))
		plt.hist(result['min_mm'], bins=bins)
		plt.xlabel(r'Mismatch')
		plt.ylabel(r'Number of Realizations')
		plt.xscale('log')
		plt.xlim(bins[0], bins[-1])
		plt.title(sxs_id+'\n'+'$\\overline{\\mathcal{MM}}_\\mathrm{avg} = %.4f \\pm %.4f$, $\\overline{\\mathcal{MM}}_\\mathrm{SNR} = %.4f \\pm %.4f$'%(result['MM_avg'], result['MM_avg_err'], result['MM_SNR_avg'], result['MM_SNR_avg_err']))
		plt.tight_layout()
		plt.show()


		#if required, sort mismatches to plot them
		if   plot_order == 'maximum mismatch': ic_plot = np.argsort(-result['min_mm'])
		elif plot_order == 'minimum mismatch': ic_plot = np.argsort( result['min_mm'])
		else:                                  ic_plot = np.arange(len(result['min_mm']))

		#plot the different waveforms corresponding to the above mismatches
		for ic in ic_plot:

			#reconstruct the NR waveform
			ht_sim, t_sim = prepare_sxs_h(hraw_sxs, result['thetas'][ic], result['phis'][ic], result['pols'][ic])
			hfft_sim, freqs_sim = compute_hfft_freqs(ht_sim, t_sim, fmin=result['avg_forb_ref']/np.pi)
			hfft_phys, freqs_phys = hfft_set_M_dL_BBH(hfft_sim, freqs_sim, M=result['M'], dL=distance, fmin=minimum_frequency, fmax=result['maximum_frequency'])

			#print information of this simulation
			print('\ntheta=%.3g, phi=%.3g, pol=%.3g'%(result['thetas'][ic], result['phis'][ic], result['pols'][ic]))

			#print the best fit pyEFPE parameters
			print("Best fit pyEFPE parameters: ", result['min_pyEFPE_params'][ic])

			#compute best fit pyEFPE waveform
			hp_pyEFPE, hc_pyEFPE = pyEFPE_like_generator(approx_string, result['freqs_phys'], result['min_pyEFPE_params'][ic])

			#Now apply all the factors to make it agree with NR waveform
			h_pyEFPE_minMM = (pol_response(hp_pyEFPE, hc_pyEFPE, result['pol_h'][ic])/result['amp_sh'][ic])*np.exp(-1j*(2*np.pi*result['freqs_phys']*result['dt_shift_sh'][ic] + result['ang_sh'][ic]))

			#Compute and print direct and saved mismatch
			mm_direct = 1 - np.real(np.sum(np.conj(normalize_h(hfft_phys, asd=result['asd'])[0])*normalize_h(h_pyEFPE_minMM, asd=result['asd'])[0]))
			print('Mismatch: Saved: %s  Direct: %s'%(result['min_mm'][ic], mm_direct))

			#create title string
			title_str = r'%s $\left(\overline{\mathcal{MM}} = %.3g \right)$'%(sxs_id, result['min_mm'][ic])

			#plot frequency domain waveform
			plot_h1_h2(hfft_phys, h_pyEFPE_minMM, freqs_phys, label_1='NR', label_2=approx_string, show=False, title_str=title_str, asd=result['asd'])

			#Initialize pyEFPE waveform generator
			if   approx_string=="pyEFPEHM": wf_minMM = pyEFPEHM.pyEFPE(result['min_pyEFPE_params'][ic])
			elif approx_string=="pyEFPE"  : wf_minMM =   pyEFPE.pyEFPE(to_legacy_pyEFPE_params(result['min_pyEFPE_params'][ic]))
			#if the time-domain computation of the waveform is not implemented, show plots and continue
			else:
				print("Time domain plots for %s not implemented"%(approx_string))
				plt.show()
				continue

			#reconstruct time domain NR waveform
			h_NR_td_phys = (result['M']*t_sun_s/(distance*Mpc_s))*np.array(ht_sim[::thinning_factor])
			t_NR_phys    = result['M']*t_sun_s*t_sim[::thinning_factor]

			#compute time-domain pyEFPE waveform (compute it setting t=0 at the extreme, consistently with what was assumed after FFT)
			hp_pyEFPE_td, hc_pyEFPE_td = wf_minMM.generate_tdomain_waveform(t_NR_phys - t_NR_phys[-1])
			#compute frequency domain waveform to apply the phase corrections
			hp_pyEFPE_fft, hc_pyEFPE_fft = rfft(hp_pyEFPE_td), rfft(hc_pyEFPE_td)
			#compute correction due to overal phase and time shift
			freqs_fft = np.arange(len(hp_pyEFPE_fft))/(np.mean(np.diff(t_NR_phys))*len(t_NR_phys))
			h_pyEFPE_fft = (pol_response(hp_pyEFPE_fft, hc_pyEFPE_fft, result['pol_h'][ic])/result['amp_sh'][ic])*np.exp(-1j*(2*np.pi*freqs_fft*result['dt_shift_sh'][ic] + result['ang_sh'][ic]))
			#re-compute time domain waveform
			h_pyEFPE_td = irfft(h_pyEFPE_fft, n=len(t_NR_phys))

			#if requested, filter off memory
			if flowHz_to_remove_memory>0:
				imax_low_freq = int(flowHz_to_remove_memory*np.mean(np.diff(t_NR_phys))*len(t_NR_phys))
				h_NR_td_phys_fft = rfft(h_NR_td_phys)
				h_NR_td_phys_fft[:imax_low_freq] = 0
				h_NR_td_phys = irfft(h_NR_td_phys_fft, n=len(t_NR_phys))
				h_pyEFPE_td_fft = rfft(h_pyEFPE_td)
				h_pyEFPE_td_fft[:imax_low_freq] = 0
				h_pyEFPE_td = irfft(h_pyEFPE_td_fft, n=len(t_NR_phys))

			#find where the pyEFPE waveform starts and ends
			t0_pyEFPE = wf_minMM.sol.all_ts[0] + t_NR_phys[-1] + result['dt_shift_sh'][ic]
			tf_pyEFPE = wf_minMM.sol.all_ts[-1] + t_NR_phys[-1] + result['dt_shift_sh'][ic]
			#time to start and end plot from
			t0_plot   = max(t0_pyEFPE, 0)
			tf_plot   = result['M']*t_sun_s*min(hraw_sxs.max_norm_time()+t_end_after_t_peak_in_M, hraw_sxs.t[-1])
			#Select plotting times for NR
			iplot_NR = (t_NR_phys > t0_plot) & (t_NR_phys < tf_plot)
			t_plot_NR = t_NR_phys[iplot_NR]
			#select plotting times for pyEFPE
			iplot_pyEFPE = (t_NR_phys > t0_plot) & (t_NR_phys < tf_pyEFPE)
			t_plot_pyEFPE = t_NR_phys[iplot_pyEFPE]

			#select points to plot
			h_NR_td_plot = h_NR_td_phys[iplot_NR]
			h_pyEFPE_td_plot = h_pyEFPE_td[iplot_pyEFPE]

			#compute Amplitude and phase differences
			dA, dphi = compute_td_dA_dphi_wavelets(h_pyEFPE_td, h_NR_td_phys, t_NR_phys[1]-t_NR_phys[0], minimum_frequency, maximum_frequency)
			dA_plot, dphi_plot = dA[iplot_pyEFPE], dphi[iplot_pyEFPE]/(2*np.pi)
			
			#substract required constant number of cycles to have a median that it closer to 0
			dphi_plot = dphi_plot - np.round(np.median(dphi_plot))

			#compute where we want to break the waveform plotting
			t_break = result['M']*t_sun_s*(hraw_sxs.max_norm_time() - t_break_b4_t_peak_in_M)

			#y-limits for plots
			h_ylim_abs = 1.05*np.amax(np.abs(h_NR_td_plot))
			i_border_start = int(2/(np.mean(np.diff(t_plot_pyEFPE))*minimum_frequency))
			i_border_end   = int(2/(np.mean(np.diff(t_plot_pyEFPE))*maximum_frequency))
			dphi_ylim_abs  = 1.05*np.amax(np.abs(dphi_plot[i_border_start:-1-i_border_end]))
			dA_ylim_abs    = 1.05*np.amax(np.abs(  dA_plot[i_border_start:-1-i_border_end]))

			#plot with break to zoom into merger ringdown
			fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(24,5), gridspec_kw={'width_ratios': [3, 1], 'height_ratios': [2, 1]})
			#common things for rows
			axs2 = [None, None]
			for icol in range(2):
				#plots of strain
				axs[0,icol].plot(t_plot_NR    , h_NR_td_plot    , alpha=0.6, label=r"$\mathrm{NR}$", color=color_NR)
				axs[0,icol].plot(t_plot_pyEFPE, h_pyEFPE_td_plot, alpha=0.6, label=r"$\mathtt{%s}$"%(approx_string), color=color_EFPE)
				axs[0,icol].grid(True)
				axs[0,icol].set_ylim(-h_ylim_abs, h_ylim_abs)
				axs[0,icol].tick_params(axis='x', labelbottom=False)
				#plot phase difference
				axs[1,icol].plot(t_plot_pyEFPE, dphi_plot, color=color_dphi)
				axs[1,icol].set_ylim(-dphi_ylim_abs, dphi_ylim_abs)
				axs[1,icol].grid(True, axis='x')
				axs[1,icol].axhline(y=0, color='k')
				#plot amplitude difference in a twin axis
				axs2[icol] = axs[1,icol].twinx()
				axs2[icol].plot(t_plot_pyEFPE, dA_plot, color=color_dA)
				axs2[icol].set_ylim(-dA_ylim_abs, dA_ylim_abs)
				axs2[icol].axhline(y=0, color='k')
			#common things for columns
			for irow in range(2):
				axs[irow,0].set_xlim(t_plot_NR[0], t_break)
				axs[irow,1].set_xlim(t_break, t_plot_NR[-1])
				axs[irow,1].set_yticklabels([])
			axs2[0].set_yticklabels([])
			#labels
			axs[0,0].set_ylabel(r"Strain $h$")
			axs[0,0].legend(ncols=2, loc='upper left')
			axs[1,0].set_ylabel(r"$\frac{\phi_\mathtt{%s} - \phi_\mathrm{NR}}{2 \pi}$"%(approx_string), color=color_dphi)
			axs[1,0].tick_params(axis='y', which='both', colors=color_dphi)
			axs[1,1].set_yticks([])
			axs2[1].set_ylabel(r"$\frac{A_\mathtt{%s} - A_\mathrm{NR}}{A_\mathrm{NR}}$"%(approx_string), color=color_dA)
			axs2[0].set_yticks([])
			axs2[1].tick_params(axis='y', which='both', colors=color_dA)
			axs2[0].spines['left'].set_color(color_dphi)
			axs2[1].spines['right'].set_color(color_dA)
			#remove spaces between subplots
			plt.tight_layout()
			plt.subplots_adjust(wspace=0, hspace=0)
			#put a common x-label
			fig.text(0.5, 0.97, title_str, ha='center', va='top'   , fontsize=plt.rcParams['axes.labelsize'])
			fig.text(0.5, 0,  r"Time [s]", ha='center', va='bottom', fontsize=plt.rcParams['axes.labelsize'])

			#save maximum mismatch waveform
			if (ic==ic_plot[0]) and (plot_order == 'maximum mismatch'):
				if not os.path.exists(outdir+'/Plots/'): os.makedirs(outdir+'/Plots/')
				plt.savefig(outdir+'/Plots/sxs_bbh_%s_vs_%s_max_MM_td_wf.pdf'%(sxs_num, approx_string))

			plt.show()


