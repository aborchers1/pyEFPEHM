import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

import numpy as np
from matplotlib import pyplot as plt
from bilby.gw.conversion import bilby_to_lalsimulation_spins
from lalsimulation import SimInspiralTransformPrecessingWvf2PE
from lal.lal import MSUN_SI
import pyEFPEHM

import sys
mismatch_utils_path = "../../model_validation/"
sys.path.append(mismatch_utils_path)
from utils_compute_mismatches import *

# pybop downloaded from https://github.com/gmorras/pybop
from pybop import bayesian_minimization, plot_slices

#function to convert from lalsimulation to bilby initial conditions
def lalsimulation_to_bilby_spins(incl, chi1, chi2, mass1, mass2, f_ref, phi_ref):

	#note that the masses have to be in solar masses	
	return SimInspiralTransformPrecessingWvf2PE(incl, *chi1, *chi2, mass1, mass2, f_ref, phi_ref)

######################################################################

#lalsimulations spins to generate
chi1    = [-0.2, 0.3, 0.9]
chi2    = [-0.4, -0.4, 0.8]
incl    = np.pi/3
mass1   = 14.
mass2   = 1.4
f_ref   = 20.
phi_ref = 1.2

#rest of parameters
dL = 500.0
phase = 1.2
eccentricity = 0.3
mean_anomaly = 4.3

#frequencies
fmin = 20
fmax = 270
duration = 32

#psd to use for mismatches
psd_name = 'AplusDesign'

#Settings to use in bayesian optimization
n_initial = 32
n_calls   = 100
verbose   = False

######################################################################

#convert spins from lalsimulation to bilby
theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2 = lalsimulation_to_bilby_spins(incl, chi1, chi2, mass1, mass2, f_ref, phi_ref)

print('\nBilby initial conditions:')
print(dict(theta_jn=theta_jn, phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12, a_1=a_1, a_2=a_2))

#convert from bilby to lalsimulation again
iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = bilby_to_lalsimulation_spins(theta_jn=theta_jn, phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12, a_1=a_1, a_2=a_2, mass_1=mass1*MSUN_SI, mass_2=mass2*MSUN_SI, reference_frequency=f_ref, phase=phi_ref)

print('\nReconstructed LAL initial conditions:')
print('iota  =', iota)
print('spin1 =', [spin_1x, spin_1y, spin_1z])
print('spin2 =', [spin_2x, spin_2y, spin_2z])

######################################################################

#generate waveform with and without horizon absorption
waveform_dict_1 = {'f22_start': f_ref, 'horizon_absorption': True}
waveform_dict_2 = {'f22_start': f_ref, 'horizon_absorption': False}

#frequency array
freqs = np.arange(fmin, fmax, 1./duration)

#compute referece waveform polarization with bilby generator
hpols_1 = pyEFPEHM.waveform_generator.EFPE_binary_black_hole(freqs, mass1, mass2, dL, a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, eccentricity, mean_anomaly, **waveform_dict_1)
hpols_2 = pyEFPEHM.waveform_generator.EFPE_binary_black_hole(freqs, mass1, mass2, dL, a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, eccentricity, mean_anomaly, **waveform_dict_2)
#compute ASD
asd = compute_asd(1./duration, fmin, fmax, psd_name=psd_name, asd_folder=mismatch_utils_path+'/ASDs')

#compute mismatches
print('\nPolarization mismatches:')
for key, h_1 in hpols_1.items():

	# Compute the overlap
	MM_amp_min = mismatch_amp_minimized(hpols_1[key], hpols_2[key], asd=asd)

	print('\n'+key+' polarization:')
	print('MM without minimization:', MM_amp_min)

	# Function to compute mismatch varying phi0
	def mismatch_varying_phi0(angs):
		phi0 = angs[0]
		#compute second pyEFPE polarizations
		hphc_2 = pyEFPEHM.waveform_generator.EFPE_binary_black_hole(freqs, mass1, mass2, dL, a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phi0, eccentricity, mean_anomaly, **waveform_dict_2)
		#return the mismatch
		return mismatch_t_ph_amp_minimized(h_1, hphc_2[key], freqs, asd=asd)[0]

	result_phi = bayesian_minimization(
	mismatch_varying_phi0, [(-np.pi, np.pi)],
	periodic_dims=np.arange(1), periods=[2*np.pi],
	n_initial=n_initial, n_calls=n_calls, verbose=verbose,
	)

	MM_t_ph_amp_min = result_phi['y_best']

	print('MM with time and phase minimization:', MM_t_ph_amp_min)

	#function to compute mismatch varying phi0, theta_jl and mean_anomaly
	def mismatch_varying_angs(angs):
		phi0, phiJL0, ell0 = angs
		#compute second pyEFPE polarizations
		hphc_2 = pyEFPEHM.waveform_generator.EFPE_binary_black_hole(freqs, mass1, mass2, dL, a_1, tilt_1, phi_12, a_2, tilt_2, phiJL0, theta_jn, phi0, eccentricity, ell0, **waveform_dict_2)
		#return the mismatch
		return mismatch_t_ph_amp_pol_minimized(h_1, hphc_2['plus'], hphc_2['cross'], freqs, asd=asd)

	#minimize it using bayesian optimization
	bounds, periodic_dims, periods = [(-np.pi, np.pi)]*3, np.arange(3), [2*np.pi]*3
	result = bayesian_minimization(
	mismatch_varying_angs, bounds,
	periodic_dims=periodic_dims, periods=periods,
	n_initial=n_initial, n_calls=n_calls, verbose=verbose,
	)

	print('MM with full minimization:', result['y_best'])
	print('phi0=%.4g, phi_jl=%.4g, ell0=%.4g'%tuple(result['x_best']))

	#plot result of bayesian minimization
	plot_slices(result, bounds, objective=mismatch_varying_angs, n_grid=100)

plt.show()

