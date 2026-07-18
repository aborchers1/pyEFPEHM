import numpy as np
try:
	from lal.lal import MSUN_SI
	from bilby.gw.conversion import bilby_to_lalsimulation_spins
	from bilby.core.utils import logger
except:
	pass

#function to generate EFPE waveforms
from pyEFPEHM.waveform.EFPE import pyEFPE

# Function to compute EFPE parameter dictionary from bilby inputs
def EFPE_params_from_bilby(mass_1, mass_2, luminosity_distance, a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, eccentricity, mean_anomaly, default_f22_start=20, **kwargs):

	#put some default parameters to call pyEFPE that will be used in this function
	params = {'mass1': mass_1,
	          'mass2': mass_2,
	          'e_start': eccentricity,
	          'distance': luminosity_distance,
	          'f22_start': default_f22_start,
	          'phi_start': phase,
	          'mean_anomaly_start': mean_anomaly,
	         }

	#update them with input kwargs
	params.update(kwargs)

	#convert spins from spherical to cartesian
	iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = bilby_to_lalsimulation_spins(theta_jn=theta_jn, phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12, a_1=a_1, a_2=a_2, mass_1=mass_1*MSUN_SI, mass_2=mass_2*MSUN_SI, reference_frequency=params['f22_start'], phase=phase)

	#now put all missing parameters in the parameter dictionary
	params['spin1x'] = spin_1x
	params['spin1y'] = spin_1y
	params['spin1z'] = spin_1z
	params['spin2x'] = spin_2x
	params['spin2y'] = spin_2y
	params['spin2z'] = spin_2z
	params['inclination'] = iota

	#return parameter dictionary
	return params


# Function to strip bilby-internal kwargs out of the parameter kwargs and resolve the frequency array
def _resolve_bilby_kwargs(frequency_array, kwargs):

	#pop bilby-internal keys so they do not leak into the pyEFPE parameter dictionary
	frequencies = kwargs.pop('frequencies', None)
	kwargs.pop('reference_frequency', None)
	kwargs.pop('waveform_approximant', None)
	kwargs.pop('minimum_frequency', None)
	kwargs.pop('maximum_frequency', None)
	catch_waveform_errors = kwargs.pop('catch_waveform_errors', False)

	#Bilby sometimes passes frequencies in kwargs and these take precedent over frequency_array
	if frequencies is not None: frequency_array = frequencies

	return frequency_array, catch_waveform_errors


# Function to handle a waveform-generation exception following bilby conventions
def _handle_waveform_error(e, params, catch_waveform_errors):

	#if we were not asked to catch waveform errors, re-raise
	if not catch_waveform_errors:
		raise e

	failed_parameters = dict(mass1=params['mass1'], mass2=params['mass2'],
	                         spin_1=(params['spin1x'], params['spin1y'], params['spin1z']),
	                         spin_2=(params['spin2x'], params['spin2y'], params['spin2z']),
	                         distance=params['distance'], iota=params['inclination'],
	                         eccentricity=params['e_start'], start_frequency=params['f22_start'])
	logger.warning("Evaluating the waveform failed with error: {}\n".format(e) +
	               "The parameters were {}\n".format(failed_parameters) +
	               "Likelihood will be set to -inf.")
	return None


# Function to compute EFPE waveforms for bilby
def EFPE_binary_black_hole(frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, eccentricity, mean_anomaly, **kwargs):

	#strip bilby-internal kwargs and resolve the frequency array
	frequency_array, catch_waveform_errors = _resolve_bilby_kwargs(frequency_array, kwargs)

	#compute EFPE parameter dictionary from bilby inputs
	params = EFPE_params_from_bilby(mass_1, mass_2, luminosity_distance, a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, eccentricity, mean_anomaly, **kwargs)

	#try to compute the waveform
	try:
		#compute the polarizations
		hp, hc = pyEFPE(params).generate_waveform(frequency_array)

		#return them in a dictionary with standard bilby convention
		return dict(plus=hp, cross=hc)
	#handle the case where the waveform raises an exception
	except Exception as e:
		return _handle_waveform_error(e, params, catch_waveform_errors)

