import numpy as np
try:
	from lal.lal import MSUN_SI
	from bilby.gw.conversion import bilby_to_lalsimulation_spins
except:
	pass

#function to generate EFPE waveforms
from pyEFPEHM.waveform.EFPE import pyEFPE

def EFPE_binary_black_hole(frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, eccentricity, mean_anomaly, **kwargs):

	#put some default parameters to call pyEFPE that will be used in this function
	params = {'mass1': mass_1,
	          'mass2': mass_2,
	          'e_start': eccentricity,
	          'distance': luminosity_distance,
	          'f22_start': 20,
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

	#Bilby sometimes passes frequencies in kwargs and these take precedent over frequency_array
	if 'frequencies' in kwargs: frequency_array = kwargs['frequencies']
	
	#compute the polarizations
	hp, hc = pyEFPE(params).generate_waveform(frequency_array)
	
	#return them in a dictionary with standard bilby convention
	return dict(plus=hp, cross=hc)
