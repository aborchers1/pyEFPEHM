import numpy as np
from pyEFPEHM.waveform.functions import pyEFPE_PN_derivatives

#define the samples we want to test
test_samples = []
test_samples.append({'m1': 1.58, 'm2': 0.975, 'chi_eff': 0.76, 's2_1': 0.916, 's2_2': 0.541, 'q1': 1.78, 'q2': 0.567, 'o1':2.475, 'o2':-1.514, 'y': 0.89, 'e2': 0.78, 'dchi2': 0.81, 'dchi': 0.9, 'sperp2': 0.38})
test_samples.append({'m1': 1.58, 'm2': 0.975, 'chi_eff': 0, 's2_1': 0, 's2_2': 0, 'q1': 1.78, 'q2': 0.567, 'o1':2.475, 'o2':-1.514, 'y': 0.89, 'e2': 0.78, 'dchi2': 0, 'dchi': 0, 'sperp2': 0})
test_samples.append({'m1': 1.58, 'm2': 0.457, 'chi_eff': 0.86, 's2_1': 0.916, 's2_2': 0.541, 'q1': 1.78, 'q2': 0.567, 'o1':2.475, 'o2':-1.514, 'y': 0.99, 'e2': 0.97, 'dchi2': 0.95, 'dchi': -0.9, 'sperp2': 0.86})
test_samples.append({'m1': 1.58, 'm2': 0.457, 'chi_eff': 0.86, 's2_1': 0.916, 's2_2': 0.541, 'q1': 1.78, 'q2': 0.567, 'o1':2.475, 'o2':-1.514, 'y': 0.2, 'e2': 0.97, 'dchi2': 0.87, 'dchi': -0.8, 'sperp2': 0.86})

#compute the PN derivatives for the different samples
for p in test_samples:
	
	#print the parameters
	print(p)
	
	#initialize class
	PN_derivatives = pyEFPE_PN_derivatives(p['m1'], p['m2'], p['s2_1'], p['s2_2'], q1=p['q1'], q2=p['q2'], o1=p['o1'], o2=p['o2'])
	
	print(PN_derivatives.Dy_De2_Dl_Ddl(p['y'], p['e2'], p['chi_eff'], p['dchi'], p['dchi2'], p['sperp2']))
	print()

#samples with tidal effects
tidal_samples = []
tidal_samples.append({'m1': 1.58, 'm2': 0.975, 'chi_eff': 0.76, 's2_1': 0.916, 's2_2': 0.541, 'q1': 1.78, 'q2': 0.567, 'o1':2.475, 'o2':-1.514, 'y': 0.89, 'e2': 0.78, 'dchi2': 0.81, 'dchi': 0.9, 'sperp2': 0.38, 'Lambda2_1': 58.64, 'Lambda2_2': 64.78, 'Lambda3_1': 80.84, 'Lambda3_2': 27.43, 'Sigma2_1': 47.13, 'Sigma2_2': 38.75, 'Lambda23_1': 47.30, 'Lambda23_2': 94.37, 'Lambda32_1': 51.45, 'Lambda32_2': 36.44, 'Sigma23_1': 64.18, 'Sigma23_2': 70.01, 'Sigma32_1':  6.59, 'Sigma32_2': 62.19})
tidal_samples.append({'m1': 1.58, 'm2': 1.457, 'chi_eff': 0.86, 's2_1': 0.916, 's2_2': 0.541, 'q1': 1.78, 'q2': 0.567, 'o1':2.475, 'o2':-1.514, 'y': 0.99, 'e2': 0.97, 'dchi2': 0.95, 'dchi': -0.9, 'sperp2': 0.86, 'Lambda2_1': 58.64, 'Lambda2_2': 64.78, 'Lambda3_1': 80.84, 'Lambda3_2': 27.43, 'Sigma2_1': 47.13, 'Sigma2_2': 38.75, 'Lambda23_1': 47.30, 'Lambda23_2': 94.37, 'Lambda32_1': 51.45, 'Lambda32_2': 36.44, 'Sigma23_1': 64.18, 'Sigma23_2': 70.01, 'Sigma32_1':  6.59, 'Sigma32_2': 62.19})

#compute the PN derivatives for the different samples
for p in tidal_samples:
	
	#print the parameters
	print(p)
	
	#initialize class
	PN_derivatives = pyEFPE_PN_derivatives(p['m1'], p['m2'], p['s2_1'], p['s2_2'], q1=p['q1'], q2=p['q2'], o1=p['o1'], o2=p['o2'],
	Lambda2_1=p['Lambda2_1'], Lambda2_2=p['Lambda2_2'], Lambda3_1=p['Lambda3_1'], Lambda3_2=p['Lambda3_2'], Sigma2_1=p['Sigma2_1'], Sigma2_2=p['Sigma2_2'],
	Lambda23_1=p['Lambda23_1'], Lambda23_2=p['Lambda23_2'], Lambda32_1=p['Lambda32_1'], Lambda32_2=p['Lambda32_2'], Sigma23_1=p['Sigma23_1'], Sigma23_2=p['Sigma23_2'], Sigma32_1=p['Sigma32_1'], Sigma32_2=p['Sigma32_2']
	, pn_tidal_order=15)
	
	print(PN_derivatives.Dy_De2_Dl_Ddl(p['y'], p['e2'], p['chi_eff'], p['dchi'], p['dchi2'], p['sperp2']))
	print()

