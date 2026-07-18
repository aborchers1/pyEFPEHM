import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

import pyEFPEHM

import numpy as np
from scipy.interpolate import CubicSpline

import time
start_runtime = time.time()

#universal constants
t_sun_s = 4.92549094831e-6  #GMsun/c**3 [s]

#function to compute the dephasing due to horizon absortion
def horizon_absorption_dphasing(p_pyEFPE, y=None, e2=0):
	
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
	
	#if no y is given, use final PN parameter
	if y is None:
		#maximum PN parameter is at ISCO
		y = 6**-0.5
		#compute it from f22_end if given in pyEFPE
		if 'f22_end' in p_pyEFPE:
			y = min((np.pi*M*p_pyEFPE['f22_end'])**(1/3), y)

	#return expected dephasing	
	return ((5./32.)**2)*(4./5.)*kHA*np.log(y/y0)/nu

################################################################

#initial eccentricities to consider
e20s = np.linspace(0., 0.8, 10)
y0 = 0.02
yf = 0.2

#parameters to assume for pyEFPE comparisson
base_pyEFPE_params = dict(
mass1 = 10., mass2 = 1.,
spin1x=-0.2, spin1y= 0.3, spin1z=0.9,
spin2x=-0.4, spin2y=-0.4, spin2z=0.8,
mean_anomaly=4.3, inclination=np.pi/3, phase=1.2, distance=100.
)

#things to add to the two waveforms being compared
params_pyEFPE_1 = {'horizon_absorption': True , 'Interpolate_Amplitudes': False}
params_pyEFPE_2 = {'horizon_absorption': False, 'Interpolate_Amplitudes': False}

#number of points for plots
Npoints = 10000

#label for analytical expression
analytical_label = r"$\frac{5}{256} \frac{\tilde{\kappa}_{H,0}}{\nu} \log{\frac{y}{y_0}}$"

#folder to save the plots in
plotdir = './Plots'

################################################################

#compute mass in seconds
M_s = (base_pyEFPE_params['mass1'] + base_pyEFPE_params['mass2'])*t_sun_s
base_pyEFPE_params['f22_end'] = (yf**3)/(np.pi*M_s)

#loop over eccentricities
Mfs, ys, Delta_l_EFPE, Delta_l_exp, Delta_l_EFPE_tphimin = [np.zeros((len(e20s), Npoints)) for _ in range(5)]
for ie, e20 in enumerate(e20s):

	#input the current eccentricity and initial frequency
	base_pyEFPE_params['eccentricity'] = e20**0.5
	base_pyEFPE_params['f22_start'] = ((np.sqrt(1 - e20)*y0)**3)/(np.pi*M_s)

	#update parameter dictionaries with base parameters
	params_pyEFPE_1.update(base_pyEFPE_params)
	params_pyEFPE_2.update(base_pyEFPE_params)
	
	#initialize waveforms
	wf1 = pyEFPEHM.pyEFPE(params_pyEFPE_1)
	wf2 = pyEFPEHM.pyEFPE(params_pyEFPE_2)

	#compute array of times for both waveforms both waveforms
	t1 = np.geomspace(wf1.sol.all_ts[0], wf1.sol.all_ts[-1], Npoints)
	t2 = np.geomspace(wf2.sol.all_ts[0], wf2.sol.all_ts[-1], Npoints)

	#compute ys and lambdas
	y1, e21, l1 = wf1.sol(t1, idxs=[0, 1, 2])
	y2, e22, l2 = wf2.sol(t2, idxs=[0, 1, 2])
	ys[ie] = y1

	#make interpolator of l2(y2)
	l2_spline = CubicSpline(y2, l2)

	#compute dephasing as a function of y
	Delta_l_EFPE[ie] = l1 - l2_spline(y1)

	#compute expected dephasing
	Delta_l_exp[ie] = horizon_absorption_dphasing(params_pyEFPE_1, y=y1, e2=e21)

	#compute dimensionless frequency
	Mfs[ie] = (np.sqrt(1 - e21)*y1)**3/np.pi

	#do a linear fit to \lambda(f)
	m, b = np.polyfit(Mfs[ie], Delta_l_EFPE[ie], 1)
	Delta_l_EFPE_tphimin[ie] = Delta_l_EFPE[ie] - (m*Mfs[ie] + b)

################################################################

from matplotlib import pyplot as plt
import matplotlib.ticker
import matplotlib.collections as mc
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

#Formatter to label minor ticks of log plot
def minor_log_formatter(x, pos, sub_ticks=[1., 2., 3., 5., 7.]):
	exponent = np.floor(np.log10(x))
	mantissa = x*(10**-exponent)
	if np.any(np.isclose(mantissa, sub_ticks)):
		return "%s"%(x)
	else:
		return ""

#make sure directory for plots exists
if not os.path.exists(plotdir): os.makedirs(plotdir)

#make a figure of dephasing as a function of frequency for the different eccentricities
segments_EFPE = [np.column_stack([yy, Delta_l]) for (yy, Delta_l) in zip(ys, Delta_l_EFPE)]
lc_EFPE = mc.LineCollection(segments_EFPE, array=e20s, cmap='viridis', label=r'$\mathtt{pyEFPEHM}$')
segments_exp = [np.column_stack([yy, Delta_l]) for (yy, Delta_l) in zip(ys, Delta_l_exp)]
lc_exp  = mc.LineCollection(segments_exp, color='k', ls='--', label=analytical_label)
Delta_l_diff = Delta_l_EFPE - Delta_l_exp
segments_diff = [np.column_stack([yy, Delta_l]) for (yy, Delta_l) in zip(ys, Delta_l_diff)]
lc_diff  = mc.LineCollection(segments_diff, array=e20s, cmap='viridis')
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True, height_ratios=[2,1], constrained_layout=True)
axs[0].add_collection(lc_EFPE)
axs[0].add_collection(lc_exp)
axs[0].set_ylim(Delta_l_EFPE.min(), Delta_l_EFPE.max())
axs[0].legend(loc='upper left')
axs[0].set_ylabel(r'$\Delta\lambda_H$')
axs[1].add_collection(lc_diff)
axs[1].set_ylim(Delta_l_diff.min(), Delta_l_diff.max())
axs[1].set_ylabel(r'$\Delta\lambda_H -$'+analytical_label)
axs[1].set_xlabel(r'$y$')
axs[1].set_xscale('log')
axs[1].xaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(minor_log_formatter))
axs[1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(minor_log_formatter))
axs[1].set_xlim(ys.min(), ys.max())
fig.colorbar(lc_EFPE, ax=axs, label=r'$e_0^2$')
fig.savefig(plotdir+'/DeltaLambda_EFPE_exp_of_y.pdf')

#make a figure of dephasing minimized over time and phase
segments_tphimin = [np.column_stack([Mf, Delta_l]) for (Mf, Delta_l) in zip(Mfs, Delta_l_EFPE_tphimin)]
lc_tphimin = mc.LineCollection(segments_tphimin, array=e20s, cmap='viridis', label=r'$\mathtt{pyEFPEHM}$')
fig, ax = plt.subplots(figsize=(10, 8))
ax.add_collection(lc_tphimin)
ax.set_xlim(Mfs.min(), Mfs.max())
ax.set_ylim(Delta_l_EFPE_tphimin.min(), Delta_l_EFPE_tphimin.max())
ax.set_xlabel(r'$M f$')
plt.ylabel(r'$\overline{\Delta\lambda}$')
ax.set_xscale('log')
ax.legend(loc='upper left')
fig.colorbar(lc_tphimin, ax=ax, label=r'$e_0^2$')
fig.tight_layout()
fig.savefig(plotdir+'/eff_DeltaLambda_EFPE_of_Mf.pdf')

plt.show()



