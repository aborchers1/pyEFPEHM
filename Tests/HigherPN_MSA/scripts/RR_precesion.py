import numpy as np
import time
from precesion_functions import *

##############################################################################################

#Inputs for random parameters
q_min = 0.1
q_max = 1
a_min = 0
a_max = 1
e0_min = 0
e0_max = 0.5

#pn spin orders to loop over
pn_spin_order_min = 4
pn_spin_order_max = 7

#simulation PN parameters
y0 = 0.05
yf = 0.4
Ny = int(1e5)

#orientation of initial angular momentum
lN_0 = np.array([0,0,1.])
pn_phase_order = 6

#For debugging purposes we can pass pyEFPE params
pyEFPE_params = None

##############################################################################################

#if pyEFPE params were not passed, generate random params
if pyEFPE_params is None:
	#generate random parameters
	q = np.random.uniform(low=q_min, high=q_max)
	e0 = np.random.uniform(low=e0_min, high=e0_max)
	#generate random angular momenta
	chi_1_0, chi_2_0 = np.random.uniform(low=[-1,-1,-1], high=[1,1,1], size=(2,3))
	a_1, a_2 = np.random.uniform(low=a_min, high=a_max, size=2)
	chi_1_0, chi_2_0 = a_1*chi_1_0/np.linalg.norm(chi_1_0), a_2*chi_2_0/np.linalg.norm(chi_2_0)
else:
	#extract initial eccentricity
	e0 = pyEFPE_params['e_start']
	#compute initial PN parameter
	y0 = ((np.pi*pyEFPE_params['f22_start']*(pyEFPE_params['mass1'] + pyEFPE_params['mass2'])*t_sun_s)**(1./3.))/np.sqrt(1 - e0*e0)
	#compute mass ratio
	q = pyEFPE_params['mass2']/pyEFPE_params['mass1']
	#compute dimensionless spins
	chi_1_0 = np.array([pyEFPE_params['spin1x'], pyEFPE_params['spin1y'], pyEFPE_params['spin1z']])
	chi_2_0 = np.array([pyEFPE_params['spin2x'], pyEFPE_params['spin2y'], pyEFPE_params['spin2z']])

print("y0 = %.3g, q = %.3g, e0 = %.3g"%(y0, q, e0))
print("chi_1_0 =", chi_1_0)
print("chi_2_0 =", chi_2_0)
print("lN_0  =", lN_0)

#compute squared eccentricity
e20 = e0*e0

#compute nu and dmu from mass ratio
dmu = (1 - q)/(1 + q)
nu  = q/((1+q)**2)

#go from dimensionless spins to dimensionfull ones
s_1_0, s_2_0 = 0.5*(1 + dmu)*chi_1_0 , 0.5*(1 - dmu)*chi_2_0
a_1, a_2 = np.linalg.norm(s_1_0), np.linalg.norm(s_2_0)

#array of ys to simulate
ys = np.geomspace(y0, yf, Ny)

#loop over pn_spin_orders
save_keys = ['s1s', 's2s', 'lNs', 's2_1', 's2_2', 'lN2', 'chi_eff', 'dchi', 'J', 'JN', 'e2', 'lamb', 'dlamb', 'phiz', 'zeta', 'cth', 'jNjN0', 'jj0', 'DJ2']
result = {key: {} for key in save_keys}
result['pyEFPE'] = {key: {} for key in save_keys}
pn_spin_orders = np.arange(pn_spin_order_min, pn_spin_order_max+1)
for pn_spin_order in pn_spin_orders:
	
	#evolve PN equations
	soltime = time.time()
	ivp_sol = evolve_precessing_system_with_RR(y0, yf, e20, s_1_0, s_2_0, lN_0, nu, dmu, pn_spin_order=pn_spin_order, pn_phase_order=pn_phase_order)
	soltime = time.time() - soltime
	print("Time to evolve spins for pn_spin_order=%s: %.3g s"%(pn_spin_order, soltime))
	
	#evaluate solution
	sol = ivp_sol.sol(ys)
	s1s = sol[ :3,:]
	s2s = sol[3:6,:]
	lNs = sol[6:9,:]
	result['e2'][pn_spin_order]    = sol[9]
	result['lamb'][pn_spin_order]  = sol[10]
	result['dlamb'][pn_spin_order] = sol[11]
	result['phiz'][pn_spin_order]  = sol[12]
	result['zeta'][pn_spin_order]  = sol[13]

	#save values in result dictionary
	result['s1s'][pn_spin_order] = s1s
	result['s2s'][pn_spin_order] = s2s
	result['lNs'][pn_spin_order] = lNs

	#compute the different scalars
	result['s2_1'][pn_spin_order] = np.sum(s1s*s1s, axis=0)
	result['s2_2'][pn_spin_order] = np.sum(s2s*s2s, axis=0)
	result['lN2'][pn_spin_order]  = np.sum(lNs*lNs, axis=0)
	result['chi_eff'][pn_spin_order]  = np.sum((s1s + s2s)*lNs, axis=0)
	result['dchi'][pn_spin_order]     = np.sum((s1s - s2s)*lNs, axis=0)
	
	#compute total angular momentum
	result['J'][pn_spin_order]  = compute_J(s1s, s2s, lNs, ys, nu, dmu, pn_spin_order=pn_spin_order)

	#compute newtonian total angular momentum
	result['JN'][pn_spin_order] = (nu/ys[None,:])*lNs + 0.5*(1 + dmu)*s1s + 0.5*(1 - dmu)*s2s
	JN2 = np.sum(np.square(result['JN'][pn_spin_order]), axis=0)

	#compute cosine of the angle between and initial and evolved newtonian total angular momentum
	jNs = result['JN'][pn_spin_order]/np.sqrt(JN2)
	result['jNjN0'][pn_spin_order] = np.sum(jNs[:,[0]]*jNs, axis=0)

	#compute cosine of the angle between and initial and evolved PN total angular momentum
	J2 = np.sum(np.square(result['J'][pn_spin_order]), axis=0)
	js = result['J'][pn_spin_order]/np.sqrt(J2)
	result['jj0'][pn_spin_order] = np.sum(js[:,[0]]*js, axis=0)

	#compute cosine of the angle between lN and total angular momentum
	result['cth'][pn_spin_order] = np.sum(js*lNs, axis=0)

	#compute quantities with pyEFPE
	wf_pyEFPE, result['pyEFPE']['e2'][pn_spin_order], result['pyEFPE']['phiz'][pn_spin_order], result['pyEFPE']['zeta'][pn_spin_order], result['pyEFPE']['cth'][pn_spin_order], result['pyEFPE']['dchi'][pn_spin_order], result['pyEFPE']['chi_eff'][pn_spin_order], result['pyEFPE']['DJ2'][pn_spin_order] = compute_quantities_with_pyEFPE(ys, e0, s_1_0, s_2_0, lN_0, nu, dmu, pn_spin_order=pn_spin_order, pn_phase_order=pn_phase_order)
	
	#compute DJ2 as described in Eq.(51) of 2502.03929
	result['DJ2'][pn_spin_order] = compute_DJ2_exact_wth_pyEFPE(ys, J2, wf_pyEFPE)

##############################################################################################

from matplotlib import pyplot as plt
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

from matplotlib.collections import LineCollection

#plot things that should be constant
fig, ax = plt.subplots(figsize=(16,10))
s2_1_segments = LineCollection([np.column_stack([ys, np.sqrt(result['s2_1'][pn_spin_order]) - a_1]) for pn_spin_order in pn_spin_orders], array=pn_spin_orders, ls='-', label=r'$s_1 - s_{1,0}$')
s2_2_segments = LineCollection([np.column_stack([ys, np.sqrt(result['s2_2'][pn_spin_order]) - a_2]) for pn_spin_order in pn_spin_orders], array=pn_spin_orders, ls='--', label=r'$s_2 - s_{2,0}$')
lN2_segments  = LineCollection([np.column_stack([ys, np.sqrt(result['lN2'][pn_spin_order])  - 1  ]) for pn_spin_order in pn_spin_orders], array=pn_spin_orders, ls=':', label=r'$l_N - 1$')
ax.add_collection(s2_1_segments)
ax.add_collection(s2_2_segments)
ax.add_collection(lN2_segments)
fig.colorbar(s2_1_segments, label='pn_spin_orders', ax=ax)
ax.set_xlabel(r'$y$')
ax.set_xscale('log')
ax.autoscale()
ax.set_xlim(y0, yf)
ax.legend()
plt.tight_layout()

#plot exact dchi and the one predicted by pyEFPE
dchi0 = np.dot(lN_0, s_1_0 - s_2_0)
fig, ax = plt.subplots(figsize=(16,10))
dchi_segments = LineCollection([np.column_stack([ys, result['dchi'][pn_spin_order] - dchi0]) for pn_spin_order in pn_spin_orders], array=pn_spin_orders, ls='-', label=r'exact')
pyEFPE_dchi_segments = LineCollection([np.column_stack([ys, result['pyEFPE']['dchi'][pn_spin_order] - dchi0]) for pn_spin_order in pn_spin_orders], array=pn_spin_orders, ls=':', label=r'pyEFPE')
ax.add_collection(dchi_segments)
ax.add_collection(pyEFPE_dchi_segments)
fig.colorbar(dchi_segments, label='pn_spin_orders', ax=ax)
ax.set_xlabel(r'$y$')
ax.set_ylabel(r'$\delta\chi - \delta\chi_0$')
ax.set_xscale('log')
ax.autoscale()
ax.set_xlim(y0, yf)
ax.legend()
plt.tight_layout()

#plot exact chi_eff and the one predicted by pyEFPE
chi_eff0 = np.dot(lN_0, s_1_0 + s_2_0)
fig, ax = plt.subplots(figsize=(16,10))
chi_eff_segments = LineCollection([np.column_stack([ys, result['chi_eff'][pn_spin_order] - chi_eff0]) for pn_spin_order in pn_spin_orders], array=pn_spin_orders, ls='-', label=r'exact')
pyEFPE_chi_eff_segments = LineCollection([np.column_stack([ys, result['pyEFPE']['chi_eff'][pn_spin_order] - chi_eff0]) for pn_spin_order in pn_spin_orders], array=pn_spin_orders, ls=':', label=r'pyEFPE')
ax.add_collection(chi_eff_segments)
ax.add_collection(pyEFPE_chi_eff_segments)
fig.colorbar(chi_eff_segments, label='pn_spin_orders', ax=ax)
ax.set_xlabel(r'$y$')
ax.set_ylabel(r'$\chi_\mathrm{eff} - \chi_{\mathrm{eff},0}$')
ax.set_xscale('log')
ax.autoscale()
ax.set_xlim(y0, yf)
ax.legend()
plt.tight_layout()

#plot exact DJ2 and the one predicted by pyEFPE
DJ20 = 2*np.dot(s_1_0[[0,1]], s_2_0[[0,1]])
fig, ax = plt.subplots(figsize=(16,10))
DJ2_segments = LineCollection([np.column_stack([ys, result['DJ2'][pn_spin_order] - DJ20]) for pn_spin_order in pn_spin_orders], array=pn_spin_orders, ls='-', label=r'$\Delta_{J_N^2}^\mathrm{exact}$')
pyEFPE_DJ2_segments = LineCollection([np.column_stack([ys, result['pyEFPE']['DJ2'][pn_spin_order] - DJ20]) for pn_spin_order in pn_spin_orders], array=pn_spin_orders, ls='--', label=r'$\Delta_{J_N^2}^\mathrm{pyEFPE}$')
ax.add_collection(DJ2_segments)
ax.add_collection(pyEFPE_DJ2_segments)
fig.colorbar(DJ2_segments, label='pn_spin_orders', ax=ax)
ax.set_xlabel(r'$y$')
ax.set_xscale('log')
ax.autoscale()
ax.set_xlim(y0, yf)
ax.legend()
plt.tight_layout()

#plot exact e2 and the one predicted by pyEFPE
fig, ax = plt.subplots(figsize=(16,10))
e2_segments = LineCollection([np.column_stack([ys, result['e2'][pn_spin_order]]) for pn_spin_order in pn_spin_orders], array=pn_spin_orders, ls='-', label=r'$e^2_\mathrm{exact}$')
pyEFPE_e2_segments = LineCollection([np.column_stack([ys, result['pyEFPE']['e2'][pn_spin_order]]) for pn_spin_order in pn_spin_orders], array=pn_spin_orders, ls='--', label=r'$e^2_\mathrm{pyEFPE}$')
ax.add_collection(e2_segments)
ax.add_collection(pyEFPE_e2_segments)
fig.colorbar(e2_segments, label='pn_spin_orders', ax=ax)
ax.set_xlabel(r'$y$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.autoscale()
ax.set_xlim(y0, yf)
ax.legend()
plt.tight_layout()

#plot evolution of cos(\theta)
fig, ax = plt.subplots(figsize=(16,10))
cth_segments = LineCollection([np.column_stack([ys, result['cth'][pn_spin_order]]) for pn_spin_order in pn_spin_orders], array=pn_spin_orders, ls='-', label=r'$(\hat{j} \cdot \hat{l}_N)$')
pyEFPE_cth_segments = LineCollection([np.column_stack([ys, result['pyEFPE']['cth'][pn_spin_order]]) for pn_spin_order in pn_spin_orders], array=pn_spin_orders, ls='--', label=r'$(\hat{j} \cdot \hat{l}_N)^\mathrm{pyEFPE}$')
ax.add_collection(cth_segments)
ax.add_collection(pyEFPE_cth_segments)
fig.colorbar(cth_segments, label='pn_spin_orders', ax=ax)
ax.set_xlabel(r'$y$')
ax.set_xscale('log')
ax.autoscale()
ax.set_xlim(y0, yf)
ax.legend()
plt.tight_layout()

#plot evolution of initial and evolved angular momenta
fig, ax = plt.subplots(figsize=(16,10))
jNjN0_segments = LineCollection([np.column_stack([ys, result['jNjN0'][pn_spin_order]]) for pn_spin_order in pn_spin_orders], array=pn_spin_orders, ls='-', label=r'$(\hat{j}_N \cdot \hat{j}_{N,0})$')
jj0_segments = LineCollection([np.column_stack([ys, result['jj0'][pn_spin_order]]) for pn_spin_order in pn_spin_orders], array=pn_spin_orders, ls='--', label=r'$(\hat{j} \cdot \hat{j}_{0})$')
ax.add_collection(jNjN0_segments)
ax.add_collection(jj0_segments)
fig.colorbar(jNjN0_segments, label='pn_spin_orders', ax=ax)
ax.set_xlabel(r'$y$')
ax.set_xscale('log')
ax.autoscale()
ax.set_xlim(y0, yf)
ax.legend(loc='lower left')
plt.tight_layout()


#plot of evolution of Euler Angle difference
fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(16,10), sharex=True)
phiz_segments = LineCollection([np.column_stack([ys, result['phiz'][pn_spin_order]]) for pn_spin_order in pn_spin_orders], array=pn_spin_orders, ls='-', label=r'$\phi_z$')
pyEFPE_phiz_segments = LineCollection([np.column_stack([ys, result['pyEFPE']['phiz'][pn_spin_order]]) for pn_spin_order in pn_spin_orders], array=pn_spin_orders, ls='--', label=r'$\phi_z^\mathrm{pyEFPE}$')
dphiz_segments = LineCollection([np.column_stack([ys, result['pyEFPE']['phiz'][pn_spin_order] - result['phiz'][pn_spin_order]]) for pn_spin_order in pn_spin_orders], array=pn_spin_orders, ls='-', label=r'$\phi_z^\mathrm{pyEFPE} - \phi_z$')
axs[0].add_collection(phiz_segments)
axs[0].add_collection(pyEFPE_phiz_segments)
axs[1].add_collection(dphiz_segments)
axs[1].set_xlabel(r'$y$')
for ax in axs:
	ax.set_xscale('log')
	ax.autoscale()
	ax.set_xlim(y0, yf)
	ax.legend(loc='upper left')

fig.tight_layout()
fig.colorbar(phiz_segments, ax=axs, orientation='vertical', fraction=0.05, pad=0.04, label='pn_spin_orders')

#plot evolution of phiz+zeta
fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(16,10), sharex=True)
Dllphase_segments = LineCollection([np.column_stack([ys, result['phiz'][pn_spin_order] + result['zeta'][pn_spin_order]]) for pn_spin_order in pn_spin_orders], array=pn_spin_orders, ls='-', label=r'$\phi_z + \zeta$')
pyEFPE_Dllphase_segments = LineCollection([np.column_stack([ys, result['pyEFPE']['phiz'][pn_spin_order] + result['pyEFPE']['zeta'][pn_spin_order]]) for pn_spin_order in pn_spin_orders], array=pn_spin_orders, ls='--', label=r'$\phi_z^\mathrm{pyEFPE} + \zeta^\mathrm{pyEFPE}$')
dDllphase_segments = LineCollection([np.column_stack([ys, (result['pyEFPE']['phiz'][pn_spin_order] + result['pyEFPE']['zeta'][pn_spin_order]) - (result['phiz'][pn_spin_order] + result['zeta'][pn_spin_order])]) for pn_spin_order in pn_spin_orders], array=pn_spin_orders, ls='-', label=r'Difference')
axs[0].add_collection(Dllphase_segments)
axs[0].add_collection(pyEFPE_Dllphase_segments)
axs[1].add_collection(dDllphase_segments)
axs[1].set_xlabel(r'$y$')
for ax in axs:
	ax.set_xscale('log')
	ax.autoscale()
	ax.set_xlim(y0, yf)
	ax.legend(loc='upper left')

fig.tight_layout()
fig.colorbar(Dllphase_segments, ax=axs, orientation='vertical', fraction=0.05, pad=0.04, label='pn_spin_orders')

plt.show()

