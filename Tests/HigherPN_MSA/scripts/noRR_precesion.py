import numpy as np
import time
from precesion_functions import *

##############################################################################################

#Inputs for random parameters
y_min = 0
y_max = 6**-0.5
q_min = 0
q_max = 1
a_min = 0
a_max = 1

#pn spin orders to loop over
pn_spin_order_min = 4
pn_spin_order_max = 8

#simulation time
y5tmax = 100
N_t    = int(1e3)

##############################################################################################

#generate random parameters
q = np.random.uniform(low=q_min, high=q_max)
y = np.random.uniform(low=y_min, high=y_max) 

#generate random angular momenta
s_1_0, s_2_0, lN_0 = np.random.uniform(low=[-1,-1,-1], high=[1,1,1], size=(3,3))
a_1, a_2 = np.random.uniform(low=a_min, high=a_max, size=2)
s_1_0, s_2_0, lN_0 = a_1*s_1_0/np.linalg.norm(s_1_0), a_2*s_2_0/np.linalg.norm(s_2_0), lN_0/np.linalg.norm(lN_0)

print("y = %.3g, q = %.3g"%(y, q))
print("s_1_0 =", s_1_0)
print("s_2_0 =", s_2_0)
print("lN_0  =", lN_0)

#array of times to simulate
tmax = y5tmax*(y**-5)
times = np.linspace(0, tmax, N_t)

#compute nu and dmu from mass ratio
dmu = (1 - q)/(1 + q)
nu  = q/((1+q)**2)

#loop over pn_spin_orders
result = {key: {} for key in ['s1s', 's2s', 'lNs', 's2_1', 's2_2', 'lN2', 'chi_eff', 'dchi', 'J', 'JN']}
pn_spin_orders = np.arange(pn_spin_order_min, pn_spin_order_max+1)
for pn_spin_order in pn_spin_orders:
	
	#evolve spins
	soltime = time.time()
	ivp_sol = evolve_precessing_system_noRR(s_1_0, s_2_0, lN_0, y, nu, dmu, tmax, pn_spin_order=pn_spin_order)
	soltime = time.time() - soltime
	print("Time to evolve spins for pn_spin_order=%s: %.3g s"%(pn_spin_order, soltime))
	
	#evaluate solution
	momenta = ivp_sol.sol(times)
	s1s = momenta[ :3,:]
	s2s = momenta[3:6,:]
	lNs = momenta[6:9,:]

	#save values in result dictionary
	result['s1s'][pn_spin_order] = s1s
	result['s2s'][pn_spin_order] = s2s
	result['lNs'][pn_spin_order] = lNs

	#compute the different scalars
	result['s2_1'][pn_spin_order] = np.einsum('ij,ij -> j', s1s, s1s)
	result['s2_2'][pn_spin_order] = np.einsum('ij,ij -> j', s2s, s2s)
	result['lN2'][pn_spin_order]  = np.einsum('ij,ij -> j', lNs, lNs)
	result['chi_eff'][pn_spin_order]  = np.einsum('ij,ij -> j', s1s + s2s, lNs)
	result['dchi'][pn_spin_order]     = np.einsum('ij,ij -> j', s1s - s2s, lNs)
	
	#compute total angular momentum
	result['J'][pn_spin_order]  = compute_J(s1s, s2s, lNs, y, nu, dmu, pn_spin_order=pn_spin_order)

	#compute newtonian total angular momentum
	result['JN'][pn_spin_order] = compute_J(s1s, s2s, lNs, y, nu, dmu, pn_spin_order=0)
	
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
s2_1_segments = LineCollection([np.column_stack([times, np.sqrt(result['s2_1'][pn_spin_order]) - a_1]) for pn_spin_order in pn_spin_orders], array=pn_spin_orders, ls='-', label=r'$s_1 - s_{1,0}$')
s2_2_segments = LineCollection([np.column_stack([times, np.sqrt(result['s2_2'][pn_spin_order]) - a_2]) for pn_spin_order in pn_spin_orders], array=pn_spin_orders, ls='--', label=r'$s_2 - s_{2,0}$')
lN2_segments  = LineCollection([np.column_stack([times, np.sqrt(result['lN2'][pn_spin_order])  - 1  ]) for pn_spin_order in pn_spin_orders], array=pn_spin_orders, ls=':', label=r'$l_N - 1$')
ax.add_collection(s2_1_segments)
ax.add_collection(s2_2_segments)
ax.add_collection(lN2_segments)
fig.colorbar(s2_1_segments, label='pn_spin_orders', ax=ax)
ax.set_xlabel(r'$t/M$')
ax.set_xlim(0, tmax)
ax.autoscale()
ax.legend()
plt.tight_layout()

#plot chi_eff and dchi
chi_eff_0 = np.dot(lN_0, s_1_0 + s_2_0)
dchi_0    = np.dot(lN_0, s_1_0 - s_2_0)
fig, ax = plt.subplots(figsize=(16,10))
chi_eff_segments = LineCollection([np.column_stack([times, result['chi_eff'][pn_spin_order] - chi_eff_0]) for pn_spin_order in pn_spin_orders], array=pn_spin_orders, ls='-', label=r'$\chi_\mathrm{eff}$')
dchi_segments    = LineCollection([np.column_stack([times,    result['dchi'][pn_spin_order] - dchi_0   ]) for pn_spin_order in pn_spin_orders], array=pn_spin_orders, ls='--', label=r'$\delta\chi$')
ax.add_collection(chi_eff_segments)
ax.add_collection(dchi_segments)
fig.colorbar(chi_eff_segments, label='pn_spin_orders', ax=ax)
ax.set_xlabel(r'$t/M$')
ax.set_xlim(0, tmax)
ax.autoscale()
ax.legend()
plt.tight_layout()

#plot of evolution of total angular momentum
fig, ax = plt.subplots(figsize=(16,10))
J_segments  = LineCollection([np.column_stack([times, np.linalg.norm(result['J'][pn_spin_order]  - result['J'][pn_spin_order][:,[0]], axis=0) ]) for pn_spin_order in pn_spin_orders], array=pn_spin_orders, ls='-', label=r'$|\mathbf{J}_\mathrm{PN} - \mathbf{J}_{\mathrm{PN},0}|$')
JN_segments = LineCollection([np.column_stack([times, np.linalg.norm(result['JN'][pn_spin_order] - result['JN'][pn_spin_order][:,[0]], axis=0)]) for pn_spin_order in pn_spin_orders], array=pn_spin_orders, ls=':', label=r'$|\mathbf{J}_\mathrm{N}  - \mathbf{J}_{\mathrm{N},0}|$')
ax.add_collection(J_segments)
ax.add_collection(JN_segments)
fig.colorbar(J_segments, label='pn_spin_orders', ax=ax)
ax.set_xlabel(r'$t/M$')
ax.set_xlim(0, tmax)
ax.autoscale()
ax.legend()
plt.tight_layout()


plt.show()

