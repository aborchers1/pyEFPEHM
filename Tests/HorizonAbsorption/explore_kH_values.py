import os

import numpy as np
import tqdm

#function to compute mean and standard deviation from histogram counts
def hist_mean_std(hist_counts, bin_edges):

	#make sure we have numpy arrays
	hist_counts = np.asarray(hist_counts)
	bin_edges = np.asarray(bin_edges)

	#compute bin centers
	bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])

	#compute total count
	total = np.sum(hist_counts)

	#compute mean
	mean = np.sum(hist_counts*bin_centers)/total

	#compute standard deviation
	std = np.sqrt(np.sum(hist_counts*np.square(bin_centers - mean))/total)

	#return mean and standard deviation
	return mean, std

#function to compute kH as in pyEFPEHM
def compute_kH_pyEFPEHM(q, chi1, cth1, chi2, cth2):

	#compute required things
	mu1 = 1/(1 + q)
	mu2 = q*mu1
	dmu = mu1 - mu2
	nu  = mu1*mu2
	s1 = mu1*chi1
	s2 = mu2*chi2
	s2_1 = s1*s1
	s2_2 = s2*s2
	chi_eff = s1*cth1 + s2*cth2
	dchi    = s1*cth1 - s2*cth2

	#return the value of kH
	return (dmu + (9./8.)*(s2_1 - s2_2))*dchi + chi_eff*(1 - 2*nu + (9./8.)*(s2_1 + s2_2) + (45./16.)*dchi*dchi + (15./16.)*chi_eff*chi_eff)

####################################################################

#minimum and maximum parameters to consider
q_min = 0
q_max = 1
chi_min = 0
chi_max = 1

#number of samples per batch
N_per_batch = int(1e7)
#number of batches
Nbatches = int(1e3)

#Bins to use for kH histo
bin_edges_kH = np.linspace(-8, 8, 1601)

#bin edges to use for chi_eff-kH 2D histogram
bin_edges_2D_chi = np.linspace(-1, 1, 101)
bin_edges_2D_kH  = np.linspace(-8, 8, 161)

#folder to save the plots in
plotdir = './Plots'

####################################################################

#make a string ID
stringID = '_q_%.3g_%.3g_chi_%.3g_%.3g_N_%.3g'%(q_min, q_max, chi_min, chi_max, Nbatches*N_per_batch)

#parameter ranges for [q, chi1, cth1, chi2, cth2]
params_min = [q_min, chi_min, -1, chi_min, -1]
params_max = [q_max, chi_max,  1, chi_max,  1]

#loop over batches
hist_kH = np.zeros(len(bin_edges_kH)-1)
hist2D_chi_kH = np.zeros((len(bin_edges_2D_chi)-1, len(bin_edges_2D_kH)-1))
bin_edges_2D = [bin_edges_2D_chi, bin_edges_2D_kH]
for ibatch in tqdm.tqdm(range(Nbatches)):
	
	#generate random spins and mass rations
	q, chi1, cth1, chi2, cth2 = np.random.uniform(low=params_min, high=params_max, size=(N_per_batch, len(params_min))).T

	#compute reduced masses
	mu1 = 1/(1 + q)
	mu2 = q*mu1
	
	#compute chi_eff
	chi_eff = mu1*chi1*cth1 + mu2*chi2*cth2
	
	#compute kH
	kH = 2*((mu1**3)*chi1*cth1*(1 + chi1*chi1*(9./8. + (15/8)*cth1*cth1)) + (mu2**3)*chi2*cth2*(1 + chi2*chi2*(9./8. + (15/8)*cth2*cth2)))
	
	#make 1D histogram of kH
	hist_kH_ibatch, _ = np.histogram(kH, bins=bin_edges_kH)
	#add the counts to the total
	hist_kH += hist_kH_ibatch
	
	#make 2D histogram of chi_eff-kH
	hist2D_chi_kH_ibatch, _, _ = np.histogram2d(chi_eff, kH, bins=bin_edges_2D)
	#add the counts to the total
	hist2D_chi_kH += hist2D_chi_kH_ibatch


#compute kH as in pyEFPEHM
kH_EFPE = compute_kH_pyEFPEHM(q, chi1, cth1, chi2, cth2)

print('Relative MSE between the two kH formulas:',np.linalg.norm(kH_EFPE - kH)/np.linalg.norm(kH))

#compute the mean and standard deviation from the histogram
mean_kH, std_kH = hist_mean_std(hist_kH, bin_edges_kH)

#compute the probabilty density of kH
bin_widths_kH = bin_edges_kH[1:] - bin_edges_kH[:-1]
p_kH = hist_kH/(np.sum(hist_kH)*bin_widths_kH)

#compute the probabilty density of p(chi, kH)
bin_widths_2D_chi = np.diff(bin_edges_2D_chi)[:,None]
bin_widths_2D_kH  = np.diff(bin_edges_2D_kH)[None,:]
p2D_chi_kH = hist2D_chi_kH/(np.sum(hist2D_chi_kH)*bin_widths_2D_chi*bin_widths_2D_kH)

#compute conditional probability p(kH|chi)
p_chi = np.sum(bin_widths_2D_kH*p2D_chi_kH, axis=1)
p_kH_given_chi = p2D_chi_kH/p_chi[:,None]

print("mean(kH) =", mean_kH)
print("std(kH)  =", std_kH)

####################################################################

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
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
        'patch.force_edgecolor':True,
        })

#make sure directory for plots exists
if not os.path.exists(plotdir): os.makedirs(plotdir)

#plot 1D histogram of kH
y = np.append(p_kH, p_kH[-1])
fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)
ax.step(bin_edges_kH, y, where='post', lw=2)
ax.set_xlim(bin_edges_kH[0], bin_edges_kH[-1])
ax.set_xlabel(r'$\tilde{\kappa}_H$')
plt.ylabel(r'$p(\tilde{\kappa}_H)$')
ax.set_yscale('log')
fig.savefig(plotdir+'/p_kH'+stringID+'.pdf')

#plot 2D histogram of chi_eff-kH
CHI_2D, KH_2D = np.meshgrid(bin_edges_2D_chi, bin_edges_2D_kH)
fig, ax = plt.subplots(figsize=(10, 8), tight_layout=True)
pmesh = ax.pcolormesh(CHI_2D, KH_2D, p2D_chi_kH.T,
    norm=LogNorm(vmin=p2D_chi_kH[p2D_chi_kH>0].min(), vmax=p2D_chi_kH.max()),
    edgecolors='none',
    linewidth=0,
    shading='auto',
    rasterized=True
)
fig.colorbar(pmesh, ax=ax, label=r"$p(\chi_\mathrm{eff},\tilde{\kappa}_H)$")
ax.set_xlim(bin_edges_2D_chi[0], bin_edges_2D_chi[-1])
ax.set_xlabel(r"$\chi_\mathrm{eff}$")
ax.set_ylim(bin_edges_2D_kH[0], bin_edges_2D_kH[-1])
ax.set_ylabel(r"$\tilde{\kappa}_H$")
fig.savefig(plotdir+'/p2D_chi_kH'+stringID+'.pdf')

#plot 2D histogram of p(kH|chi)
fig, ax = plt.subplots(figsize=(10, 8), tight_layout=True)
pmesh = ax.pcolormesh(CHI_2D, KH_2D, p_kH_given_chi.T,
    norm=LogNorm(vmin=p_kH_given_chi[p_kH_given_chi>0].min(), vmax=p_kH_given_chi.max()),
    edgecolors='none',
    linewidth=0,
    shading='auto',
    rasterized=True
)
fig.colorbar(pmesh, ax=ax, label=r"$p(\tilde{\kappa}_H|,\chi_\mathrm{eff})$")
ax.set_xlim(bin_edges_2D_chi[0], bin_edges_2D_chi[-1])
ax.set_xlabel(r"$\chi_\mathrm{eff}$")
ax.set_ylim(bin_edges_2D_kH[0], bin_edges_2D_kH[-1])
ax.set_ylabel(r"$\tilde{\kappa}_H$")
fig.savefig(plotdir+'/p_kH_given_chi'+stringID+'.pdf')


plt.show()

