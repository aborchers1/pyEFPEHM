import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

import pyEFPEHM
import numpy as np
import time
start_runtime = time.time()

#array with parameters
params = {'mass1': 1.824,
          'mass2': 0.739,
          'e_start': 0.7,
          'spin1x': -0.44,
          'spin1y': -0.26,
          'spin1z': 0.48,
          'spin2x': -0.31,
          'spin2y': 0.01,
          'spin2z': -0.84,
          'inclination': 1.57,
          'f22_start': 10,
          'Amplitude_tol': 1e-4,
          'mode_array': [[2,0],[2,1],[2,2],[3,0],[3,1],[3,2],[3,3],[4,0],[4,2],[4,4]],
          'SUA_kmax': 1,
          }

#stuff for frequency array generation
seglen = 256
flow = 20
fhigh = 4096

#initialize waveform class
start_soltime = time.time()
wf = pyEFPEHM.pyEFPE(params)
print("\nTime to initialize waveform class: %s seconds \n" % (time.time() - start_soltime))

start_soltime = time.time()
wf.interpolate_Apc_prec()
print("\nTime to interpolate Apc: %s seconds \n" % (time.time() - start_soltime))

start_soltime = time.time()
wf.interpolate_Nlm_p()
print("\nTime to interpolate Nlm_p: %s seconds \n" % (time.time() - start_soltime))

#print waveform time-domain amplitude
print('Waveform duration: %.5gs'%(-wf.sol.ts[0]))
#print info on the necessary modes
interp_idx, mode_counts = np.unique(wf.mode_interp_idx, return_counts=True)
print('Necessary Modes: Total: %s modes in %s segments'%(len(wf.mode_interp_idx),len(interp_idx)))
print(mode_counts[np.argsort(interp_idx)])
print('Number of points to interpolate Wigner D-matrices:', len(wf.prec_interp_ts))

#test series reversion
from pyEFPEHM.utils import series_reversion, power_range
Nx_test = 1001
orders = [1,2,3,4,5]

#compute test input (it goes between 0 and 1)
x = np.linspace(0, 1, Nx_test)

#extract matrices of all interpolants
Q = wf.mode_phases_Qs[1]

#compute value of interpolant
px = np.cumprod(np.tile(x, (Q.shape[1],1)), axis=0)
y = np.dot(Q, px)

#loop over orders
errors = np.zeros(len(orders))
for iorder, order in enumerate(orders):
	
	#compute inverse of interpolant
	invQ = series_reversion(Q, order=order)
		
	#compute its appoximate inverse (it should be equal to x)
	py = np.cumprod(np.tile(y, (invQ.shape[1],1,1)), axis=0)
	x_approx = np.einsum('ij,jil->il',invQ, py, optimize='greedy')
		
	#compute error in approximate index
	errors[iorder] = np.amax(np.abs(x[...,:] - x_approx)) #np.linalg.norm(x[...,:] - x_approx)/np.sqrt(np.prod(x_approx.shape))

print('\nSeries reversing errors for orders =', orders, '->', errors)

#test subroutine to compute stationary times
freqs = np.arange(flow, fhigh, 1/seglen)

start_soltime = time.time()
f_idxs, interp_idxs, t_stationary, phase_stationary, Tnm_stationary = wf.stationary_times(freqs)
print("\nTime to compute stationary times: %s seconds \n" % (time.time() - start_soltime))

#compute frequencies deduced from these stationary times
idxs_t_interp = wf.mode_interp_idx[interp_idxs]
x_stationary = (t_stationary - wf.sol.ts[idxs_t_interp])/wf.sol.hs[idxs_t_interp]
ws_approx = wf.ts_interp_w0[interp_idxs] + np.sum(np.transpose(wf.mode_phases_Qs[1][interp_idxs,:])*power_range(x_stationary, wf.mode_phases_Qs[1].shape[1]), axis=0)

print('Mean squared error in f(ts(f)):     %.3g Hz'%(np.linalg.norm((ws_approx/(2*np.pi) - freqs[f_idxs])/freqs[f_idxs])/np.sqrt(len(f_idxs))))
print('Maximum relative error in f(ts(f)): %.3g'%(np.amax(np.abs(ws_approx/(2*np.pi) - freqs[f_idxs])/freqs[f_idxs])))

#test exact and approximated amplitude interpolation
start_soltime = time.time()
Nlm_p_interp = wf.compute_Nlm_p_interpolated(t_stationary, interp_idxs)
print("\nTime to compute Nlm_p interp: %s seconds" % (time.time() - start_soltime))
start_soltime = time.time()
Nlm_p_exact = wf.compute_Nlm_p_exact(t_stationary, interp_idxs)
print(  "Time to compute Nlm_p exact: %s seconds" % (time.time() - start_soltime))

#analize error
Nlm_p_abs_err = np.abs(Nlm_p_exact - Nlm_p_interp)
Nlm_p_rel_err = np.abs(0.5*(Nlm_p_exact - Nlm_p_interp)/(Nlm_p_exact + Nlm_p_interp))
print('Absolute error in Nlm_p mode interpolation: MSE: %.3g    max: %.3g'%(np.linalg.norm(Nlm_p_abs_err)/(Nlm_p_exact.size**0.5), np.amax(Nlm_p_abs_err)))
print('Relative error in Nlm_p mode interpolation: MSE: %.3g    max: %.3g'%(np.linalg.norm(Nlm_p_rel_err)/(Nlm_p_exact.size**0.5), np.amax(Nlm_p_rel_err)))

#extract the modes that are in the stationary times
multipole_idxs = wf.necessary_multipole_idxs[interp_idxs]
ps = wf.necessary_ps[interp_idxs]

#test Amplitude generation
start_soltime = time.time()
Apc_prec = wf.compute_Apc_prec(t_stationary, multipole_idxs, ps)
print("\nTime to compute Apc_prec: %s seconds \n" % (time.time() - start_soltime))

#test SUA amplitude generation
start_soltime = time.time()
SUA_Amps = wf.SUA_Amplitudes(t_stationary, interp_idxs, Tnm_stationary)
print("\nTime to compute SUA Amplitudes: %s seconds \n" % (time.time() - start_soltime))

#test polarization generation
start_soltime = time.time()
hp, hc = wf.generate_waveform(freqs)
print("\nTime to compute hp, hc: %s seconds \n" % (time.time() - start_soltime))

#compute frequency domain modes
start_soltime = time.time()
result_fd_modes = wf.generate_modes(freqs)
print("\nTime to compute frequency domain modes: %s seconds" % (time.time() - start_soltime))

#check that the sum of the frequency-domain modes gives the frequency-domain waveform
wf_fd_from_modes = np.zeros((2,len(result_fd_modes['freqs'])), dtype=np.complex128)
for mode_label, mode in result_fd_modes['modes'].items():
	wf_fd_from_modes[:,mode['freq_idxs']] += mode['polarizations']

#print the error between direct frequency-domain waveform and waveform from modes
wf_fd_direct = np.array([hp, hc])
print("Relative error for computing waveform from modes: %.3g\n"%(np.linalg.norm(wf_fd_direct - wf_fd_from_modes)/np.linalg.norm(wf_fd_direct)))

#compute the waveforms in the time domain
from scipy.fft import irfft, rfft
#compute the time array
delta_t = 1/(2*fhigh)
times = delta_t*np.arange(int(2*seglen*fhigh))
fft_len = int(2*seglen*fhigh)
#loop over polarizations
h_td = list()
for h in [hp, hc]:
	#put the low frequencies of h (they are 0)
	h_padded = np.zeros(int(seglen*fhigh), dtype=h.dtype)
	h_padded[int(seglen*flow):int(seglen*fhigh)] = h
	#perform the inverse FFT, taking into account that h is the FFT of a real signal
	fft_len = int(2*seglen*fhigh)
	h_td.append((fft_len/seglen)*irfft(h_padded, n=fft_len))

#compute time-domain waveform
times = times-times[-1]
start_soltime = time.time()
hp_td_direct, hc_td_direct = wf.generate_tdomain_waveform(times)
print("\nTime to compute time-domain waveform: %s seconds \n" % (time.time() - start_soltime))

#compute time-domain waform modes
start_soltime = time.time()
result_td_modes = wf.generate_tdomain_modes(times)
print("\nTime to compute time-domain modes: %s seconds" % (time.time() - start_soltime))

#check that the sum of the time-domain modes gives the time-domain waveform
wf_td_from_modes = np.zeros((2,len(result_td_modes['times'])))
for mode_label, mode in result_td_modes['modes'].items():
	wf_td_from_modes[:,mode['time_idxs']] += mode['polarizations']

#print the error between direct time-domain waveform and waveform from modes
wf_td_direct = np.array([hp_td_direct, hc_td_direct])
print("Relative error for computing waveform from modes: %.3g\n"%(np.linalg.norm(wf_td_direct - wf_td_from_modes)/np.linalg.norm(wf_td_direct)))

#compute frequency-domain and filtered waveforms
h_fd = list()
h_filtered = list()
for h in [hp_td_direct, hc_td_direct]:
	#compute fourier domain waveform
	h_rfft = delta_t*rfft(h)
	#create frequency mask
	frequency_mask = np.zeros(len(h_rfft), dtype=bool)
	frequency_mask[int(seglen*flow):int(seglen*fhigh)] = True
	#save it without low and high frequencies
	h_fd.append(h_rfft[frequency_mask])
	#make zero the values of rfft outside frequency range
	h_rfft[~frequency_mask] = 0
	#do the inverse fft
	h_filtered.append((fft_len/seglen)*irfft(h_rfft, n=fft_len))

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

#Compute points for plots
Nt_plot = 100000
times_plot = -np.geomspace(-wf.sol.all_ts[0], -wf.sol.all_ts[-1], Nt_plot)
bpsip_plot, phiz_plot = wf.sol(times_plot, idxs=[5,6])
x_plot = bpsip_plot + phiz_plot

#Compare exact and interpolated polarization amplitudes for the different modes
plt.figure(figsize=(13,8))
params_exact = params.copy()
params_exact['Interpolate_Amplitudes'] = False
wf_exact = pyEFPEHM.pyEFPE(params_exact)
Apc_prec_exact, Apc_prec_interp = [], []
	
for im, mode in enumerate(wf.mode_array):

	if wf.Apc_prec_cspline[im] is not None:
	
		#compute exact and interpolated amplitudes (set p=1 to not conjugate amplitude)
		Apc_prec_exact.append(wf_exact.compute_Apc_prec(times_plot, np.full(len(times_plot),im), np.full(len(times_plot), 1)))
		Apc_prec_interp.append(     wf.compute_Apc_prec(times_plot, np.full(len(times_plot),im), np.full(len(times_plot), 1)))

		iP, label_P = 1, r'\times' #iP, label_P = 0, r'+'
		plt.plot(x_plot, np.abs( Apc_prec_exact[-1][:,iP])**2, label=r'$|A_{%s, %s}|^2$'%(label_P, mode))
		plt.plot(x_plot, np.abs(Apc_prec_interp[-1][:,iP])**2, 'k:')

plt.xlabel(r'$\overline{\psi}_p + \phi_{z,0} $ [s]')
plt.legend()
plt.tight_layout()

#compute error between exact and interpolated polarization amplitudes
Apc_prec_exact, Apc_prec_interp = np.array(Apc_prec_exact), np.array(Apc_prec_interp)
Apc_prec_abs_error = np.abs(Apc_prec_exact - Apc_prec_interp)
Apc_prec_rel_err = np.abs(0.5*(Apc_prec_exact - Apc_prec_interp)/(Apc_prec_exact + Apc_prec_interp))
print('\nAbsolute error in Apc interpolation: MSE: %.3g    max: %.3g'%(np.linalg.norm(Apc_prec_abs_error)/(Apc_prec_abs_error.size**0.5), np.amax(Apc_prec_abs_error)))
print('Relative error in Apc interpolation: MSE: %.3g    max: %.3g'%(np.linalg.norm(Apc_prec_rel_err)/(Apc_prec_rel_err.size**0.5), np.amax(Apc_prec_rel_err)))

#compute exact dinamical variables
y, e2, DJ2, bpsip, phiz, zeta = wf.sol(times_plot, idxs=[0,1,4,5,6,7])
wf.MSA.update(y, DJ2)
dphiz, dzeta, costhL = wf.MSA.precession_Euler_angles(bpsip)
phiz += dphiz
zeta += dzeta

#compute the derivatives of the Euler angles
Dphiz_dt = np.diff(phiz)/np.diff(times_plot)
Dzeta_dt = np.diff(zeta)/np.diff(times_plot)
costhL_mid = 0.5*(costhL[1:] + costhL[:-1])
bpsip_plot_mid = 0.5*(bpsip_plot[1:] + bpsip_plot[:-1])
times_plot_mid = 0.5*(times_plot[1:] + times_plot[:-1])
from scipy.integrate import cumulative_trapezoid
min_rot_viol = cumulative_trapezoid(Dzeta_dt + costhL_mid*Dphiz_dt , x=times_plot_mid, initial=0)

#make a plot of Euler angles
plt.figure(figsize=(13,8))
plt.plot((2/np.pi)*bpsip_plot, dphiz, label=r'$\delta\phi_z$')
plt.plot((2/np.pi)*bpsip_plot, dzeta, label=r'$\delta\zeta$')
plt.plot((2/np.pi)*bpsip_plot, 1-costhL, label=r'$1-\cos{\theta_L}$')
plt.plot((2/np.pi)*bpsip_plot_mid, min_rot_viol , label=r'$\int \mathrm{d}t \left(\dot{\zeta} + \cos(\theta_L) \dot{\phi}_z \right)$')
plt.xlabel(r'$2\overline{\psi}_p/\pi$ [s]')
plt.legend()
plt.tight_layout()

plt.figure(figsize=(13,8))
plt.plot(freqs,  np.abs(hp), 'C0-', alpha=0.5, label=r'$|h_+|$')
plt.plot(freqs,  np.abs(h_fd[0]), 'C0--', alpha=0.5, label=r'$|h_+^\mathrm{FFT}|$')
plt.plot(freqs,  np.abs(hc), 'C1-', alpha=0.5, label=r'$|h_\times|$')
plt.plot(freqs,  np.abs(h_fd[1]), 'C1--', alpha=0.5, label=r'$|h_\times^\mathrm{FFT}|$')
plt.xlabel(r'$f$ [Hz]')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.tight_layout()

#plot the waveform in the time domain
plt.figure(figsize=(13,8))
plt.plot(times, h_td[0], 'C0-', alpha=0.5, label=r'$h_+^\mathrm{iFFT}(t)$')
plt.plot(times, h_filtered[0], 'C0--', alpha=0.5, label=r'$h_+(t)$')
plt.plot(times, h_td[1], 'C1-', alpha=0.5, label=r'$h_\times^\mathrm{iFFT}(t)$')
plt.plot(times, h_filtered[1], 'C1--', alpha=0.5, label=r'$h_\times(t)$')
plt.xlabel(r'$t$ [s]')
plt.legend()
plt.tight_layout()


from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

#put polarizations back into an array
hphc_fd = np.asarray([hp, hc])
hphc_td = np.asarray([hp_td_direct, hc_td_direct])

# Plot the frequency-domain polarizations and their mode decompositions
fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, constrained_layout=True, figsize=(14,8))
for ip, plabel in enumerate([r'+', r'\times']):
	# Plot polarization strain
	axs[ip].plot(freqs, np.abs(hphc_fd[ip]), label=r'$|\tilde{h}_{%s}|$'%(plabel), color='C1')
	# Put modes in a list to plot as a Line Collection
	mode_lines_fd = []
	for mode_label, mode in result_fd_modes['modes'].items():
		# We want to plot consecutive chunks, find breaks
		breaks = np.nonzero(np.diff(mode['freq_idxs']) > 1)[0] + 1
		# Create a Line with [f, |h|_{l,m,n}] per consecutive chunk
		for fplot, hplot in zip(np.split(freqs[mode['freq_idxs']], breaks), np.split(np.abs(mode['polarizations'][ip]), breaks)):
			mode_lines_fd.append(np.column_stack([fplot, hplot]))
	# Color modes by their n value, using that mode_label=(l,m.n)
	n_modes = [mode_label[2] for mode_label in result_fd_modes['modes'].keys()]
	# Plot Line Collection
	line_collection = LineCollection(mode_lines_fd, array=n_modes, norm=LogNorm(vmin=min(n_modes), vmax=max(n_modes)), label=r'$|\tilde{h}_{l,m,n,%s}|$'%(plabel))
	axs[ip].add_collection(line_collection)
	# Aesthetics
	axs[ip].legend(loc='upper right')
	axs[ip].set_yscale('log')
	axs[ip].set_ylabel(r'FD Strain $[\mathrm{Hz}^{-1}]$')
axs[-1].set_xlim(freqs[0], freqs[-1])
axs[-1].set_xscale('log')
axs[-1].set_xlabel(r'Frequency [Hz]')
fig.colorbar(line_collection, ax=axs, label=r'$n$')

# Plot the time-domain polarizations and their mode decompositions
fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, constrained_layout=True, figsize=(14,8))
for ip, plabel in enumerate([r'+', r'\times']):
	# Plot polarization strain
	axs[ip].plot(times, hphc_td[ip], label=r'$h_{%s}$'%(plabel), color='C1')
	# Put modes in a list to plot as a Line Collection
	mode_lines_td = []
	for mode_label, mode in result_td_modes['modes'].items():
		# We want to plot consecutive chunks, find breaks
		breaks = np.nonzero(np.diff(mode['time_idxs']) > 1)[0] + 1
		# Create a Line with [t, h_{l,m,n}] per consecutive chunk
		for tplot, hplot in zip(np.split(times[mode['time_idxs']], breaks), np.split(mode['polarizations'][ip], breaks)):
			mode_lines_td.append(np.column_stack([tplot, hplot]))
	# Color modes by their n value, using that mode_label=(l,m.n)
	n_modes = [mode_label[2] for mode_label in result_td_modes['modes'].keys()]
	# Plot Line Collection
	line_collection = LineCollection(mode_lines_td, array=n_modes, norm=LogNorm(vmin=min(n_modes), vmax=max(n_modes)), label=r'$h_{l,m,n,%s}$'%(plabel))
	axs[ip].add_collection(line_collection)
	# Aesthetics
	axs[ip].legend(loc='lower left')
	axs[ip].set_ylabel(r'TD Strain')
axs[-1].set_xlim(times[0], times[-1])
axs[-1].set_xlabel(r'Time [s]')
fig.colorbar(line_collection, ax=axs, label=r'$n$')

#Runtime
print("\nRuntime: %s seconds" % (time.time() - start_runtime))

plt.show()
