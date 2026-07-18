# pyEFPEHM

`pyEFPEHM` is a Python-based post-Newtonian waveform model for inspiralling precessing-eccentric compact binaries.

## References

If you use `pyEFPEHM` in your research, please cite:

- G. Morras, G. Pratten, P. Schmidt, and A. Buonanno, _Post-Newtonian inspiral waveform model for eccentric precessing binaries with higher-order modes and matter effects_, [arXiv:2604.11903](https://arxiv.org/abs/2604.11903) | [INSPIRE](https://inspirehep.net/literature/3144772)

You can also consider citing:

- G. Morras, _Modeling gravitational wave modes from the inspiral of binaries with arbitrary eccentricity_, [Phys. Rev. D 112 (2025), 084015](https://doi.org/10.1103/d73g-5pfc) | [INSPIRE](https://inspirehep.net/literature/2940874)

- G. Morras, G. Pratten, and P. Schmidt, _Improved post-Newtonian waveform model for inspiralling precessing-eccentric compact binaries_, [Phys. Rev. D 111 (2025), 084052](https://doi.org/10.1103/PhysRevD.111.084052) | [INSPIRE](https://inspirehep.net/literature/2877111)

- A. Alvaro-Díaz and G. Morras, _Horizon absorption in eccentric precessing binary black hole inspirals and its importance for gravitational wave data analysis_, [arXiv:2606.11705](https://arxiv.org/abs/2606.11705) | [INSPIRE](https://inspirehep.net/literature/3167570)

## Installation

To install `pyEFPEHM`, navigate to the root directory of the repository (where `setup.py` is located) and run:

```bash
pip install .
```

## Getting Started

To generate pyEFPEHM waveforms, try the following example:

```python
# Import required packages
import pyEFPEHM
import numpy as np

# Define binary parameters (for additional details see pyEFPEHM/waveform/EFPE.py)
params = {
    'mass1': 10.0,       # Mass of companion 1 (solar masses)
    'mass2': 2.0,        # Mass of companion 2 (solar masses)
    'eccentricity': 0.3, # Eccentricity at the reference frequency
    'spin1x': -0.44,     # Spin components of companion 1 at the reference frequency
    'spin1y': -0.26,
    'spin1z': 0.48,
    'spin2x': -0.31,     # Spin components of companion 2 at the reference frequency
    'spin2y': 0.01,
    'spin2z': -0.84,
    'inclination': 1.57, # Binary inclination at the reference frequency (radians)
    'f22_start': 10,     # Starting (simulation) waveform frequency of GW 22 mode (Hz)
    'f22_ref': 20,       # Reference frequency of GW 22 mode at which the binary parameters are defined (Hz). If None, f22_ref = f22_start
    'Amplitude_tol': 1e-4,    # Amplitude tolerance controling the eccentric harmonics included (see Sec.IID of 2502.03929)
    'mode_array': [[2,0],[2,1],[2,2],[3,0],[3,1],[3,2],[3,3],[4,0],[4,2],[4,4]], # Array containing [l, m] GW modes to take into account
}

# Initialize pyEFPEHM waveform model
wf = pyEFPEHM.pyEFPE(params)

# Define frequency array for waveform generation
freqs = np.arange(20, 1024, 1/64)

# Define time array for waveform generation
times = np.arange(wf.return_start_time(), wf.return_end_time(), 1/2048)

# Compute frequency-domain gravitational wave polarizations
hphc_fd = wf.generate_waveform(freqs)

# Compute time-domain gravitational wave polarizations
hphc_td = wf.generate_tdomain_waveform(times)

# Compute frequency-domain gravitational wave modes
result_fd_modes = wf.generate_modes(freqs)

# Compute time-domain gravitational wave modes
result_td_modes = wf.generate_tdomain_modes(times)
```

You can visualize the time and frequency domain waveform polarizations and their mode decomposition using the following code:

```python
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

# Plot the frequency-domain polarizations and their mode decompositions
fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, constrained_layout=True)
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
fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, constrained_layout=True)
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

# Show plots
plt.show()
```

## Repository Overview

The installable waveform model is located in the [pyEFPEHM/](pyEFPEHM) folder, while the [Tests/](Tests) folder contains various examples and tests of the waveform. Most notably, the [Tests/model_validation/](Tests/model_validation) directory contains the scripts used to generate most of the model-validation figures from [the pyEFPEHM paper](https://arxiv.org/abs/2604.11903), and the [Tests/test_theory/](Tests/test_theory) directory contains Mathematica notebooks and Python scripts used to validate some of the post-Newtonian expressions that the model is based on.

## Feedback & Issues

We welcome feedback, bug reports, and feature requests! If you encounter any issues or have suggestions for improvements, please open an issue in the GitHub issue tracker.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

