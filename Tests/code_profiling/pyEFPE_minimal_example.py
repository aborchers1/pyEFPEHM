#Turn off multithreading
import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

# Import required packages
import pyEFPEHM
import numpy as np

# Define binary parameters (for additional details see pyEFPE/waveform/EFPE.py)
params = {
    'mass1': 2.4,       # Mass of companion 1 (solar masses)
    'mass2': 1.2,       # Mass of companion 2 (solar masses)
    'e_start': 0.5,     # Initial eccentricity
    'spin1x': -0.44,    # Spin components of companion 1
    'spin1y': -0.26,
    'spin1z': 0.48,
    'spin2x': -0.31,    # Spin components of companion 2
    'spin2y': 0.01,
    'spin2z': -0.84,
    'inclination': 1.57,# Initial binary inclination (radians)
    'f22_start': 20,    # Starting (simulation) waveform frequency of GW 22 mode (Hz)
}

# Frequencies and seglen to study
fmin   = 20
fmax   = 1024
seglen = 128

# Initialize pyEFPE waveform model
wf = pyEFPEHM.pyEFPE(params)

# Define frequency array for waveform generation
freqs = np.arange(fmin, fmax, 1./seglen)
# Define time array for waveform generation
times = np.arange(-seglen, 0, 0.5/fmax)

# Compute frequency-domain gravitational wave polarizations
wf.generate_waveform(freqs)

# Compute frequency-domain gravitational wave modes
#wf.generate_modes(freqs)

# Compute time-domain gravitational wave polarizations
#wf.generate_tdomain_waveform(times)

# Compute time-domain gravitational wave modes
#wf.generate_tdomain_modes(times)

