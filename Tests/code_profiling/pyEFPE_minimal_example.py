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
}

# Frequencies and seglen to study
fmin   = 20
fmax   = 1024
seglen = 64

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

