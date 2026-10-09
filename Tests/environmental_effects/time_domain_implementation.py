import numpy as np
import pyEFPEHM
from astropy.cosmology import FlatLambdaCDM

# Compute the luminosity distance, as in your earlier example.
z = 0.02
cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3075)
dL = cosmo.luminosity_distance(z).to_value("Mpc")

# Binary parameters, using pyEFPEHM's native frequency names.
params = {
    "mass1": 100.0,
    "mass2": 100.0,
    "eccentricity": 0.0,
    "distance": dL,
    "inclination": 0.7,
    "f22_start": 0.05,
    "f22_ref": 0.05,
    "f22_end": 0.1,
    "environmental_redshift": z,
    "environmental_phase_sign": 1.0,
}

# Environmental configurations to test individually.
AU = 1.495978707e11

environmental_options = {
    "bhl": {
        "environmental_model": "bhl",
        "gas_density": 1e-11,
        "sound_speed": 1e4,
    },
    "supersonic": {
        "environmental_model": "supersonic",
        "gas_density": 1e-7,
    },
    "roemer": {
        "environmental_model": "roemer",
        "tertiary_mass": 1e5,
        "tertiary_distance": 3e4 * AU,
    },
}

# Choose the effect for this test.
model = "bhl"
params.update(environmental_options[model])

# Construct the generator containing the orbital evolution and phase helpers.
waveform = pyEFPEHM.pyEFPE(params)


env = waveform

# Construct the corresponding vacuum generator.
vacuum_params = env.params.copy()
vacuum_params["environmental_model"] = None
vac = pyEFPEHM.pyEFPE(vacuum_params)

# A short interval for your current circular, low-frequency example.
t_start = max(env.return_start_time(), vac.return_start_time())
t_end = min(
    env.return_end_time(),
    vac.return_end_time(),
    t_start + 1024.0,
)
times = np.arange(t_start, t_end, 0.5)

h_env = env.generate_tdomain_waveform(times=times)
h_vac = vac.generate_tdomain_waveform(times=times)

assert np.all(np.isfinite(h_env))
assert np.all(np.isfinite(h_vac))

print(
    "Relative waveform difference:",
    np.linalg.norm(h_env - h_vac) / np.linalg.norm(h_vac),
)

# Check that summing the individual modes reproduces the waveform.
pieces = env.generate_tdomain_modes(
    times=times,
    return_waveform_pieces=True,
)

h_sum = np.zeros_like(h_env)

for piece in pieces["modes"].values():
    h_sum[:, piece["time_idxs"]] += piece["polarizations"]

np.testing.assert_allclose(
    h_sum,
    h_env,
    rtol=1e-8,
    atol=1e-8 * np.max(np.abs(h_env)),
)

print("Mode-sum consistency check passed.")
