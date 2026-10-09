import numpy as np
from astropy.cosmology import FlatLambdaCDM
from pycbc.waveform import get_td_waveform

# compute the distance for your example redshift
z = 0.02
cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3075)
dL = cosmo.luminosity_distance(z).to_value("Mpc")

# shared binary and sampling parameters
common_td = {
    "approximant": "pyEFPEHM",
    "mass1": 100.0,
    "mass2": 100.0,
    "eccentricity": 0.0,
    "distance": dL,
    "inclination": 0.7,
    "f_lower": 0.05,
    "f_ref": 0.05,
    "f_final": 0.1,
    "delta_t": 0.5,
    "mode_array": [(2, 2, 2)],
}

# generate the vacuum signal
hp_vac, hc_vac = get_td_waveform(
    **common_td,
    environmental_model=None,
)

# generate the signal with BHL dephasing
hp_env, hc_env = get_td_waveform(
    **common_td,
    environmental_model="roemer",
    environmental_redshift=z,
    environmental_phase_sign=1.0,
    tertiary_mass=1.0e5,
    tertiary_distance=1.495978707e11,  # 1 AU
    #gas_density=1e-11,
    #sound_speed=1e4,
)

# check that both signals use matching time grids
assert len(hp_env) == len(hp_vac)
assert hp_env.delta_t == hp_vac.delta_t
assert hp_env.start_time == hp_vac.start_time

# extract both polarizations as ordinary NumPy arrays
h_env = np.vstack((hp_env.numpy(), hc_env.numpy()))
h_vac = np.vstack((hp_vac.numpy(), hc_vac.numpy()))

assert np.all(np.isfinite(h_env))
assert np.all(np.isfinite(h_vac))
assert np.linalg.norm(h_vac) > 0.0

print("Relative waveform difference:", np.linalg.norm(h_env - h_vac) / np.linalg.norm(h_vac))
