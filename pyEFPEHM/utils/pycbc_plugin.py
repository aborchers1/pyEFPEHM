import numpy as np
from pycbc.types import FrequencySeries, TimeSeries
from pyEFPEHM.waveform.EFPE import pyEFPE


def _pyefpe_parameters_from_pycbc(params):
    """Translate pyCBC names into pyEFPEHM parameter names."""

    # get pyCBC's final frequency
    f_final = params.get("f_final", 0.0)

    # check that the user introduces a positive final frequency
    if f_final is None or f_final <= 0.0:
        raise ValueError("The pyEFPEHM PyCBC plugin requires a positive f_final.")

    # get pyCBC's reference frequency
    f_ref = params.get("f_ref", 0.0)
    # interpret pyCBC's zero reference frequency as pyEFPEHM's default
    if f_ref == 0.0:
        f_ref = None

    # construct parameter dictionary for pyEFPEHM copying the values from the pycbc parameter dictionary
    pyefpe_params = {
        "mass1": params["mass1"],
        "mass2": params["mass2"],
        "spin1x": params.get("spin1x", 0.0),
        "spin1y": params.get("spin1y", 0.0),
        "spin1z": params.get("spin1z", 0.0),
        "spin2x": params.get("spin2x", 0.0),
        "spin2y": params.get("spin2y", 0.0),
        "spin2z": params.get("spin2z", 0.0),
        "eccentricity": params.get("eccentricity", 0.0),
        "distance": params.get("distance", 1.0),
        "inclination": params.get("inclination", 0.0),

        # pyCBC-to-pyEFPEHM name translations
        "phase": params.get("coa_phase", 0.0),
        "mean_anomaly": params.get("mean_per_ano", 0.0),
        "f22_start": params["f_lower"],
        "f22_ref": f_ref,
        "f22_end": f_final,

        # environmental parameters
        "environmental_model": params.get("environmental_model", None),
        "environmental_redshift": params.get("environmental_redshift", 0.0),
        "environmental_phase_sign": params.get("environmental_phase_sign", 1.0),
        "tertiary_mass": params.get("tertiary_mass", 0.0),
        "tertiary_distance": params.get("tertiary_distance", 0.0),
        "gas_density": params.get("gas_density", 0.0),
        "sound_speed": params.get("sound_speed", 0.0),
    }

    # forward an optional duration limit to pyEFPEHM
    if params.get("max_duration") is not None:
        pyefpe_params["max_duration"] = params["max_duration"]
    
    # get values from pyCBC's mode-selection array
    mode_array = params.get("mode_array")

    # process the mode array only when the user provided one
    if mode_array is not None:
        # determine whether at least one entry has an eccentric harmonic index
        contains_harmonics = any(len(mode) == 3 for mode in mode_array)

        # determine whether every entry has an eccentric harmonic index
        all_have_harmonics = all(len(mode) == 3 for mode in mode_array)

        # check whether ordinary modes and harmonic modes were mixed
        if contains_harmonics and not all_have_harmonics:
            raise ValueError("Do not mix (l, m) and (l, m, n) entries in mode_array.")

        # handle arrays whose entries contain (l, m, n)
        if all_have_harmonics:
            # copy the complete harmonic selection 
            pyefpe_params["harmonic_array"] = [list(mode) for mode in mode_array]

            # extract the unique (l, m) multipoles from the harmonic array
            unique_lm_modes = {(mode[0], mode[1]) for mode in mode_array}

            # sort and copy the unique multipoles into pyEFPEHM's mode array.
            pyefpe_params["mode_array"] = [list(mode) for mode in sorted(unique_lm_modes)]

        # handle ordinary mode arrays containing only (l, m)
        else:
            # copy the ordinary multipole selection into pyEFPEHM
            pyefpe_params["mode_array"] = [list(mode) for mode in mode_array]

    return pyefpe_params


def pyefpe_fd(**params):
    """Generate summed FD polarizations for pyCBC."""
    
    # get frequency parameters
    delta_f = float(params["delta_f"])
    f_lower = float(params["f_lower"])
    f_final = float(params["f_final"])

    # ensure that the frequency sampling size is physically meaningful
    if delta_f <= 0.0:
        raise ValueError("delta_f must be positive.")

    # ensure that the frequency interval is physically meaningful
    if f_final <= f_lower:
        raise ValueError("f_final must be greater than f_lower.")

    # pyCBC FrequencySeries objects start at zero frequency
    frequencies = np.arange(0.0, f_final + 0.5 * delta_f, delta_f)

    # identify the frequency bins at or above the waveform's lower cutoff
    active = frequencies >= f_lower
    
    # extract only frequencies at which pyEFPEHM should be evaluated
    active_frequencies = frequencies[active]

    # allocate the plus and cross polarization arrays with zeros
    hp = np.zeros(len(frequencies), dtype=np.complex128)
    hc = np.zeros(len(frequencies), dtype=np.complex128)

    # translate pyCBC's arguments into a pyEFPEHM parameter dictionary.
    pyefpe_params = _pyefpe_parameters_from_pycbc(params)

    # initialise the pyEFPEHM waveform
    waveform = pyEFPE(pyefpe_params)

    # generate the plus and cross polarizations
    hp_active, hc_active = waveform.generate_waveform(active_frequencies)

    # insert the polarization values into hp and hc, which include the complete frequency grid
    hp[active] = hp_active
    hc[active] = hc_active

    return (FrequencySeries(hp, delta_f=delta_f), FrequencySeries(hc, delta_f=delta_f))


def pyefpe_td(**params):
    """Generate summed TD polarizations for pyCBC"""

    # read and validate the requested time spacing
    delta_t = float(params["delta_t"])

    if not np.isfinite(delta_t) or delta_t <= 0.0:
        raise ValueError("delta_t must be finite and positive.")

    # reuse the parameter conversion used by the FD adapter
    pyefpe_params = _pyefpe_parameters_from_pycbc(params)

    # validate the requested frequency interval
    f_lower = float(pyefpe_params["f22_start"])
    f_final = float(pyefpe_params["f22_end"])

    if not np.isfinite(f_lower) or f_lower <= 0.0:
        raise ValueError("f_lower must be finite and positive.")

    if not np.isfinite(f_final) or f_final <= f_lower:
        raise ValueError("f_final must be finite and greater than f_lower.")

    # construct the generator containing your environmental TD corrections
    waveform = pyEFPE(pyefpe_params)

    # evaluate the time-domain polarizations on a uniform time grid
    polarizations, times = waveform.generate_tdomain_waveform(delta_t=delta_t, return_time_array=True)

    # ensure that a nonempty waveform was produced
    if len(times) == 0:
        raise ValueError("No time samples were generated.")

    # preserve pyEFPEHM's existing time origin
    epoch = float(times[0])

    # wrap the plus polarization in a PyCBC TimeSeries
    hp = TimeSeries(np.asarray(polarizations[0], dtype=np.float64), delta_t=delta_t, epoch=epoch)

    # wrap the cross polarization using the same sampling and epoch
    hc = TimeSeries(np.asarray(polarizations[1], dtype=np.float64), delta_t=delta_t, epoch=epoch)

    return hp, hc


# tell PyCBC which arguments must be provided
pyefpe_td.required = ["mass1", "mass2", "f_lower", "delta_t", "approximant"]