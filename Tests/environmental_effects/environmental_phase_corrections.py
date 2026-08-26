import numpy as np
import pytest

import pyEFPEHM


def base_parameters():
    """Simple vacuum binary configuration for testing the new code"""
    return {
        "mass1": 10.0,
        "mass2": 5.0,
        "eccentricity": 0.1,
        "f22_start": 0.01,
        "f22_end": 0.1,
        "distance": 100.0,

        # Keep the test reasonably fast and explicitly select harmonics.
        "harmonic_array": [
            [2, 2, 2],
            [2, 2, 3],
        ],

        "environmental_model": None,
        "environmental_redshift": 0.0,
        "environmental_phase_sign": 1.0,

        "tertiary_mass": 0.0,
        "tertiary_distance": 0.0,
        "gas_density": 0.0,
        "sound_speed": 0.0,
    }


def make_waveform(**updates):
    """Construct a waveform after updating the base parameters."""
    params = base_parameters()
    params.update(updates)
    return pyEFPEHM.pyEFPE(params)


# --------- 1. Test vacuum recovery ---------
    
def test_environmental_model_none_recovers_vacuum():
    """Test whether the default model and the explicit 'environmental_model=None' produce identical waveforms"""
    freqs = np.linspace(20.0, 80.0, 128)

    params_default = base_parameters()
    params_default.pop("environmental_model")

    params_explicit_none = base_parameters()
    params_explicit_none["environmental_model"] = None

    waveform_default = pyEFPEHM.pyEFPE(params_default)
    waveform_none = pyEFPEHM.pyEFPE(params_explicit_none)

    h_default = waveform_default.generate_waveform(freqs)
    h_none = waveform_none.generate_waveform(freqs)

    np.testing.assert_allclose(h_none, h_default, rtol=0.0, atol=0.0)


def test_no_environment_returns_zero_phase():
    """Test whether the environmental helper returns zero with 'environmental_model=None' """
    waveform = make_waveform(environmental_model=None)

    f22 = np.array([20.0, 30.0, 40.0])
    e = np.array([0.1, 0.08, 0.05])

    correction = waveform._delta_psi2_environment(f22, e)

    np.testing.assert_array_equal(
        correction,
        np.zeros_like(f22),
    )


# --------- 2. Test every environmental effect ---------

def test_roemer_ell2_scalings():
    """Test whether the Roemer delay function returns finite positive values, has the correct frequency scaling and has the correct environmental-parameter scalings"""
    waveform = make_waveform(
        environmental_model="roemer",
        tertiary_mass=1.0e5,
        tertiary_distance=1.495978707e11,  # 1 AU
    )

    e = np.array([0.1])
    f = np.array([20.0])

    phase_1 = waveform._delta_psi2_roemer(f, e)

    assert phase_1.shape == f.shape
    assert np.all(np.isfinite(phase_1))
    assert np.all(phase_1 > 0.0)

    # Frequency scaling: f^(-13/3)
    phase_2f = waveform._delta_psi2_roemer(2.0 * f, e)

    np.testing.assert_allclose(
        phase_2f / phase_1,
        2.0**(-13.0 / 3.0),
        rtol=1.0e-13,
    )

    # Tertiary-mass scaling: proportional to m3
    original_mass = waveform.params["tertiary_mass"]
    waveform.params["tertiary_mass"] = 2.0 * original_mass

    phase_2m = waveform._delta_psi2_roemer(f, e)

    np.testing.assert_allclose(
        phase_2m / phase_1,
        2.0,
        rtol=1.0e-13,
    )

    # Restore the parameter
    waveform.params["tertiary_mass"] = original_mass

    # Separation scaling: proportional to R^(-2)
    original_distance = waveform.params["tertiary_distance"]
    waveform.params["tertiary_distance"] = 2.0 * original_distance

    phase_2R = waveform._delta_psi2_roemer(f, e)

    np.testing.assert_allclose(
        phase_2R / phase_1,
        0.25,
        rtol=1.0e-13,
    )


def test_bhl_ell2_scalings():
    """Test whether the Bondi-Hoyle-Littleton drag function returns finite positive values, has the correct frequency scaling and has the correct environmental-parameter scalings"""
    waveform = make_waveform(
        environmental_model="bhl",
        gas_density=1.0e-7,
        sound_speed=1.0e4,
    )

    e = np.array([0.1])
    f = np.array([20.0])

    phase_1 = waveform._delta_psi2_bhl(f, e)

    assert phase_1.shape == f.shape
    assert np.all(np.isfinite(phase_1))
    assert np.all(phase_1 > 0.0)

    # Frequency scaling: f^(-14/3)
    phase_2f = waveform._delta_psi2_bhl(2.0 * f, e)

    np.testing.assert_allclose(
        phase_2f / phase_1,
        2.0**(-14.0 / 3.0),
        rtol=1.0e-13,
    )

    # Density scaling: proportional to rho
    original_density = waveform.params["gas_density"]
    waveform.params["gas_density"] = 2.0 * original_density

    phase_2rho = waveform._delta_psi2_bhl(f, e)

    np.testing.assert_allclose(
        phase_2rho / phase_1,
        2.0,
        rtol=1.0e-13,
    )

    waveform.params["gas_density"] = original_density

    # Sound-speed scaling: proportional to cs^(-2)
    original_sound_speed = waveform.params["sound_speed"]
    waveform.params["sound_speed"] = 2.0 * original_sound_speed

    phase_2cs = waveform._delta_psi2_bhl(f, e)

    np.testing.assert_allclose(
        phase_2cs / phase_1,
        0.25,
        rtol=1.0e-13,
    )


def test_supersonic_ell2_scalings():
    """Test whether the supersonic drag function returns finite positive values, has the correct frequency scaling and has the correct environmental-parameter scalings"""
    waveform = make_waveform(
        environmental_model="supersonic",
        gas_density=1.0e-7,
    )

    e = np.array([0.1])
    f = np.array([20.0])

    phase_1 = waveform._delta_psi2_supersonic(f, e)

    assert phase_1.shape == f.shape
    assert np.all(np.isfinite(phase_1))
    assert np.all(phase_1 > 0.0)

    # Frequency scaling: f^(-16/3)
    phase_2f = waveform._delta_psi2_supersonic(2.0 * f, e)

    np.testing.assert_allclose(
        phase_2f / phase_1,
        2.0**(-16.0 / 3.0),
        rtol=1.0e-13,
    )

    # Density scaling: proportional to rho
    original_density = waveform.params["gas_density"]
    waveform.params["gas_density"] = 2.0 * original_density

    phase_2rho = waveform._delta_psi2_supersonic(f, e)

    np.testing.assert_allclose(
        phase_2rho / phase_1,
        2.0,
        rtol=1.0e-13,
    )

# --------- 3. Test mapping from dominant eccentric harmonic to higher harmonics ---------
    
@pytest.mark.parametrize(
    "model, extra_parameters",
    [
        (
            "roemer",
            {
                "tertiary_mass": 1.0e5,
                "tertiary_distance": 1.495978707e11,
            },
        ),
        (
            "bhl",
            {
                "gas_density": 1.0e-7,
                "sound_speed": 1.0e4,
            },
        ),
        (
            "supersonic",
            {
                "gas_density": 1.0e-7,
            },
        ),
    ],
)
def test_mapping_to_higher_eccentric_harmonics(model, extra_parameters):
    waveform = make_waveform(
        environmental_model=model,
        **extra_parameters,
    )

    f_detector = np.array([0.01, 0.05, 0.1])
    e = np.array([0.1, 0.1, 0.1])
    ell = np.array([2, 3, 5])

    calculated = waveform._environmental_harmonic_phase(
        f_detector,
        e,
        ell,
    )

    f22_equivalent = 2.0 * f_detector / ell

    expected = (
        0.5
        * ell
        * waveform._delta_psi2_environment(
            f22_equivalent,
            e,
        )
    )

    np.testing.assert_allclose(
        calculated,
        expected,
        rtol=1.0e-13,
        atol=0.0,
    )

def test_ell2_mapping_recovers_base_phase():
    waveform = make_waveform(
        environmental_model="bhl",
        gas_density=1.0e-7,
        sound_speed=1.0e4,
    )

    f = np.array([0.01, 0.05, 0.1])
    e = np.array([0.1, 0.08, 0.05])
    ell = np.full(f.shape, 2)

    harmonic_phase = waveform._environmental_harmonic_phase(
        f,
        e,
        ell,
    )

    base_phase = waveform._delta_psi2_environment(f, e)

    np.testing.assert_allclose(
        harmonic_phase,
        base_phase,
        rtol=1.0e-13,
        atol=0.0,
    )

def test_zero_harmonic_receives_no_correction():
    waveform = make_waveform(
        environmental_model="bhl",
        gas_density=1.0e-7,
        sound_speed=1.0e4,
    )

    f = np.array([20.0, 30.0])
    e = np.array([0.1, 0.1])
    ell = np.array([0, 0])

    correction = waveform._environmental_harmonic_phase(
        f,
        e,
        ell,
    )

    np.testing.assert_array_equal(
        correction,
        np.zeros_like(f),
    )


# --------- 4. Test complete waveform generation ---------

@pytest.mark.parametrize(
    "model, extra_parameters",
    [
        (
            "roemer",
            {
                "tertiary_mass": 1.0e5,
                "tertiary_distance": 1.495978707e11,
            },
        ),
        (
            "bhl",
            {
                "gas_density": 1.0e-7,
                "sound_speed": 1.0e4,
            },
        ),
        (
            "supersonic",
            {
                "gas_density": 1.0e-7,
            },
        ),
    ],
)

def test_complete_environmental_waveform_generation(model, extra_parameters):
    """Test all three models run through 'stationary_times()', harmonic correction, SUA amplitudes, and mode summation without errors or NANs"""
    freqs = np.linspace(0.01, 0.1, 128)

    waveform = make_waveform(
        environmental_model=model,
        **extra_parameters,
    )

    hplus, hcross = waveform.generate_waveform(freqs)

    assert hplus.shape == freqs.shape
    assert hcross.shape == freqs.shape

    assert np.all(np.isfinite(hplus))
    assert np.all(np.isfinite(hcross))

    # A valid waveform should contain at least some nonzero samples.
    assert np.any(np.abs(hplus) + np.abs(hcross) > 0.0)


@pytest.mark.parametrize(
    "model, extra_parameters",
    [
        (
            "roemer",
            {
                "tertiary_mass": 1.0e5,
                "tertiary_distance": 1.495978707e11,
            },
        ),
        (
            "bhl",
            {
                "gas_density": 1.0e-7,
                "sound_speed": 1.0e4,
            },
        ),
        (
            "supersonic",
            {
                "gas_density": 1.0e-7,
            },
        ),
    ],
)

def test_complete_mode_generation(model, extra_parameters):
    """Test whether 'generate_modes()' produces corrected 'psi_SPA' arrays"""
    freqs = np.linspace(0.01, 0.1, 128)

    waveform = make_waveform(
        environmental_model=model,
        **extra_parameters,
    )

    result = waveform.generate_modes(
        freqs,
        return_waveform_pieces=True,
    )

    assert len(result["modes"]) > 0

    for mode_label, mode in result["modes"].items():
        assert np.all(np.isfinite(mode["polarizations"]))
        assert np.all(np.isfinite(mode["psi_SPA"]))
        assert np.all(np.isfinite(mode["t_SPA"]))
        assert np.all(np.isfinite(mode["T_SPA"]))