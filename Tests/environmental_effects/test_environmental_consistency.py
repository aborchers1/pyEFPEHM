"""Environmental derivatives and the optional TD-consistent SPA construction.

Run: python -m pytest Tests/test_environmental_consistency.py
"""
import numpy as np
import pytest

import pyEFPEHM
from pyEFPEHM.waveform.plugins import native_parameters


BASE = dict(mass1=150, mass2=100, eccentricity=0.05, inclination=0.9,
            f22_start=0.09, f22_end=0.1, pn_amplitude_order=0,
            harmonic_array=[[2, 2, 2]], Compute_hlm_Modes=True,
            environmental_model='bhl', gas_density=1e-7, sound_speed=1e4,
            tertiary_mass=1e5, tertiary_distance=1e14)


@pytest.mark.parametrize('model', ['roemer', 'bhl', 'supersonic'])
@pytest.mark.parametrize('eccentricity', [0, 0.05, 0.3])
def test_analytic_derivatives_match_phase_finite_differences(model, eccentricity):
    waveform = pyEFPEHM.pyEFPE(dict(BASE, environmental_model=model, eccentricity=eccentricity))
    start, end = waveform.return_start_time(), waveform.return_end_time()
    times = np.linspace(start+0.2*(end-start), end-0.2*(end-start), 7)
    step = (end-start)*0.001
    f = waveform._environmental_tdomain_orbital_phase
    fm2, fm1, f0, fp1, fp2 = [f(times+k*step) for k in (-2, -1, 0, 1, 2)]
    first = (fm2-8*fm1+8*fp1-fp2)/(12*step)
    second = (-fm2+16*fm1-30*f0+16*fp1-fp2)/(12*step**2)
    np.testing.assert_allclose(f(times, 1), first, rtol=2e-6, atol=1e-20)
    np.testing.assert_allclose(f(times, 2), second, rtol=2e-5, atol=1e-22)


@pytest.mark.parametrize('mass1', [100, 150])
@pytest.mark.parametrize('eccentricity', [0.05, 0.1])
@pytest.mark.parametrize('precessing', [False, True])
def test_no_roundoff_induced_chirp_reversal_at_initial_rk_knots(mass1, eccentricity, precessing):
    params = dict(BASE, mass1=mass1, eccentricity=eccentricity)
    if precessing:
        params.update(spin1x=0.2, spin1z=0.15, spin2y=0.15)
    waveform = pyEFPEHM.pyEFPE(params)
    times = waveform.sol.all_ts[:7]
    curvature = 2*waveform.sol(times, derivative=2, idxs=[2])[0]
    correction = 2*waveform._environmental_tdomain_orbital_phase(times, 2)
    assert np.all(np.isfinite(correction))
    # Actual correction in this band is about 0.6%, not the former endpoint spike.
    assert np.max(np.abs(correction/curvature)) < 0.02
    assert np.all(curvature+correction > 0)


@pytest.mark.parametrize('phase_sign', [-1, 1])
@pytest.mark.parametrize('harmonic', [[2, 2, 2], [2, 2, 3], [2, 0, 2], [2, 2, -2]])
def test_stationary_point_phase_curvature_and_projection_contract(phase_sign, harmonic):
    waveform = pyEFPEHM.pyEFPE(dict(BASE, harmonic_array=[harmonic], SUA_kmax=3,
                                   environmental_phase_sign=phase_sign,
                                   environmental_fd_method='stationary_phase'))
    freqs = np.arange(0.02, 0.3, 2e-5)
    fidx, interp, times, phase, scale = waveform.stationary_times(freqs)
    assert len(fidx) > 32
    omega = waveform._corrected_mode_phase(times, interp, 1)
    curvature = waveform._corrected_mode_phase(times, interp, 2)
    np.testing.assert_allclose(omega, 2*np.pi*freqs[fidx], rtol=1e-12)
    np.testing.assert_allclose(scale, 1/np.sqrt(curvature), rtol=1e-14)
    np.testing.assert_allclose(phase, omega*times-waveform._corrected_mode_phase(times, interp)-np.pi/4, atol=2e-7)
    modes = waveform.generate_hlm_modes(freqs)
    actual = waveform.reconstruct_polarizations_from_native_modes(modes)
    np.testing.assert_allclose(actual, waveform.generate_waveform(freqs), rtol=1e-11, atol=1e-35)
    td_modes = waveform.generate_tdomain_hlm_modes(delta_t=4)
    actual_td = waveform.reconstruct_polarizations_from_native_modes(td_modes)
    reference_td = waveform.generate_tdomain_waveform(td_modes['times'])
    assert np.linalg.norm(actual_td-reference_td)/np.linalg.norm(reference_td) < 1e-10


def test_stationary_method_rejects_genuine_nonmonotonic_carrier():
    waveform = pyEFPEHM.pyEFPE(dict(BASE, gas_density=1e-3, environmental_fd_method='stationary_phase'))
    with pytest.raises(ValueError, match='turning points'):
        waveform.stationary_times([0.095])


def test_vacuum_and_legacy_defaults_are_unchanged():
    a = pyEFPEHM.pyEFPE(dict(BASE, environmental_model=None, environmental_fd_method='phase_only'))
    b = pyEFPEHM.pyEFPE(dict(BASE, environmental_model=None, environmental_fd_method='stationary_phase'))
    for actual, expected in zip(a.stationary_times([0.095, 0.096]), b.stationary_times([0.095, 0.096])):
        np.testing.assert_array_equal(actual, expected)
    assert a._environmental_tdomain_orbital_phase(np.array([]), 2).size == 0


def test_weak_field_difference_from_phase_only_is_second_order():
    freqs = np.array([0.094, 0.095, 0.096])
    differences = []
    for density in (1e-9, 2e-9):
        legacy = pyEFPEHM.pyEFPE(dict(BASE, gas_density=density, environmental_fd_method='phase_only',))
        consistent = pyEFPEHM.pyEFPE(dict(BASE, gas_density=density, environmental_fd_method='stationary_phase'))
        differences.append(np.linalg.norm(consistent.stationary_times(freqs)[3]-legacy.stationary_times(freqs)[3]))
    assert all(np.isfinite(value) and value > 0 for value in differences)
    assert 3.9 < differences[1] / differences[0] < 4.1


def test_parameter_translation_consumes_aliases_and_handles_kerr_deviations():
    original = dict(mass1=100, mass2=100, f_lower=0.09, f_ref=0.09,
                    f_final=0, coa_phase=0.4, mean_per_ano=0.3,
                    dquad_mon1=0, dquad_mon2=0.2, delta_t=2, delta_f=1e-4,
                    approximant='pyEFPEHM', environmental_fd_method='stationary_phase')
    native = native_parameters(original)
    assert native['q1'] == 1
    assert native['q2'] == 1.2
    assert native['phase'] == 0.4
    assert native['mean_anomaly'] == 0.3
    assert native['environmental_fd_method'] == 'stationary_phase'
    assert 'f22_end' not in native
    assert 'dquad_mon1' not in native
    assert 'delta_t' not in native
    assert original['f_lower'] == 0.09


def test_phase_shapes_outside_domain_and_invalid_inputs():
    waveform = pyEFPEHM.pyEFPE(BASE)
    start = waveform.return_start_time()
    assert np.isnan(waveform._environmental_tdomain_orbital_phase(start-1))
    assert waveform._environmental_tdomain_orbital_phase(start).shape == ()
    assert waveform._environmental_tdomain_orbital_phase(np.array([[start, start+1]])).shape == (1, 2)
    with pytest.raises(ValueError, match='derivative'):
        waveform._environmental_tdomain_orbital_phase(start, 3)
    with pytest.raises(ValueError, match='environmental_fd_method'):
        pyEFPEHM.pyEFPE(dict(BASE, environmental_fd_method='typo'))


def test_shared_endpoint_is_not_counted_twice_and_empty_support_is_typed():
    waveform = pyEFPEHM.pyEFPE(dict(BASE, environmental_fd_method='stationary_phase'))
    idx = 10
    time = waveform.sol.all_ts[waveform.mode_interp_idx[idx]]
    freq = waveform._corrected_mode_phase(np.array([time]), np.array([idx]), 1)[0]/(2*np.pi)
    result = waveform.stationary_times([freq])
    assert len(result[0]) == 1
    empty = waveform.stationary_times([1000.0])
    assert all(len(values) == 0 for values in empty)
    assert empty[0].dtype.kind == 'i' and empty[1].dtype.kind == 'i'
