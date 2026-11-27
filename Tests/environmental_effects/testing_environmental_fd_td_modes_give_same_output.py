#!/usr/bin/env python3
"""Check that the TD and FD mode implementations give the same output. 
We make this test by inverse-transforming the FD mode and comparing it to the 
(directly called) TD mode.

Dependencies: numpy, scipy, and the environmental/mode branch of pyEFPEHM.
Optional plots also need matplotlib.

Examples:
    python testing_environmental_fd_td_modes_give_same_output.py --check
    python testing_environmental_fd_td_modes_give_same_output.py --harmonic 2 2 3 --check
    python testing_environmental_fd_td_modes_give_same_output.py --model roemer --plot comparison.png

The waveform tolerance defaults to 1% and the environmental-residual
tolerance to 5%. These are adjustable diagnostic choices, not universal
accuracy guarantees or noise-weighted detector mismatches.
The metrics use the central half of the trajectory and both polarizations.
No fitted time, phase, or amplitude alignment is performed. Each waveform
uses its own FD support and stationary-time taper on the same padded FFT grid.
The first-order phase-prescription comparison is diagnostic only; it need not
vanish for the stationary_phase FD construction.
""" 
import argparse  # handle command-line arguments and help messages
import sys  # report errors to stderr and return process exit codes

import numpy as np  # use NumPy for arrays, waveform comparisons, and phase calculations
from scipy.fft import irfft, next_fast_len, rfftfreq  # import the real inverse FFT, its frequency grid, and efficient FFT sizes

import pyEFPEHM  # load the waveform package from the active Python environment


def parse_args():  # define and validate the command-line options
    parser = argparse.ArgumentParser(description=__doc__,  # use the module docstring as the command-line description
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # keep the examples and line breaks in the help output
    parser.add_argument("--mass1", type=float, default=100)  # set the first component mass in solar masses
    parser.add_argument("--mass2", type=float, default=100)  # set the second component mass in solar masses
    parser.add_argument("--eccentricity", type=float, default=0.1)  # set the eccentricity at the reference frequency
    parser.add_argument("--f-start", type=float, default=0.05)  # set the starting quadrupole frequency in Hz
    parser.add_argument("--f-end", type=float, default=0.1)  # set the ending quadrupole frequency in Hz
    parser.add_argument("--dt", type=float, default=2)  # set the time sample spacing in seconds
    parser.add_argument("--harmonic", type=int, nargs=3, default=[2, 2, 2],  # select one harmonic using three integers
                        metavar=("L", "M", "N"))  # label the degree, azimuthal number, and eccentric harmonic index
    parser.add_argument("--model", choices=["all", "roemer", "bhl", "supersonic"],  # list the available environmental models
                        default="all")  # run all three models by default
    parser.add_argument("--environmental-fd-method", choices=("phase_only", "stationary_phase"),
                        default="stationary_phase", help="Select the FD environmental construction.")
    parser.add_argument("--sua-kmax", type=int, default=0,
                        help="FD precession correction order; use 3 for precessing comparisons.")
    parser.add_argument("--redshift", type=float, default=0.0)  # set the redshift used by the environmental prescriptions
    parser.add_argument("--phase-sign", type=int, choices=[-1, 1], default=1)  # choose the sign of the phase correction
    parser.add_argument("--tertiary-mass", type=float, default=1.0)  # set the tertiary mass in solar masses
    parser.add_argument("--tertiary-distance", type=float, default=1e14)  # set the tertiary distance in metres
    parser.add_argument("--gas-density", type=float, default=1e-7)  # set the gas density in kilograms per cubic metre
    parser.add_argument("--sound-speed", type=float, default=1e4)  # set the sound speed in metres per second
    parser.add_argument("--target-phase", type=float, default=0.1,  # set the desired phase magnitude near the start, in radians
                        help="Near-initial first-order harmonic phase magnitude; 0 preserves physical inputs.")  # document that zero keeps the supplied physical parameters
    parser.add_argument("--max-samples", type=int, default=2_000_000,  # cap the number of samples in the padded time grid
                        help="Refuse a larger FFT before allocating it.")  # explain that the limit is checked before FFT allocation
    parser.add_argument("--check", action="store_true",  # enable threshold checks for automated runs
                        help="Exit 1 if the selected error thresholds are exceeded.")  # document the failure exit code
    parser.add_argument("--waveform-tolerance", type=float, default=0.01)  # set the relative error limit for the complete waveform
    parser.add_argument("--effect-tolerance", type=float, default=0.05)  # set the relative error limit for the change due to the environment
    parser.add_argument("--plot", metavar="PNG", help="Optionally save diagnostic plots.")  # accept an optional PNG filename for the plots
    args = parser.parse_args()  # parse the options supplied on the command line
    if args.sua_kmax < 0:
        parser.error("sua-kmax must be nonnegative.")
    numbers = [args.mass1, args.mass2, args.f_start, args.f_end, args.dt,  # collect the quantities that must be positive and finite
               args.tertiary_mass, args.tertiary_distance, args.sound_speed,  # include the tertiary and gas parameters that must be positive
               args.waveform_tolerance, args.effect_tolerance]  # apply the same requirement to both error tolerances
    if not all(np.isfinite(x) and x > 0 for x in numbers):  # check every quantity in the list
        parser.error("Masses, frequencies, cadence, distances, sound speed and tolerances must be positive and finite.")  # report invalid positive-valued inputs through the argument parser
    if args.f_end <= args.f_start or not 0 <= args.eccentricity < 1:  # check the frequency ordering and the allowed eccentricity range
        parser.error("Require f-end > f-start and 0 <= eccentricity < 1.")  # report the required frequency ordering and eccentricity bounds
    if not np.isfinite(args.target_phase) or args.target_phase < 0:  # check that the phase target is finite and nonnegative
        parser.error("target-phase must be finite and nonnegative.")  # report an invalid phase target
    if not np.isfinite(args.gas_density) or args.gas_density <= 0:  # check that the gas density is finite and positive
        parser.error("gas-density must be positive and finite.")  # report an invalid gas density
    if not np.isfinite(args.redshift) or args.redshift < 0 or args.max_samples < 16:  # validate the redshift and the minimum FFT sample limit
        parser.error("Require nonnegative finite redshift and max-samples >= 16.")  # report which bounds those two inputs must satisfy
    l, m, n = args.harmonic  # unpack the harmonic label into its three indices
    if [l, m] not in [[2, 0], [2, 2]] or n < 1:  # check which multipoles and harmonic indices this 0PN test supports
        parser.error("This 0PN diagnostic supports (l,m)=(2,0) or (2,2), with n >= 1.")  # report the allowed 0PN harmonic labels
    return args  # return the validated options


def taper(times, start, end):  # define the endpoint taper on the supplied time grid
    """Sin-squared ramps over each outer 15% of the vacuum trajectory."""  # describe the taper shape and width
    fraction = (np.asarray(times) - start) / (end - start)  # express each time as a fraction of the waveform duration
    rise = np.sin(0.5 * np.pi * np.clip(fraction / 0.15, 0, 1)) ** 2  # ramp up smoothly over the first 15 percent
    fall = np.sin(0.5 * np.pi * np.clip((1 - fraction) / 0.15, 0, 1)) ** 2  # ramp down smoothly over the last 15 percent
    return rise * fall  # combine both ramps into one window


def compare(candidate, reference, mask):  # compare two waveforms on the selected time samples
    a, b = candidate[:, mask].ravel(), reference[:, mask].ravel()  # flatten the selected samples across both polarizations
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)  # measure the amplitude of each waveform with its norm
    if not np.isfinite(norm_a + norm_b) or norm_a == 0 or norm_b == 0:  # check that both norms can serve as divisors
        raise ValueError("Zero waveform or environmental residual; comparison is undefined.")  # stop if a waveform has no signal in the comparison window
    error = np.linalg.norm(a - b) / norm_b  # normalize the difference by the reference waveform norm
    overlap = np.clip(np.dot(a, b) / (norm_a * norm_b), -1, 1)  # compute the normalized inner product and clip roundoff at its bounds
    if not np.isfinite(error) or not np.isfinite(overlap):
        raise ValueError("Nonfinite comparison metric.")
    return float(error), float(1 - overlap)  # return the relative error and the unweighted mismatch


def fd_to_td(waveform, freqs, count, dt, origin, start, end, key):
    """Build positive-frequency polarizations and invert with correct units."""  
    spectrum = np.zeros((2, len(freqs)), dtype=complex)  # allocate two one-sided spectra and leave DC and Nyquist at zero
    pieces = waveform.generate_modes(freqs[1:-1], return_waveform_pieces=True)  # generate positive-frequency bins and keep the stationary-time metadata
    if set(pieces["modes"]) != {key}:  # verify that the FD result contains exactly the selected harmonic
        raise ValueError(f"Expected just {key}, got {list(pieces['modes'])}.")  # report the returned labels if the selection did not match
    mode = pieces["modes"][key]  # retrieve the spectrum and metadata for the chosen harmonic
    if len(mode["freq_idxs"]) < 32:  # require enough active bins for a meaningful inverse transform
        raise ValueError("Fewer than 32 active frequency bins; choose a longer chirp.")  # explain how to obtain finer frequency resolution
    indices = 1 + mode["freq_idxs"]  # offset the bin indices because the input excluded DC
    weights = taper(mode["t_SPA"], start, end)  # evaluate the time taper at each bin's stationary time
    for polarization in range(2):  # process the plus and cross polarizations separately
        np.add.at(spectrum[polarization], indices,  # accumulate contributions that share a frequency bin
                  mode["polarizations"][polarization] * weights)  # weight each contribution before adding it to the spectrum
    spectrum *= np.exp(2j * np.pi * freqs * origin)  # shift the inverse transform to the requested time origin
    return irfft(spectrum, n=count, axis=-1) / dt, mode  # reconstruct real polarizations, restore strain units, and return the mode metadata


def environment(args, model):  # build model parameters and identify the parameter used to scale the correction
    if model == "roemer":  # handle the model driven by the tertiary mass and distance
        return dict(environmental_model=model, tertiary_mass=args.tertiary_mass,  # pass the requested tertiary mass to the Roemer model
                    tertiary_distance=args.tertiary_distance), "tertiary_mass"  # pass the distance and identify the mass as the adjustable strength
    options = dict(environmental_model=model, gas_density=args.gas_density)  # initialize the shared density input for the gas models
    if model == "bhl":  # check whether the model needs a sound speed
        options["sound_speed"] = args.sound_speed  # include the sound speed in the BHL parameters
    return options, "gas_density"  # return the gas parameters with density as the adjustable strength


def main():  # run the waveform comparison for the selected models
    args = parse_args()  # read and validate the command-line inputs
    key = tuple(args.harmonic)  # make the harmonic label usable as a dictionary key
    base = dict(mass1=args.mass1, mass2=args.mass2,  # begin the common vacuum and environmental parameter set with the masses
                eccentricity=args.eccentricity, distance=100.0, inclination=0.9,  # set the eccentricity, luminosity distance in Mpc, and inclination in radians
                f22_start=args.f_start, f22_ref=args.f_start, f22_end=args.f_end,  # set the reference frequency to the start of the quadrupole interval
                harmonic_array=[list(key)], pn_amplitude_order=0, SUA_kmax=args.sua_kmax,  # select one harmonic and the requested precession correction
                environmental_redshift=args.redshift,  # pass the redshift used for the environmental correction
                environmental_phase_sign=args.phase_sign,  # include the requested correction sign
                environmental_fd_method=args.environmental_fd_method)
    print("pyEFPEHM source", pyEFPEHM.__file__)  # show which pyEFPEHM installation Python imported
    print("Binary and harmonic", base, flush=True)  # report the common parameters before waveform generation
    vacuum = pyEFPEHM.pyEFPE(base)  # construct the vacuum waveform generator
    start, end = vacuum.return_start_time(), vacuum.return_end_time()  # get the model's start and end times in seconds
    duration = end - start  # measure the duration of the vacuum trajectory
    requested_count = int(np.ceil(1.5 * duration / args.dt))  # estimate the uniform sample count with 50 percent padding
    if requested_count > args.max_samples:  # refuse grids beyond the requested limit before allocating arrays
        raise ValueError(  # begin the error message for an oversized trajectory
            f"Full trajectory is {duration:.6g} s and needs about {requested_count:,} "  # report its duration and estimated sample count
            f"samples before FFT rounding, above --max-samples {args.max_samples:,}. "  # show the limit that triggered the refusal
            "Use a narrower frequency interval or a higher-mass diagnostic binary. "  # suggest configurations that shorten the signal
            "Raise the limit only if the required memory is available.")  # explain when raising the limit is practical
    count = 2 * next_fast_len(max(8, int(np.ceil(requested_count / 2))), real=True)  # round up to an efficient even FFT length to retain a Nyquist bin
    if count > args.max_samples:  # check whether FFT rounding pushed the count over the cap
        raise ValueError(f"Rounded FFT size {count:,} exceeds --max-samples.")  # report the final count if it exceeds the limit
    origin = start - 0.25 * duration  # leave a quarter of the signal duration as leading padding
    times = origin + np.arange(count) * args.dt  # build the uniform time grid from the padded origin
    freqs = rfftfreq(count, args.dt)  # build the matching nonnegative FFT frequency grid in Hz
    interior = (times >= start + 0.25 * duration) & (times <= end - 0.25 * duration)  # select the central half of the trajectory for comparison
    if np.count_nonzero(interior) < 64:  # check that the comparison window contains at least 64 samples
        raise ValueError("Too few interior time samples; decrease --dt.")  # request a finer cadence if the comparison window is undersampled
    time_weights = taper(times, start, end)  # evaluate the endpoint taper on the full time grid
    print(f"Duration {duration:.6g} s; FFT samples {count:,}; df {freqs[1]:.6g} Hz")  # report the trajectory duration, FFT length, and bin spacing
    print("Metrics use both polarizations, the central 50%, and no fitted alignment.")  # describe the comparison window and alignment settings

    probe_times = np.linspace(start, end, 1025)  # build a smaller time grid for inspecting the instantaneous frequency
    vacuum_modes = vacuum.generate_tdomain_modes(probe_times, return_waveform_pieces=True)  # obtain the vacuum mode and its phase derivatives at those times
    if set(vacuum_modes["modes"]) != {key}:  # verify that the mode generator kept the requested harmonic
        raise ValueError(f"Requested harmonic {key} is absent or was replaced.")  # report a missing or replaced harmonic
    # Normalize strength at a vacuum orbital epoch. A fixed frequency near
    # the vacuum band edge can lie outside the environmental FD support.
    reference_time = probe_times[1]

    def evaluate(waveform):  # evaluate one waveform in both domains on the common grids
        mode = waveform.generate_tdomain_modes(  # probe the instantaneous phase derivatives for this model
            probe_times, return_waveform_pieces=True)["modes"][key]  # retain the selected harmonic's time-domain metadata
        highest = np.max(np.abs(mode["omega"])) / (2 * np.pi)  # convert the highest absolute probed angular frequency to Hz
        if np.min(mode["omega"]) <= 0 or np.min(mode["DomegaDt"]) <= 0:  # check that the phase describes a positive, increasing chirp
            raise ValueError("Phase is not an increasing chirp; this SPA test is inapplicable.")  # stop when the stationary-phase test assumptions do not hold
        if highest >= 0.8 * freqs[-1]:  # require the maximum probed frequency to stay below 80 percent of Nyquist
            raise ValueError(f"Highest probed frequency is {highest:.6g} Hz; decrease --dt.")  # request a smaller time step if the frequency is too high
        direct = waveform.generate_tdomain_waveform(times) * time_weights  # generate both direct time-domain polarizations and taper them
        inverse, fd_mode = fd_to_td(  # transform the tapered frequency-domain waveform to the time grid
            waveform, freqs, count, args.dt, origin, start, end, key)  # keep its frequency bins and stationary-phase data for later checks
        if not np.all(np.isfinite(direct)) or not np.all(np.isfinite(inverse)):  # check both waveforms for nonfinite values
            raise ValueError("Nonfinite waveform samples.")  # stop before computing metrics on invalid samples
        return direct, inverse, fd_mode  # return the two waveforms and the frequency-domain mode metadata

    direct_vacuum, inverse_vacuum, vacuum_fd_mode = evaluate(vacuum)  # evaluate the common vacuum reference and retain its metadata
    vacuum_error, vacuum_mismatch = compare(inverse_vacuum, direct_vacuum, interior)  # measure baseline agreement before any environmental corrections
    print(f"vacuum: relative L2 {vacuum_error:.6g}; unweighted mismatch {vacuum_mismatch:.6g}",  # print the vacuum relative error and mismatch
          flush=True)  # flush the vacuum metrics before evaluating the environmental models
    passed = vacuum_error <= args.waveform_tolerance  # initialize the pass flag from the vacuum waveform comparison
    models = ["roemer", "bhl", "supersonic"] if args.model == "all" else [args.model]  # choose all environmental models or one named model
    plot_data = []  # prepare a list for optional plot traces
    for model in models:  # run the selected environmental effects one at a time
        options, strength_key = environment(args, model)  # get each model's parameters and the strength used for rescaling
        waveform = pyEFPEHM.pyEFPE({**base, **options})  # construct an initial waveform with the supplied physical parameters
        initial_phase = float(-key[2] * waveform._environmental_tdomain_orbital_phase(reference_time))
        if args.target_phase:  # rescale the effect when a nonzero target phase is requested
            if not np.isfinite(initial_phase) or initial_phase == 0:  # check that the initial phase can be used as a scale
                raise ValueError(f"Cannot normalize {model} phase {initial_phase}.")  # report a phase that cannot be normalized
            options[strength_key] *= args.target_phase / abs(initial_phase)  # rescale the linear strength parameter to reach the target phase magnitude
            waveform = pyEFPEHM.pyEFPE({**base, **options})  # rebuild the waveform with the rescaled environmental parameters
        direct, inverse, fd_mode = evaluate(waveform)  # compute the direct and inverse-transformed environmental waveforms
        # Match physical frequencies for the phase diagnostic only. The FFTs
        # above retain each model's FULL support, including unmatched edge bins.
        common_bins, env_indices, vac_indices = np.intersect1d(
            fd_mode["freq_idxs"], vacuum_fd_mode["freq_idxs"],
            return_indices=True,
        )
        env_times = fd_mode["t_SPA"][env_indices]
        vac_times = vacuum_fd_mode["t_SPA"][vac_indices]
        left, right = start + 0.25*duration, end - 0.25*duration
        phase_mask = ((env_times >= left) & (env_times <= right)
                      & (vac_times >= left) & (vac_times <= right))
        if not np.any(phase_mask):
            raise ValueError("No common interior FD bins for the phase diagnostic.")
        fd_phase = (fd_mode["psi_SPA"][env_indices[phase_mask]]
                    - vacuum_fd_mode["psi_SPA"][vac_indices[phase_mask]])
        # This is the first-order relation evaluated at VACUUM emission times.
        # It is exact for the phase_only prescription, not a requirement on the
        # finite-strength stationary_phase construction.
        td_phase = key[2] * waveform._environmental_tdomain_orbital_phase(vac_times[phase_mask])
        phase_norm = np.linalg.norm(fd_phase)
        if phase_norm == 0 or not np.isfinite(phase_norm):
            raise ValueError(f"{model} FD correction is too small or nonfinite for the phase diagnostic.")
        prescription_error = np.linalg.norm(fd_phase + td_phase) / phase_norm
        if not np.isfinite(prescription_error):
            raise ValueError("Nonfinite phase-prescription discrepancy.")
        stationary_shift = float(np.max(np.abs(env_times[phase_mask] - vac_times[phase_mask])))
        error, mismatch = compare(inverse, direct, interior)  # compare the complete environmental waveforms
        delta_direct = direct - direct_vacuum  # subtract vacuum from the direct TD waveform
        delta_inverse = inverse - inverse_vacuum  # subtract vacuum after both FD waveforms have been inverse-transformed
        effect_size = np.linalg.norm(delta_direct[:, interior]) / np.linalg.norm(  # measure the TD environmental change as a fraction of the vacuum norm
            direct_vacuum[:, interior])  # normalize using the same interior time samples
        if effect_size < 1e-8:  # check that the environmental change is large enough to measure
            raise ValueError(f"{model} waveform change {effect_size:.3g} is too small to test.")  # report the change that fell below the diagnostic floor
        effect_error, _ = compare(delta_inverse, delta_direct, interior)  # compare the environmental residuals in the two domains
        print(f"{model}: {options}")  # print the model parameters actually used
        print(f"  FD active bins {len(fd_mode['freq_idxs'])}; vacuum {len(vacuum_fd_mode['freq_idxs'])}; common {len(common_bins)}")
        print(f"  max interior stationary-time shift {stationary_shift:.6g} s")
        print(f"  relative first-order phase-prescription discrepancy {prescription_error:.6g} (diagnostic only)")  # report the relative disagreement between the phase prescriptions
        print(f"  relative L2 {error:.6g}; unweighted mismatch {mismatch:.6g}")  # report the complete waveform error and mismatch
        print(f"  TD change/vacuum {effect_size:.6g}; relative environmental residual error {effect_error:.6g}",  # print the size of the TD change and the error between residuals
              flush=True)  # flush the model metrics before moving to the next run
        passed = passed and error <= args.waveform_tolerance and effect_error <= args.effect_tolerance  # require both waveform and residual errors to meet their limits
        if args.plot:  # keep plot traces only when a plot was requested
            plot_data.append((model, direct[0, interior], inverse[0, interior],  # save the interior plus-polarization waveforms for this model
                              delta_direct[0, interior], delta_inverse[0, interior]))  # include the corresponding changes relative to vacuum

    if args.plot:  # generate a plot if an output filename was supplied
        import matplotlib  # load Matplotlib only for the optional plot
        matplotlib.use("Agg")  # select the backend that renders without a display
        import matplotlib.pyplot as plt  # import plotting functions after choosing the backend
        fig, axes = plt.subplots(len(plot_data), 2, figsize=(12, 3 * len(plot_data)),  # allocate two panels per model to show the waveform and residual
                                 squeeze=False)  # keep the axes in a two-dimensional array for a single model
        elapsed = times[interior] - start  # express the plotted times in seconds since the trajectory start
        visible = elapsed <= elapsed[0] + min(1000.0, 0.5 * duration)  # show at most the first 1000 seconds of the interior comparison window
        for row, (model, direct, inverse, delta_direct, delta_inverse) in enumerate(plot_data):  # assign each environmental model to a plot row
            for column, (a, b, title) in enumerate([  # pair each panel's direct and inverse-transformed traces with a title
                (direct, inverse, model),  # use the left panel for the complete waveforms
                (delta_direct, delta_inverse, model + " change relative to vacuum"),  # use the right panel for the changes relative to vacuum
            ]):  # finish defining the two panels for this model
                ax = axes[row, column]  # select the axes for the current panel
                ax.plot(elapsed[visible], a[visible], label="Direct TD")  # plot the direct time-domain result
                ax.plot(elapsed[visible], b[visible], "--", label="Inverse FFT of FD")  # overlay the inverse-transformed frequency-domain result with a dashed line
                ax.set(xlabel="Seconds since trajectory start", ylabel="Plus strain", title=title)  # label the time axis, strain axis, and panel
                ax.legend()  # identify the two traces in a legend
        fig.tight_layout()  # adjust spacing so labels fit
        fig.savefig(args.plot, dpi=160)  # write the diagnostic figure to the requested path
        plt.close(fig)  # close the figure to release its resources
        print("Saved plot", args.plot)  # confirm where the image was saved
    if args.check:  # report pass or fail when threshold checks are enabled
        print("PASS" if passed else "FAIL", "using the supplied diagnostic tolerances.")  # show the result of the accumulated checks
        return 0 if passed else 1  # return exit code 1 if any threshold was exceeded
    print("Diagnostic complete. Use --check to enforce error thresholds.")  # explain how to enable threshold checks for a later run
    return 0  # return success after a completed diagnostic-only run


if __name__ == "__main__":  # run the entry point when launched as a script
    try:  # catch expected input and waveform selection errors
        sys.exit(main())  # exit using the comparison outcome
    except (ValueError, KeyError) as error:  # intercept errors with a direct explanation for the caller
        print(f"Comparison stopped: {error}", file=sys.stderr)  # print the reason for stopping to stderr
        sys.exit(2)  # return exit code 2 when the comparison cannot run
