"""Generate a galaxy with just_specsim and observe it with the JUST ETC.

``just_specsim.SpectrumMaker`` first selects a BGS template using rest-frame
``(M_r, g-r)`` and returns a redshifted, noiseless observer-frame spectrum.
That spectrum is converted from Redrock flux units to physical FLAM and passed
through the JUST ETC noise model.  The result is written in the same
Redrock-compatible FITS layout as ``simulate_mock_spectrum.py``.

The current just_specsim repository loads its template and line data relative
to its ``example`` directory.  This program handles that repository detail via
``--just-specsim-root`` and does not modify the just_specsim checkout.

Example
-------
python examples/simulate_justspecsim_spectrum.py \
    --just-specsim-root ../just_specsim \
    --redshift 0.1 --absolute-magnitude-r -21 --color-gr 0.7 \
    --exposure-time 900 --exposures 4 \
    --output output/justspecsim_redrock.fits \
    --plot-output output/justspecsim_redrock_spectrum.png
"""

from argparse import ArgumentParser
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import warnings

import numpy as np

# When this example is run directly from a Git checkout, prefer that checkout's
# source tree over an older just_etc installation in the active environment.
JUST_ETC_SOURCE = Path(__file__).resolve().parents[1] / "py"
if str(JUST_ETC_SOURCE) not in sys.path:
    sys.path.insert(0, str(JUST_ETC_SOURCE))

from just_etc import JUSTExposureTimeCalculator, write_redrock_mock_spectrum


DEFAULT_SPECSIM_ROOT = Path(__file__).resolve().parents[2] / "just_specsim"
SPECSIM_FLUX_SCALE = 1.0e-17
ARM_COLORS = ("#3572A5", "#D97706", "#B91C1C")
ARM_LABELS = ("arm 0", "arm 1", "arm 2")
PLOT_MIN_SNR = 0.5

@contextmanager
def _just_specsim_repository(specsim_root):
    """Make the checkout importable and expose its repository-relative data."""
    root = Path(specsim_root).expanduser().resolve()
    package_dir = root / "py" / "just_specsim"
    example_dir = root / "example"
    required = (
        package_dir / "maker.py",
        example_dir / "DESI_templates",
        example_dir / "data",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "just_specsim checkout is incomplete; missing: " + ", ".join(missing)
        )

    package_parent = str(package_dir.parent)
    old_cwd = Path.cwd()
    inserted = package_parent not in sys.path
    if inserted:
        sys.path.insert(0, package_parent)
    try:
        os.chdir(example_dir)
        yield root
    finally:
        os.chdir(old_cwd)
        if inserted:
            sys.path.remove(package_parent)


def _validate_specsim_spectrum(wave_aa, flux_redrock):
    """Validate a one-object observer-frame just_specsim result."""
    wave_aa = np.asarray(wave_aa, dtype=np.float64)
    flux_redrock = np.asarray(flux_redrock, dtype=np.float64)
    if wave_aa.ndim != 1 or flux_redrock.ndim != 1:
        raise ValueError("just_specsim must return one-dimensional wave and flux arrays")
    if wave_aa.shape != flux_redrock.shape:
        raise ValueError("just_specsim wave and flux arrays must have the same shape")
    if not np.all(np.isfinite(wave_aa)) or not np.all(np.diff(wave_aa) > 0.0):
        raise ValueError("just_specsim wavelength grid must be finite and strictly increasing")
    if not np.all(np.isfinite(flux_redrock)):
        raise ValueError("just_specsim flux contains non-finite values")
    if np.any(flux_redrock < 0.0):
        warnings.warn("negative just_specsim flux values were clipped to zero", RuntimeWarning)
        flux_redrock = np.maximum(flux_redrock, 0.0)
    return wave_aa, flux_redrock


def _plot_simulated_spectrum(
    input_wave_aa,
    input_flux_redrock,
    mock_observation,
    plot_output,
    *,
    redshift,
    absolute_magnitude_r,
    color_gr,
    target_magnitude,
    exposure_time,
    exposures,
    show_plot=False,
):
    """Save the noiseless spectrum, mock observation, and per-pixel S/N."""
    import matplotlib.pyplot as plt

    plot_output = Path(plot_output).expanduser()
    plot_output.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    }):
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(11, 9),
            sharex=True,
            gridspec_kw={"height_ratios": (1.0, 1.25, 0.85), "hspace": 0.08},
        )

        axes[0].plot(
            input_wave_aa,
            input_flux_redrock,
            color="black",
            linewidth=0.8,
            label="just_specsim intrinsic spectrum",
        )
        axes[0].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")
        axes[0].legend(loc="upper right", frameon=False)
        axes[0].grid(alpha=0.18, linewidth=0.5)

        for arm_result, color, arm_label in zip(
            mock_observation, ARM_COLORS, ARM_LABELS
        ):
            wave_aa = np.asarray(arm_result["wave_aa"], dtype=np.float64)
            intrinsic = np.asarray(
                arm_result["flux_intrinsic"], dtype=np.float64
            ) * 1.0e17
            observed = np.asarray(arm_result["flux_mock"], dtype=np.float64) * 1.0e17
            snr = np.asarray(arm_result["snr"], dtype=np.float64)
            display = (
                np.isfinite(wave_aa)
                & np.isfinite(intrinsic)
                & np.isfinite(observed)
                & np.isfinite(snr)
                & (snr >= PLOT_MIN_SNR)
            )
            axes[1].plot(
                wave_aa[display],
                observed[display],
                color=color,
                linewidth=0.55,
                alpha=0.65,
                label=f"{arm_label} noisy",
            )
            axes[1].plot(
                wave_aa[display],
                intrinsic[display],
                color=color,
                linewidth=1.0,
                alpha=0.95,
                label=f"{arm_label} intrinsic",
            )
            finite_snr = np.isfinite(wave_aa) & np.isfinite(snr)
            axes[2].plot(
                wave_aa[finite_snr],
                snr[finite_snr],
                color=color,
                linewidth=0.9,
                label=arm_label,
            )

        axes[1].set_ylabel(r"$f_\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")
        axes[1].legend(loc="upper right", ncol=2, frameon=False)
        axes[1].grid(alpha=0.18, linewidth=0.5)
        axes[1].text(
            0.01,
            0.03,
            f"Display only: pixels with modeled S/N >= {PLOT_MIN_SNR:g}; FITS unchanged",
            transform=axes[1].transAxes,
            fontsize=8.5,
            color="0.35",
        )

        axes[2].axhline(
            PLOT_MIN_SNR,
            color="0.35",
            linestyle="--",
            linewidth=0.8,
            label="flux-panel cutoff",
        )
        axes[2].set_xlabel(r"Observed wavelength [$\AA$]")
        axes[2].set_ylabel("Combined S/N per pixel")
        axes[2].set_ylim(bottom=0.0)
        axes[2].legend(loc="upper right", ncol=4, frameon=False)
        axes[2].grid(alpha=0.18, linewidth=0.5)
        axes[2].set_xlim(
            min(float(np.min(result["wave_aa"])) for result in mock_observation),
            max(float(np.max(result["wave_aa"])) for result in mock_observation),
        )

        fig.suptitle(
            r"just_specsim $\rightarrow$ JUST ETC"
            f"  |  z={redshift:g}, $M_r$={absolute_magnitude_r:g}, "
            f"g-r={color_gr:g}, r={target_magnitude:g} AB"
            f"  |  {exposures} x {exposure_time:g} s",
            y=0.98,
        )
        fig.align_ylabels(axes)
        fig.subplots_adjust(top=0.92, bottom=0.10, left=0.10, right=0.98)
        fig.savefig(plot_output, dpi=180, bbox_inches="tight")
        if show_plot:
            plt.show()
        plt.close(fig)

    return plot_output


def simulate_justspecsim_spectrum(
    redshift=0.1,
    absolute_magnitude_r=None,
    color_gr=0.5,
    apparent_magnitude=19.5,
    template_seed=1,
    exposure_time=180.0,
    exposures=1,
    seeing=0.8,
    zenith_angle=45.0,
    effective_radius=0.0,
    ebv=0.03,
    noise_seed=42,
    target_id=10001,
    ra=150.0,
    dec=2.0,
    calc_mode="fast",
    just_specsim_root=DEFAULT_SPECSIM_ROOT,
    output=Path("output/justspecsim_redrock.fits"),
    plot_output=None,
    show_plot=False,
):
    """Generate and observe one BGS-like galaxy spectrum.

    Wavelengths passed between the packages are observer-frame Angstroms.
    ``just_specsim`` flux is in units of 1e-17 erg s^-1 cm^-2 Angstrom^-1;
    the factor of 1e-17 applied below converts it to the physical FLAM expected
    by ``JUSTExposureTimeCalculator``.

    ``absolute_magnitude_r`` and ``color_gr`` select the nearest basis template.
    The current ``SpectrumMaker`` assigns that template's own DECam r apparent
    magnitude. Set ``apparent_magnitude`` to rescale the complete spectrum to a
    different r-band magnitude while preserving its shape and equivalent widths.
    """
    scalar_inputs = {
        "redshift": redshift,
        "absolute_magnitude_r": absolute_magnitude_r,
        "color_gr": color_gr,
        "exposure_time": exposure_time,
        "seeing": seeing,
        "zenith_angle": zenith_angle,
        "effective_radius": effective_radius,
        "ebv": ebv,
        "ra": ra,
        "dec": dec,
    }
    if not all(np.isfinite(value) for value in scalar_inputs.values()):
        raise ValueError("all physical and target parameters must be finite")
    if redshift <= -1.0:
        raise ValueError("redshift must be greater than -1")
    if exposure_time <= 0.0:
        raise ValueError("exposure_time must be positive")
    if exposures < 1:
        raise ValueError("exposures must be at least 1")
    if seeing <= 0.0:
        raise ValueError("seeing must be positive")
    if effective_radius < 0.0:
        raise ValueError("effective_radius cannot be negative")
    if apparent_magnitude is not None and not np.isfinite(apparent_magnitude):
        raise ValueError("apparent_magnitude must be finite")

    with _just_specsim_repository(just_specsim_root) as specsim_root:
        from just_specsim.maker import SpectrumMaker

        spectrum_maker = SpectrumMaker()
        wave_aa, flux_redrock, meta, objmeta = spectrum_maker(
            z=redshift,
            Mr=absolute_magnitude_r,
            color=color_gr,
            seed=template_seed,
            nocolorcuts=True,
            saveto=None,
        )

    wave_aa, flux_redrock = _validate_specsim_spectrum(wave_aa, flux_redrock)
    native_magnitude = float(meta["MAG"])
    target_magnitude = native_magnitude
    magnitude_scale = 1.0
    if apparent_magnitude is not None:
        target_magnitude = float(apparent_magnitude)
        magnitude_scale = 10.0 ** (-0.4 * (target_magnitude - native_magnitude))
        flux_redrock = flux_redrock * magnitude_scale

    # Do not redshift or normalize again: SpectrumMaker has already done both.
    flux_flam = flux_redrock * SPECSIM_FLUX_SCALE

    calculator = JUSTExposureTimeCalculator(calc_mode=calc_mode)
    calculator.set_obs_conditions(
        seeing_fwhm_800=seeing,
        zenith_angle=zenith_angle,
        r_eff=effective_radius,
        ebv=ebv,
    )
    mock_observation = calculator.simulate_mock_observation(
        wave_aa,
        flux_flam,
        t_exp=exposure_time,
        n_exp=exposures,
        seed=noise_seed,
    )

    output = write_redrock_mock_spectrum(
        output,
        mock_observation,
        target_id=target_id,
        ra=ra,
        dec=dec,
        primary_metadata={
            "SPECSIM": ("just_specsim", "Intrinsic spectrum generator"),
            "SIMROOT": (str(specsim_root), "just_specsim checkout"),
            "SIMZ": (redshift, "Input redshift"),
            "SIMMR": (absolute_magnitude_r, "Input rest-frame r absolute magnitude"),
            "SIMGR": (color_gr, "Input rest-frame g-r color"),
            "SIMMAG": (native_magnitude, "Native just_specsim DECam r magnitude"),
            "OBSMAG": (target_magnitude, "Final DECam r magnitude"),
            "MAGSCALE": (magnitude_scale, "Optional apparent-magnitude scale"),
            "SIMTID": (int(meta["TEMPLATEID"]), "just_specsim template ID"),
            "SIMSEED": (template_seed, "just_specsim template seed"),
            "VDISP": (float(objmeta["VDISP"]), "Velocity dispersion [km/s]"),
            "EXPTIME": (exposure_time, "Seconds per exposure"),
            "NEXP": (exposures, "Number of co-added exposures"),
            "SEEING": (seeing, "FWHM at 800 nm [arcsec]"),
            "ZENITH": (zenith_angle, "Zenith angle [deg]"),
            "REFF": (effective_radius, "Effective radius [arcsec]"),
            "EBV": (ebv, "Galactic E(B-V)"),
            "RNGSEED": (noise_seed, "ETC random-noise seed"),
        },
    )

    if plot_output is None:
        output_path = Path(output)
        plot_output = output_path.with_name(f"{output_path.stem}_spectrum.png")
    plot_output = _plot_simulated_spectrum(
        wave_aa,
        flux_redrock,
        mock_observation,
        plot_output,
        redshift=redshift,
        absolute_magnitude_r=absolute_magnitude_r,
        color_gr=color_gr,
        target_magnitude=target_magnitude,
        exposure_time=exposure_time,
        exposures=exposures,
        show_plot=show_plot,
    )

    print(f"Saved Redrock-compatible JUST mock spectrum to {output.resolve()}")
    print(f"Saved simulated-spectrum plot to {plot_output.resolve()}")
    print(
        f"just_specsim: z={redshift:g}, Mr={absolute_magnitude_r:g}, "
        f"g-r={color_gr:g}, r={target_magnitude:g} AB | "
        f"JUST ETC: {exposures} x {exposure_time:g} s"
    )
    return output


def _parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--redshift", type=float, default=0.1)
    parser.add_argument("--absolute-magnitude-r", type=float, default=-21.0)
    parser.add_argument("--color-gr", type=float, default=0.5)
    parser.add_argument(
        "--apparent-magnitude",
        type=float,
        default=20.5,
    )
    parser.add_argument("--template-seed", type=int, default=1)
    parser.add_argument("--exposure-time", type=float, default=900.0)
    parser.add_argument("--exposures", type=int, default=4)
    parser.add_argument("--seeing", type=float, default=0.8)
    parser.add_argument("--zenith-angle", type=float, default=45.0)
    parser.add_argument("--effective-radius", type=float, default=0.0)
    parser.add_argument("--ebv", type=float, default=0.03)
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--target-id", type=int, default=10001)
    parser.add_argument("--ra", type=float, default=150.0)
    parser.add_argument("--dec", type=float, default=2.0)
    parser.add_argument("--calc-mode", choices=("fast", "balanced", "accurate"), default="fast")
    parser.add_argument(
        "--just-specsim-root",
        type=Path,
        default=DEFAULT_SPECSIM_ROOT,
        help=f"just_specsim checkout (default: {DEFAULT_SPECSIM_ROOT})",
    )
    parser.add_argument("--output", type=Path, default=Path("output/justspecsim_redrock.fits"))
    parser.add_argument(
        "--plot-output",
        type=Path,
        help="PNG plot path (default: <output stem>_spectrum.png)",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Also open the Matplotlib figure interactively",
    )
    return parser.parse_args()


if __name__ == "__main__":
    simulate_justspecsim_spectrum(**vars(_parse_args()))
