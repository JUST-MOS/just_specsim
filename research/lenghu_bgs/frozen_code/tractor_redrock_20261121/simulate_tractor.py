"""Frozen 4200-m survey simulation with the catalog-consistent source profile.

Calls the original deterministic/noise simulator without editing its code,
changing only its per-source fiber coupling via the independently validated
Tractor DR9.4 mixture adapter.  Redrock is run separately by the scan driver.
"""
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
POP = STUDY / "population_4200_camera_overlap"
EXT = STUDY / "leader_195_205_20261116/extended_20p5"
HIGH = STUDY / "high_sersic_legacy_20261121"
DATA, SHARED = EXT / "input", EXT / "shared"
sys.path[:0] = [str(POP), str(HIGH)]

from compare_tractor import TractorDR94Profiles
from tractor_adapter import make_source_arm_mixture
from just_etc.mixture_fiber_coupling import GaussianMixtureProfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


PROFILE_DATA = TractorDR94Profiles()


@lru_cache(maxsize=512)
def profile(morphology, n, radius, e1, e2):
    amplitudes, variances = PROFILE_DATA.profile(morphology, n)
    return GaussianMixtureProfile(radius, e1, e2, amplitudes, variances)


def source_arm(row, flux, source_wave, band, arm, inst, calc, lsf_wave,
               lsf_sigma, mapping, fallback_sigma=.7):
    galaxy = profile(row.morphology, row.SERSIC, row.SHAPE_R, row.SHAPE_E1, row.SHAPE_E2)
    return make_source_arm_mixture(row, flux, source_wave, band, arm, inst, calc,
                                   lsf_wave, lsf_sigma, mapping, galaxy, fallback_sigma)


def main():
    design = json.loads((DATA / "frozen_design.json").read_text())
    verification = json.loads((HIGH / "mixture_verification.json").read_text())
    if (sha(DATA / "frozen_sample.csv") != design["sample_sha256"] or
        sha(DATA / "normalized_sources.fits") != design["normalized_sources_sha256"] or
        verification["status"] != "passed"):
        raise ValueError("frozen source inputs or improved profile verification failed")
    for relative, digest in verification["inputs_sha256"].items():
        if sha(ROOT / relative) != digest:
            raise ValueError("verified mixture dependency changed: " + relative)
    if "--output" not in sys.argv:
        raise ValueError("output path required")
    out = Path(sys.argv[sys.argv.index("--output") + 1])
    related = [out, out.with_name(out.stem + "_truth.csv"), out.with_name(out.stem + "_masks.csv"),
               out.with_name(out.stem + "_manifest.json")]
    if any(path.exists() for path in related):
        raise FileExistsError("refusing to overwrite an existing or partial simulation")
    source = POP / "simulate_batch.py"
    spec = importlib.util.spec_from_file_location("original_forward_simulation", source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.DATA, mod.OUT = DATA, SHARED
    mod.make_source_arm = source_arm
    mod.main()
    manifest_path = related[-1]
    manifest = json.loads(manifest_path.read_text())
    manifest.update({
        "fiber_profile": "Tractor DR9.4 catalog Gaussian mixture; continuous JUST aperture",
        "offset_direction": "galaxy major axis",
        "tractor_simulator_sha256": sha(Path(__file__)),
        "tractor_adapter_sha256": sha(HIGH / "tractor_adapter.py"),
        "tractor_coefficients_reader_sha256": sha(HIGH / "compare_tractor.py"),
        "tractor_sersic_reference_sha256": sha(HIGH / "reference/tractor_dr9_4/sersic.py"),
        "tractor_mixture_reference_sha256": sha(HIGH / "reference/tractor_dr9_4/mixture_profiles.py"),
        "mixture_module_sha256": sha(ROOT / "py/just_etc/mixture_fiber_coupling.py"),
        "mixture_verification_sha256": sha(HIGH / "mixture_verification.json"),
        "extension_design_sha256": sha(DATA / "frozen_design.json"),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
