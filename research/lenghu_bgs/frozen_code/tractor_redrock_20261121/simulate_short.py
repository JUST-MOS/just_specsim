"""Repeat the frozen Tractor simulation for three one-read short exposures.

This separate entry point leaves both the original simulator and the ongoing
five-time run untouched. Output IDs are unique within each batch; across the
short and original runs join by (TARGETID, sky, total_s), not synthetic ID.
"""
import importlib.util
import json
from pathlib import Path
import sys

from simulate_tractor import DATA, HIGH, POP, ROOT, SHARED, sha, source_arm

SHORT_TIMES = {125: [125], 250: [250], 500: [500]}


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
    spec = importlib.util.spec_from_file_location("original_forward_simulation", POP / "simulate_batch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.DATA, mod.OUT = DATA, SHARED
    mod.make_source_arm = source_arm
    mod.TIMES = SHORT_TIMES
    mod.main()
    manifest_path = related[-1]
    manifest = json.loads(manifest_path.read_text())
    manifest.update({
        "fiber_profile": "Tractor DR9.4 catalog Gaussian mixture; continuous JUST aperture",
        "offset_direction": "galaxy major axis",
        "short_times_exposure_s": SHORT_TIMES,
        "short_simulator_sha256": sha(Path(__file__)),
        "tractor_simulator_sha256": sha(Path(__file__).with_name("simulate_tractor.py")),
        "tractor_adapter_sha256": sha(HIGH / "tractor_adapter.py"),
        "tractor_coefficients_reader_sha256": sha(HIGH / "compare_tractor.py"),
        "tractor_sersic_reference_sha256": sha(HIGH / "reference/tractor_dr9_4/sersic.py"),
        "tractor_mixture_reference_sha256": sha(HIGH / "reference/tractor_dr9_4/mixture_profiles.py"),
        "mixture_module_sha256": sha(ROOT / "py/just_etc/mixture_fiber_coupling.py"),
        "mixture_verification_sha256": sha(HIGH / "mixture_verification.json"),
        "extension_design_sha256": sha(DATA / "frozen_design.json"),
        "synthetic_id_scope": "unique within batch; join runs by TARGETID, sky and total_s",
    })
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
