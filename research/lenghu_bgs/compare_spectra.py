"""Compare generated spectra to matching TARGETIDs in an archived pilot FITS.

The input spectra retain native flux units of 10^-17 erg s^-1 cm^-2 Angstrom^-1;
inverse variance and resolution are compared on their saved grids. A candidate
may contain a subset of the baseline scenarios. Both files remain untouched.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits


def compare(baseline, candidate, rtol=1e-6, atol=1e-7):
    checks = {}
    # Unsigned MASK images use FITS BZERO scaling and cannot be memory-mapped
    # through Astropy. Pilot files are small enough to read without memmap.
    with fits.open(baseline, memmap=False, checksum=True) as ref, fits.open(candidate, memmap=False, checksum=True) as new:
        ref.verify("exception")
        new.verify("exception")
        old_id = np.asarray(ref["FIBERMAP"].data["TARGETID"], dtype=np.int64)
        new_id = np.asarray(new["FIBERMAP"].data["TARGETID"], dtype=np.int64)
        if len(set(old_id)) != len(old_id) or len(set(new_id)) != len(new_id):
            raise ValueError("Duplicate spectrum TARGETID")
        lookup = {int(value): index for index, value in enumerate(old_id)}
        if not all(int(value) in lookup for value in new_id):
            raise ValueError("Candidate TARGETID is not in the baseline")
        order = np.asarray([lookup[int(value)] for value in new_id])
        for arm in "BRZ":
            wave_old = np.asarray(ref[f"{arm}_WAVELENGTH"].data)
            wave_new = np.asarray(new[f"{arm}_WAVELENGTH"].data)
            if not np.array_equal(wave_old, wave_new):
                raise ValueError(f"{arm} wavelength grid changed")
            old_mask = np.asarray(ref[f"{arm}_MASK"].data[order])
            new_mask = np.asarray(new[f"{arm}_MASK"].data)
            if not np.array_equal(old_mask, new_mask):
                raise ValueError(f"{arm} mask changed")
            for field in ("FLUX", "IVAR", "RESOLUTION"):
                a = np.asarray(ref[f"{arm}_{field}"].data[order], dtype=float)
                b = np.asarray(new[f"{arm}_{field}"].data, dtype=float)
                if a.shape != b.shape or not np.array_equal(np.isfinite(a), np.isfinite(b)):
                    raise ValueError(f"{arm}_{field} shape or finite mask changed")
                difference = np.abs(a - b)
                checks[f"{arm}_{field}"] = {
                    "bitwise_equal": bool(np.array_equal(a, b)),
                    "max_absolute_difference": float(np.nanmax(difference)) if difference.size else 0.,
                    "within_tolerance": bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)),
                }
    return {"status": "passed" if all(v["within_tolerance"] for v in checks.values()) else "failed",
            "matched_spectra": len(new_id), "wavelength_and_mask_exact": True,
            "rtol": rtol, "atol": atol, "arrays": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-7)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate, args.rtol, args.atol)
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
