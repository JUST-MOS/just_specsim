"""Fetch matching ESO zenith skies for 4200 m research extrapolation.

The downloads are original models at the ESO observatory codes, not measured
Lenghu spectra. They are saved only inside this new research directory.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import requests
from astropy.io import fits

HERE = Path(__file__).resolve().parent
SKY = HERE.parent / "sky"
BASE = "https://etimecalret-002.eso.org/observing/etc"
SITES = {"lasilla_2400": "2400", "paranal_2640": "2640", "anchor_3060": "3060m"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(name, code):
    params = json.loads((SKY / "skycalc_zenith_dark_input.json").read_text())
    params.update(observatory=code, wdelta=0.01)
    pjson = HERE / f"{name}_zenith_0p01nm_input.json"
    output = HERE / f"{name}_zenith_0p01nm.fits"
    if output.exists():
        if not pjson.exists() or json.loads(pjson.read_text()) != params:
            raise RuntimeError(f"refusing to reuse mismatched file {output}")
    else:
        pjson.write_text(json.dumps(params, indent=2) + "\n")
        response = requests.post(BASE + "/bin/script/skycalc_api.py", data=json.dumps(params), timeout=110)
        response.raise_for_status()
        result = response.json()
        if result.get("status") != "success":
            raise RuntimeError(f"ESO request failed for {name}: {result}")
        url = BASE + "/tmp/" + result["tmpdir"] + "/skytable.fits"
        response = requests.get(url, timeout=100)
        response.raise_for_status()
        temp = output.with_suffix(".fits.part")
        temp.write_bytes(response.content)
        with fits.open(temp, memmap=False) as h:
            h.verify("exception")
            if len(h[1].data) != 56001:
                raise ValueError(f"unexpected sky grid length for {name}")
        temp.rename(output)
    with fits.open(output, memmap=False) as h:
        d = h[1].data
        w = np.asarray(d["lam"], dtype=float)
        f = np.asarray(d["flux"], dtype=float)
        if (h[1].columns["lam"].unit != "nm" or
            h[1].columns["flux"].unit != "ph/s/m2/micron/arcsec2" or
            len(w) != 56001 or not np.all(np.diff(w) > 0) or
            not np.all(np.isfinite(f)) or np.any(f < 0) or
            not np.isclose(w[0], 365.) or not np.isclose(w[-1], 925.)):
            raise ValueError(f"invalid sky data for {name}")
    return {"input": str(pjson), "input_sha256": sha(pjson), "fits": str(output),
            "fits_sha256": sha(output), "samples": len(w), "observatory": code}


def main():
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {name: pool.submit(fetch, name, code) for name, code in SITES.items()}
        result = {name: job.result() for name, job in jobs.items()}
    metadata = {"source": BASE, "airmass": 1.0, "spacing_nm": .01,
                "status": "ESO models; not Lenghu measurements", "sites": result,
                "elapsed_seconds": time.monotonic()-start}
    (HERE / "zenith_fetch_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
