"""Match frozen DESI targets to seven publicly downloaded DR9 Tractor bricks."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from desiutil.brick import Bricks
from desitarget.targets import decode_targetid

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
SAMPLE = STUDY / "leader_195_205_20261116/extended_20p5/input/frozen_sample.csv"
REF = HERE / "reference/legacy_dr9_north"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    data = pd.read_csv(SAMPLE)
    obj, brick, release, mock, sky, gaiadr = decode_targetid(data.TARGETID.to_numpy(np.int64))
    data["objid"], data["brickid"], data["release"] = obj, brick, release
    data["brickname"] = Bricks(bricksize=.25).brickname(data.TARGET_RA.to_numpy(), data.TARGET_DEC.to_numpy())
    rows, files = [], []
    for f in sorted(REF.glob("tractor-*.fits")):
        name = f.stem[len("tractor-"):]
        subset = data[data.brickname == name]
        with fits.open(f, memmap=True) as hdus:
            h, table = hdus[0].header, hdus[1].data
            deps = {str(h[k]).strip():str(h.get(k.replace("DEPNAM", "DEPVER"))).strip()
                    for k in h if k.startswith("DEPNAM")}
            files.append(dict(path=str(f.relative_to(ROOT)), sha256=digest(f), brick=name,
                              legpipe=h.get("LEGPIPEV"), tractor=deps.get("tractor"),
                              photutils=deps.get("photutils"), targets=len(subset)))
            for row in subset.itertuples(index=False):
                indices = np.flatnonzero((table["objid"] == row.objid) &
                            (table["brickid"] == row.brickid) & (table["release"] == row.release))
                if len(indices) != 1:
                    raise ValueError("missing or duplicate catalog TARGETID {}".format(row.TARGETID))
                t = table[indices[0]]
                rows.append(dict(TARGETID=int(row.TARGETID), split=row.split, morphology=row.morphology,
                    brick=name, legpipe=h.get("LEGPIPEV"), tractor=deps.get("tractor"),
                    ra=float(row.TARGET_RA), dec=float(row.TARGET_DEC),
                    bx=float(t["bx"]), by=float(t["by"]), flux_r=float(t["flux_r"]),
                    fiberflux_r=float(t["fiberflux_r"]), shape_r=float(t["shape_r"]),
                    shape_e1=float(t["shape_e1"]), shape_e2=float(t["shape_e2"]),
                    sersic=float(t["sersic"]), sample_fiberflux_r=float(row.FIBERFLUX_R),
                    sample_flux_r=float(row.FLUX_R), sample_shape_r=float(row.SHAPE_R),
                    sample_e1=float(row.SHAPE_E1), sample_e2=float(row.SHAPE_E2),
                    sample_sersic=float(row.SERSIC)))
    result = pd.DataFrame(rows)
    if not len(result) or not result.TARGETID.is_unique:
        raise ValueError("bad matched roster")
    for a,b in (("fiberflux_r","sample_fiberflux_r"), ("flux_r","sample_flux_r"),
                ("shape_r","sample_shape_r"), ("shape_e1","sample_e1"),
                ("shape_e2","sample_e2"), ("sersic","sample_sersic")):
        if not np.allclose(result[a], result[b], rtol=1e-6, atol=1e-7):
            raise ValueError("frozen DESI input differs from raw DR9 field: " + a)
    result.to_csv(HERE / "matched_legacy_bricks.csv", index=False)
    metadata = {"n_matched":len(result), "n_test":int((result.split=="test").sum()),
                "files":files, "sample_sha256":digest(SAMPLE), "redrock_run":False}
    (HERE / "matched_legacy_bricks.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
