"""Visualize the read-boundary behavior of the two nominal proxy models."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audit_weighted_model import sha
from weighted_effective_model import WeightedDepthModel

HERE = Path(__file__).resolve().parent


def main():
    m = WeightedDepthModel()
    times = np.unique(np.r_[np.arange(2600., 3501., 10.), 2999.999, 3000., 3000.001])
    q_ref = m.information("nominal", 3000.)
    sum_q = np.array([m.frozen.information("nominal", t)/q_ref for t in times])
    wt_q = np.array([m.information("nominal", t)/q_ref for t in times])
    sum_eff = np.array([m.frozen.effective_time("nominal", t) for t in times])
    wt_eff = np.array([m.effective_time("nominal", t) for t in times])
    if not np.all(np.diff(wt_q) >= -1e-12) or np.max(np.abs(wt_eff-times)) > 1e-4:
        raise ValueError("weighted model failed continuity or nominal identity")
    plt.rcParams.update({"font.family":"serif", "font.size":10, "pdf.fonttype":42})
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8), layout="constrained")
    for a in ax:
        a.axvline(3000, color="0.55", linestyle="--", linewidth=.9)
        a.grid(alpha=.2)
        a.set_xlabel("Nominal reference shutter time [s]")
        a.set_xlim(2600, 3500)
    ax[0].plot(times, sum_q, color="#D55E00", linewidth=1.5, label="Frozen sum coadd")
    ax[0].plot(times, wt_q, color="#0072B2", linewidth=1.5, label="Ideal weighted coadd")
    ax[0].axhline(1, color="0.3", linestyle=":", linewidth=.9)
    ax[0].set_ylabel("Information Q / Q(3000 s)")
    ax[0].legend(frameon=False, fontsize=8.5, loc="lower right")
    ax[1].plot(times, sum_eff, color="#D55E00", linewidth=1.5, label="Frozen sum-coadd inverse")
    ax[1].plot(times, wt_eff, color="#0072B2", linewidth=1.5, label="Weighted reference inverse")
    ax[1].set_ylabel("Reference-equivalent time [s]")
    ax[1].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Effect of a new read on the reference depth model", fontsize=12)
    fig.supxlabel("Ideal weighted coadd is a model candidate; it has not been tested with Redrock.", fontsize=8)
    fig.savefig(HERE/"coadd_boundary.png", dpi=200, facecolor="white")
    fig.savefig(HERE/"coadd_boundary.pdf", facecolor="white")
    plt.close(fig)
    meta = {"status":"passed", "reference_time_s":3000., "split_s":750.,
            "nominal_scene_only":True,
            "scope":"method illustration; weighted coadd not Redrock-validated",
            "sha256":{p.name:sha(p) for p in (HERE/"coadd_boundary.png", HERE/"coadd_boundary.pdf",
                                               HERE/"depth_inputs.npz", Path(__file__))}}
    (HERE/"coadd_boundary_manifest.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps(meta,indent=2))


if __name__ == "__main__":
    main()
