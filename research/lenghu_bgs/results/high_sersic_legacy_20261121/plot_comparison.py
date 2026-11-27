"""Plot catalog reconstruction and JUST coupling for the frozen test set."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_tractor import HERE, ROOT, sha


def main():
    catalog = pd.read_csv(HERE / "catalog_reconstruction_per_target.csv")
    just = pd.read_csv(HERE / "just_mixture_coupling.csv")
    d = catalog[catalog.split == "test"].merge(
        just[["TARGETID", "just_nom_mixture_over_analytic", "just_survey_mixture_over_analytic"]],
        on="TARGETID", validate="one_to_one")
    edges = [0., 1.25, 2., 3., 4., 5., np.inf]
    labels = ["0.5–1.25", "1.25–2", "2–3", "3–4", "4–5", "5–6"]
    groups = [d[(d.SERSIC >= lo) & (d.SERSIC < hi)] for lo, hi in zip(edges[:-1], edges[1:])]
    if sum(map(len, groups)) != len(d) or any(len(g) == 0 for g in groups):
        raise ValueError("unexpected Sersic groups")
    positions = np.arange(len(groups), dtype=float)
    style = {"font.family": "serif", "font.size": 10, "axes.titlesize": 11,
             "axes.labelsize": 10, "xtick.labelsize": 8.5, "ytick.labelsize": 9,
             "legend.fontsize": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), layout="constrained")
        ax, bx = axes
        ax.axhline(1., color="0.45", linestyle=":", lw=1)
        bx.axhline(1., color="0.45", linestyle=":", lw=1)
        def draw(axis, column, offset, color, marker, label):
            q = np.array([np.percentile(g[column], [16, 50, 84]) for g in groups])
            axis.errorbar(positions + offset, q[:, 1], yerr=[q[:, 1]-q[:, 0], q[:, 2]-q[:, 1]],
                          fmt=marker, color=color, capsize=2.5, elinewidth=1.1,
                          markersize=5.5, label=label)
        draw(ax, "analytic_over_catalog", -.16, "#B23A29", "o", "Analytic Sérsic")
        draw(ax, "continuous_over_catalog", 0., "#0072B2", "s", "Tractor mixture, continuous")
        draw(ax, "pixel_over_catalog", .16, "#22804D", "^", "Tractor mixture + pixels")
        draw(bx, "just_nom_mixture_over_analytic", -.10, "#0072B2", "s", "Nominal FWHM at 800 nm: 1.05″")
        draw(bx, "just_survey_mixture_over_analytic", .10, "#A45100", "D", "Survey FWHM at 800 nm: 0.80″")
        xlabels = [f"{lab}\n({len(g)})" for lab, g in zip(labels, groups)]
        for axis in axes:
            axis.set_xticks(positions, xlabels)
            axis.set_xlabel("Catalog Sérsic index $n$ (number of test galaxies)")
            axis.grid(axis="y", color="0.9", lw=.6)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(.84, 1.035)
        bx.set_ylim(.985, 1.195)
        ax.set_ylabel("Predicted / Legacy catalog fiber fraction")
        bx.set_ylabel("JUST mixture / analytic fiber fraction")
        ax.set_title("(a) Legacy DR9 r-band reconstruction", loc="left")
        bx.set_title("(b) JUST at 623.1 nm", loc="left")
        ax.legend(loc="lower left", frameon=False)
        bx.legend(loc="upper left", frameon=False)
        fig.suptitle("Galaxy fiber coupling: catalog-consistent profiles", fontsize=13)
        fig.savefig(HERE / "high_sersic_comparison.png", dpi=200, facecolor="white")
        fig.savefig(HERE / "high_sersic_comparison.pdf", facecolor="white")
        plt.close(fig)
    metadata = {"n_test": len(d), "n_per_bin": [len(g) for g in groups],
                "sersic_bin_edges": ["0", "1.25", "2", "3", "4", "5", "infinity"],
                "bin_interval": "left inclusive, right exclusive",
                "statistics": "median and 16th–84th percentiles over galaxies; not confidence intervals",
                "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                                  (HERE / "catalog_reconstruction_per_target.csv",
                                   HERE / "just_mixture_coupling.csv", Path(__file__))},
                "figure_inches": [11.6, 4.8], "png_dpi": 200,
                "matplotlib": matplotlib.__version__}
    (HERE / "figure_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
