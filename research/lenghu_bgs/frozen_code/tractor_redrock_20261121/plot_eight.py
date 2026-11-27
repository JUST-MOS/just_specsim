"""Plot the completed eight-time Tractor and historical five-time scans."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator
import numpy as np
import pandas as pd

from analyze_eight import OUT
from simulate_tractor import sha


def main():
    datafile = OUT / "success_curves.csv"
    curves = pd.read_csv(datafile)
    if len(curves) != 52:
        raise ValueError("expected the complete 52-row comparison")
    colors = {"original": "#A84132", "mixture": "#0072B2"}
    names = {"original": "Original circular exponential", "mixture": "Improved Tractor mixture"}
    ticks = [125, 250, 500, 750, 1500, 3000]
    with plt.rc_context({"font.family": "serif", "font.size": 10, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(2, 2, figsize=(10.8, 8), layout="constrained", sharex=True, sharey=True)
        for i, (group, title) in enumerate((("r_lt_19p5", r"$r_0<19.5$"), ("r_lt_20p5", r"$r_0<20.5$"))):
            for j, sky in enumerate(("dark", "bright")):
                ax = axes[i, j]
                for model, n_times in (("original", 5), ("mixture", 8)):
                    sub = curves[(curves.group == group) & (curves.sky == sky) & (curves.model == model)].sort_values("time_s")
                    if len(sub) != n_times:
                        raise ValueError("incomplete plot series")
                    marker = "o" if model == "original" else "s"
                    ax.plot(sub.time_s, sub.success_fraction, color=colors[model], marker=marker, ms=4, lw=1.5)
                    ax.plot(sub.time_s, sub.cp95_lower_bonferroni_8, color=colors[model], ls="--", marker=".", ms=3, lw=1.2)
                ax.axhline(.95, color="0.3", ls=":", lw=1.)
                ax.set_xscale("log")
                ax.xaxis.set_major_locator(FixedLocator(ticks))
                ax.xaxis.set_major_formatter(FixedFormatter([str(x) for x in ticks]))
                ax.minorticks_off()
                ax.set_xlim(105, 4200)
                ax.set_title(f"{title} · {sky.capitalize()} sky (n = {int(sub.n.iloc[0])})", loc="left")
                ax.grid(axis="y", color="0.9", lw=.6)
                ax.spines[["top", "right"]].set_visible(False)
                if j == 0:
                    ax.set_ylabel("Redshift recovery fraction")
                if i == 1:
                    ax.set_xlabel("Total exposure time (s; logarithmic scale)")
        minimum = float(curves[["success_fraction", "cp95_lower_bonferroni_8"]].min().min())
        axes[0, 0].set_ylim(max(0., np.floor((minimum - .02) * 10) / 10), 1.01)
        handles = [Line2D([0], [0], color=colors[k], marker="o" if k == "original" else "s", lw=1.5, label=names[k]) for k in ("original", "mixture")]
        handles += [Line2D([0], [0], color="0.25", lw=1.5, label="Success fraction"),
                    Line2D([0], [0], color="0.25", ls="--", lw=1.2, label="One-sided lower bound; 8-time adjustment"),
                    Line2D([0], [0], color="0.3", ls=":", lw=1, label="95% goal")]
        fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=9)
        fig.suptitle("JUST 4200 m redshift recovery: impact of fiber coupling", fontsize=13)
        fig.savefig(OUT / "recovery_comparison.png", dpi=200, facecolor="white")
        fig.savefig(OUT / "recovery_comparison.pdf", facecolor="white")
        plt.close(fig)
    manifest = {"source": str(datafile), "source_sha256": sha(datafile), "script_sha256": sha(Path(__file__)),
                "line_styles": "solid: point; dashed: one-sided 95% lower adjusted over eight times",
                "matplotlib": matplotlib.__version__, "size_inches": [10.8, 8], "png_dpi": 200,
                "original_times_note": "The original model was not evaluated below 750 s."}
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
