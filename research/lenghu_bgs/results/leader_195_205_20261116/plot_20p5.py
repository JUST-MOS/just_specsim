"""Plot completed r < 20.5 and 20 < r < 20.5 recovery at sampled times."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "extended_20p5/analysis_20261119"


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    audit = json.loads((OUT / "independent_audit.json").read_text())
    manifest = json.loads((OUT / "analysis_manifest.json").read_text())
    table = OUT / "success_curves.csv"
    if (audit["status"] != "passed" or audit["analysis_manifest_sha256"] != sha(OUT / "analysis_manifest.json") or
        manifest["outputs"][table.name] != sha(table)):
        raise ValueError("complete audited analysis is required before plotting")
    data = pd.read_csv(table)
    specs = [("all", "all", r"All $r_0<20.5$"),
             ("magnitude", "20_lt_r_lt_20p5", r"Faint $20<r_0<20.5$")]
    drawn = [data[(data.group_type == kind) & (data.group == label)] for kind, label, _ in specs]
    ymin = max(0, int(20 * min(float(x.cp95_lower_bonferroni_times.min()) for x in drawn)) / 20 - .05)
    with plt.rc_context({"font.family": "serif", "font.size": 11, "axes.titlesize": 12,
                         "axes.labelsize": 11, "pdf.fonttype": 42, "figure.facecolor": "white",
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2), sharex=True, sharey=True, layout="constrained")
        colors = {"dark": "#0072B2", "bright": "#A34F00"}
        for j, (kind, label, title) in enumerate(specs):
            for i, sky in enumerate(("dark", "bright")):
                ax = axes[i, j]
                x = data[(data.group_type == kind) & (data.group == label) & (data.sky == sky)].sort_values("time_s")
                if len(x) != 5 or x.n.nunique() != 1:
                    raise ValueError("figure group is missing a sampled time")
                col = colors[sky]
                ax.axhline(.95, color="0.35", linestyle="--", linewidth=1, label="95% target")
                ax.vlines(x.time_s, x.cp95_lower_bonferroni_times, x.success_fraction, color=col, linewidth=1.5)
                ax.scatter(x.time_s, x.success_fraction, marker="o", s=42, color=col, zorder=3, label="Sample fraction")
                ax.scatter(x.time_s, x.cp95_lower_bonferroni_times, marker="v", s=38, color=col,
                           facecolors="white", linewidths=1.4, zorder=3, label="Adjusted 95% lower bound")
                ax.grid(axis="y", color="0.88", linewidth=.65)
                ax.set_ylim(ymin, 1.012)
                ax.set_xticks([750, 1500, 2250, 3000, 3600])
                ax.set_title(f"{title}  (n = {int(x.n.iloc[0])})\n{sky.capitalize()} sky")
                if j == 0:
                    ax.set_ylabel("Redshift recovery fraction")
                if i == 1:
                    ax.set_xlabel("Total shutter-open integration (s)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=9)
        fig.suptitle("JUST simulation at 4200 m: redshift recovery on the tested grid", fontsize=14)
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"recovery_20p5.{ext}", dpi=220, facecolor="white")
        plt.close(fig)
    result = {"input_curve_sha256": sha(table), "audit_sha256": sha(OUT / "independent_audit.json"),
              "script_sha256": sha(Path(__file__)), "outputs": {e: sha(OUT / f"recovery_20p5.{e}") for e in ("png", "pdf")},
              "meaning": "Circles are sample success fractions; open triangles are one-sided 95% Clopper-Pearson lower bounds adjusted for five tested times. Vertical segments only connect these two values at each actual sampled time; no time interpolation.",
              "y_min": ymin, "figsize_inches": [10.4, 7.2], "png_dpi": 220}
    (OUT / "figure_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
