"""Check the completed BGS report bundle and write a compact SHA-256 manifest."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    progress = json.loads((HERE / "nominal_run/v2/progress.json").read_text())
    analysis = json.loads((HERE / "nominal_analysis_manifest.json").read_text())
    audit = json.loads((HERE / "nominal_independent_audit.json").read_text())
    if progress["status"] != "complete" or progress["completed_spectra"] != 2664:
        raise ValueError("nominal run is not complete")
    if (analysis["status"] != "complete" or analysis["attempted"] != 2664 or
        not 0 <= analysis["returned"] <= 2664 or audit["status"] != "passed" or
        audit["attempted"] != 2664 or audit["returned"] != analysis["returned"]):
        raise ValueError("analysis or independent audit incomplete")
    outputs = ["RESULTS_zh.md", "MODEL_zh.md", "PRELIMINARY_zh.md", "WORK_LOG.md",
               "effective_model.py", "depth_inputs.npz", "model_manifest.json",
               "weighted_effective_model.py", "weighted_model_audit.json", "weighted_thresholds.csv", "indirect_weighted_thresholds.csv",
               "analyze_nominal.py", "audit_nominal_results.py", "analyze_weighted_thresholds.py",
               "generate_report.py", "verify_delivery.py", "nominal_run/v2/run_plan.json",
               "nominal_run/v2/progress.json", "nominal_bgs_complete_early.csv", "nominal_bgs_complete_early.json",
               "nominal_analysis_manifest.json", "nominal_independent_audit.json",
               "nominal_success_curves.csv", "nominal_validated_thresholds.csv",
               "nominal_individual_results.csv", "nominal_subgroup_success.csv",
               "nominal_scene_paired_checks.csv", "nominal_recovery.png", "nominal_recovery.pdf",
               "coadd_boundary.png", "coadd_boundary.pdf", "coadd_boundary_manifest.json",
               "mapped_recovery_curves.csv", "paired_transfer_check.csv",
               "split_boundary_sensitivity.json", "weighted_coadd_sensitivity.json",
               "smooth_atmosphere_sensitivity.json", "run_nominal_prefetch.py",
               "reuse_staged_input.py", "nominal_run/prefetch_2worker/stop_record.json",
               "nominal_run/prefetch_2worker/reused_input.json"]
    for name in outputs:
        if not (HERE / name).is_file():
            raise FileNotFoundError(name)
    missing = []
    for name in ("RESULTS_zh.md", "MODEL_zh.md", "PRELIMINARY_zh.md"):
        text = (HERE / name).read_text()
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if target.startswith(("https://", "http://", "#")):
                continue
            if not (HERE / target.split("#", 1)[0]).exists():
                missing.append((name, target))
    if missing:
        raise FileNotFoundError(f"broken local links: {missing}")
    manifest = {
      "status": "passed", "generated_utc": datetime.now(timezone.utc).isoformat(),
      "description": "Research-only conditional JUST BGS effective-depth delivery; output identities and local links checked.",
      "nominal_attempted": analysis["attempted"], "nominal_returned": analysis["returned"],
      "local_markdown_links": "passed",
      "sha256": {name: sha(HERE / name) for name in outputs},
    }
    (HERE / "delivery_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "sha256"}, indent=2))


if __name__ == "__main__":
    main()
