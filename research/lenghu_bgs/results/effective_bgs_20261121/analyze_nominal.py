"""Audit, aggregate, and plot the completed nominal-condition Redrock scan."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_nominal_scan as scan
from build_analysis import sha
from effective_model import DepthModel


def main():
    progress=json.loads((scan.OUT/"progress.json").read_text())
    if progress["status"]!="complete" or progress["completed_spectra"]!=2664:
        raise ValueError("nominal scan is not complete")
    selected=scan.preflight()
    pieces=[]
    for start,count in scan.PLAN:
        spectra=scan.validate_input(start,count,selected)
        scan.validate_fit(start,spectra)
        pieces.append(pd.read_csv(scan.OUT/f"test_{start:04d}_redrock.csv"))
    rows=pd.concat(pieces,ignore_index=True)
    if (len(rows)!=2664 or rows.duplicated(["TARGETID","total_s"]).any() or
        set(rows.TARGETID)!=set(selected.TARGETID) or set(rows.total_s)!=set(scan.TIMES)):
        raise ValueError("combined nominal sample differs from frozen design")
    rows=rows.merge(selected[["TARGETID","r_dered","VI_Z","spectral_type"]],on="TARGETID",validate="many_to_one")
    np.testing.assert_allclose(rows.z_true_vi,rows.VI_Z,rtol=0,atol=1e-12)
    rows.to_csv(HERE/"nominal_individual_results.csv",index=False)
    out=[]
    for name,cut in (("r<19.5",19.5),("r<20.5",20.5)):
        group=rows[rows.r_dered<cut]
        for total in scan.TIMES:
            d=group[group.total_s==total];n=len(d);k=int(d.success.sum())
            expected=578 if name=="r<19.5" else 1332
            if n!=expected:raise ValueError("wrong group denominator")
            q=int(d.quality_pass.sum());nret=int(d.returned.sum())
            point=0 if k==0 else float(beta.ppf(.05,k,n-k+1))
            adjusted=0 if k==0 else float(beta.ppf(.05/len(scan.TIMES),k,n-k+1))
            out.append({"sample":name,"sky":"nominal","actual_s":total,"effective_s":float(total),
                        "n":n,"k_success":k,"success_fraction":k/n,"lower_pointwise":point,
                        "lower":adjusted,"n_returned":nret,"n_quality_pass":q})
    stats=pd.DataFrame(out)
    stats.to_csv(HERE/"nominal_success_curves.csv",index=False)
    # These divisions reuse previous study labels; subgroups are exploratory
    # and the two-time correction does not jointly cover the different groups.
    groups={"19.5<=r<=20":(rows.r_dered>=19.5)&(rows.r_dered<=20),
            "20<r<20.5":(rows.r_dered>20)&(rows.r_dered<20.5),
            "0.1<z<=0.25":(rows.VI_Z>.1)&(rows.VI_Z<=.25),
            "0.25<z<0.4":(rows.VI_Z>.25)&(rows.VI_Z<.4),
            "bpt_star_forming":rows.spectral_type=="bpt_star_forming",
            "weak_line_old":rows.spectral_type=="weak_line_old",
            "other_unclassified":rows.spectral_type=="other_unclassified"}
    sub=[]
    for name,mask in groups.items():
        for total in scan.TIMES:
            d=rows[mask&(rows.total_s==total)];n=len(d);k=int(d.success.sum())
            if not n:raise ValueError(f"empty subgroup {name}")
            sub.append({"group":name,"actual_s":total,"n":n,"k_success":k,
                        "success_fraction":k/n,
                        "lower_pointwise":0 if k==0 else float(beta.ppf(.05,k,n-k+1)),
                        "lower_two_times":0 if k==0 else float(beta.ppf(.05/len(scan.TIMES),k,n-k+1)),
                        "n_returned":int(d.returned.sum())})
    pd.DataFrame(sub).to_csv(HERE/"nominal_subgroup_success.csv",index=False)
    prior=pd.read_csv(scan.STUDY/"leader_195_205_20261116/extended_20p5/analysis_20261119/individual_results.csv")
    paired=[]
    for nominal_t,scene,scene_t in ((1817,"dark",1500),(3436,"dark",3000),(3436,"bright",3600)):
        a=rows[rows.total_s==nominal_t]
        b=prior[(prior.sky==scene)&(prior.total_s==scene_t)]
        p=a.merge(b,on="TARGETID",suffixes=("_nominal","_scene"),validate="one_to_one")
        if len(p)!=1332:raise ValueError("paired nominal/scene roster mismatch")
        for name,cut in (("r<19.5",19.5),("r<20.5",20.5)):
            q=p[p.r_dered_nominal<cut]
            n=len(q);aok=q.success_nominal.astype(bool);bok=q.success_scene.astype(bool)
            no=int((aok&~bok).sum());so=int((~aok&bok).sum());ndiff=no+so
            paired.append({"sample":name,"nominal_s":nominal_t,"scene":scene,"scene_s":scene_t,
                           "n":n,"nominal_success":int(aok.sum()),"scene_success":int(bok.sum()),
                           "nominal_only":no,"scene_only":so,
                           "difference_scene_minus_nominal":(so-no)/n,
                           "exploratory_mcnemar_p":binomtest(no,ndiff,.5).pvalue if ndiff else 1.})
    pd.DataFrame(paired).to_csv(HERE/"nominal_scene_paired_checks.csv",index=False)
    model=DepthModel()
    thresholds=[]
    for sample in ("r<19.5","r<20.5"):
        a=stats[stats['sample']==sample].sort_values("actual_s")
        for criterion,col in (("point","success_fraction"),("bonferroni_lower","lower")):
            passed=a[a[col]>=.95]
            if len(passed)==0:
                thresholds.append({"sample":sample,"criterion":criterion,"nominal_s":np.nan,
                                   "status":"not confirmed on the two tested nominal times",
                                   "dark_model_s":np.nan,"bright_model_s":np.nan})
            else:
                t=float(passed.iloc[0].actual_s)
                thresholds.append({"sample":sample,"criterion":criterion,"nominal_s":t,
                                   "status":"first tested nominal time reaching criterion",
                                   "dark_model_s":model.exposure_for_depth("dark",t),
                                   "bright_model_s":model.exposure_for_depth("bright",t)})
    pd.DataFrame(thresholds).to_csv(HERE/"nominal_validated_thresholds.csv",index=False)
    plot(stats)
    status={"status":"complete","attempted":len(rows),"returned":int(rows.returned.sum()),
            "times_s":scan.TIMES,"sample_n":{"r<19.5":578,"r<20.5":1332},
            "success_definition":"ZWARN=0, DELTACHI2>40, |zfit-zVI|/(1+zVI)<0.0033; all preselected test targets as denominator",
            "confidence":"one-sided 95% Clopper-Pearson with Bonferroni over the two nominal times within each group; no cross-group correction",
            "input_scope":"same DESI reference field as earlier simulation; results are conditional, not on-sky JUST performance",
            "progress_sha256":sha(scan.OUT/"progress.json"),"results_sha256":sha(HERE/"nominal_individual_results.csv"),
            "curves_sha256":sha(HERE/"nominal_success_curves.csv"),
            "subgroups_sha256":sha(HERE/"nominal_subgroup_success.csv"),
            "paired_checks_sha256":sha(HERE/"nominal_scene_paired_checks.csv"),
            "paired_scope":"post-hoc same-target tests, no multiple comparison correction",
            "script_sha256":sha(Path(__file__))}
    (HERE/"nominal_analysis_manifest.json").write_text(json.dumps(status,indent=2)+"\n")
    print(stats.to_string(index=False));print(pd.DataFrame(thresholds).to_string(index=False));print(json.dumps(status,indent=2))


def plot(nominal):
    existing=pd.read_csv(HERE/"mapped_recovery_curves.csv")
    plt.rcParams.update({"font.family":"serif","font.size":10,"pdf.fonttype":42})
    fig,axes=plt.subplots(1,2,figsize=(11.3,4.4),sharey=True,layout="constrained")
    for ax,sample in zip(axes,("r<19.5","r<20.5")):
        for sky,color,marker in (("dark","#0072B2","o"),("bright","#D55E00","s")):
            q=existing[(existing['sample']==sample)&(existing.sky==sky)].sort_values("actual_s")
            ax.plot(q.effective_s,q.success_fraction*100,color=color,marker=marker,lw=1.2,ms=4,label=sky.title()+" mapped")
            ax.vlines(q.effective_s,q.lower*100,q.success_fraction*100,color=color,lw=1)
            ax.plot(q.effective_s,q.lower*100,marker="_",ls="none",color=color,ms=6)
        q=nominal[nominal['sample']==sample].sort_values("actual_s")
        ax.plot(q.effective_s,q.success_fraction*100,color="#222222",marker="D",ls="none",ms=5,label="Nominal simulated")
        ax.vlines(q.effective_s,q.lower*100,q.success_fraction*100,color="#222222",lw=1)
        ax.plot(q.effective_s,q.lower*100,marker="_",ls="none",color="#222222",ms=6)
        ax.axhline(95,color="#333333",ls="--",lw=1)
        ax.set_title(f"{sample}, n={int(q.n.iloc[0])}")
        ax.set_xlabel("Nominal reference-equivalent time [s]")
        ax.set_xlim(left=0);ax.set_ylim(75,100.7);ax.grid(axis="y",alpha=.18)
    axes[0].set_ylabel("Redshift recovery [%]")
    axes[0].legend(frameon=False,fontsize=8.5,loc="lower right")
    fig.suptitle("JUST BGS-like recovery: SUM-coadd reference scale",fontsize=12)
    fig.supxlabel("Vertical marks: one-sided 95% lower bounds, adjusted over tested times separately per scene and sample.",fontsize=8)
    fig.savefig(HERE/"nominal_recovery.png",dpi=200,facecolor="white")
    fig.savefig(HERE/"nominal_recovery.pdf",facecolor="white")
    plt.close(fig)


if __name__=="__main__":main()
