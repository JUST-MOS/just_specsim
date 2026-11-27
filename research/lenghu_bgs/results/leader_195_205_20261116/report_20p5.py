"""Build a compact Chinese report from the completed, independently audited scan."""
from pathlib import Path
import json
import math
import pandas as pd

import analyze_20p5 as analysis

HERE = Path(__file__).resolve().parent
OUT = HERE / "extended_20p5/analysis_20261119"


def fmt_time(x):
    return "未确认" if pd.isna(x) else f"{int(x)}"


def lower_time(row):
    if row.status_adjusted == "insufficient_even_if_all_success":
        return "样本不足"
    return fmt_time(row.time_s_adjusted_lower)


def main():
    manifest = json.loads((OUT / "analysis_manifest.json").read_text())
    audit = json.loads((OUT / "independent_audit.json").read_text())
    figure = json.loads((OUT / "figure_manifest.json").read_text())
    if (manifest["status"] != "complete" or audit["status"] != "passed" or
        audit["analysis_manifest_sha256"] != analysis.sha(OUT / "analysis_manifest.json") or
        figure["input_curve_sha256"] != analysis.sha(OUT / "success_curves.csv")):
        raise ValueError("complete, audited analysis and figure are required")
    curves = pd.read_csv(OUT / "success_curves.csv")
    thresholds = pd.read_csv(OUT / "exposure_thresholds.csv")
    def one(group_type, group, sky, t):
        d = curves[(curves.group_type == group_type) & (curves.group == group) &
                   (curves.sky == sky) & (curves.time_s == t)]
        if len(d) != 1:
            raise ValueError(f"expected one curve entry: {group_type} {group} {sky} {t}")
        return d.iloc[0]
    def threshold(group_type, group, sky, level):
        d = thresholds[(thresholds.group_type == group_type) & (thresholds.group == group) &
                       (thresholds.sky == sky) & (thresholds.target == level)]
        if len(d) != 1:
            raise ValueError("expected one threshold entry")
        return d.iloc[0]
    lines = [
        "# JUST：r₀ < 20.5 的 4200 m 亮／暗夜红移恢复",
        "",
        "## 样本与成功判据",
        "",
        "共 1332 个测试星系：复用原有 r₀≤20 的 891 个，新增 20<r₀<20.5 的 441 个；"
        "r₀ 为银河系前景消光校正后的 BASS-r 总 AB 星等，0.1<z_VI<0.4。"
        "目标来自同一 DESI EDR 参考深场，经过参考谱、形态和模型质量筛选，不代表全部星系。",
        "",
        "每个星系在每个情景都进入分母，包括无拟合结果者。成功要求 Redrock 返回有限红移、"
        "ZWARN=0、DELTACHI2>40，且 |z_fit−z_VI|/(1+z_VI)<0.0033。参考红移是 DESI VI，"
        "与构造模拟源谱的数据并非完全独立。质量阈值沿用原有较亮样本的校准检查；"
        "本次新增的 104 个较暗校准源没有拟合，尚未独立验证该阈值对较暗样本的表现。",
        "",
        "## 全样本随曝光时间的结果",
        "",
        "| 快门总积分 / s | 暗夜成功数 / 1332 | 暗夜成功率 | 暗夜调整后下限 | 亮夜成功数 / 1332 | 亮夜成功率 | 亮夜调整后下限 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for t in analysis.scan.TIMES:
        a, b = one("all", "all", "dark", t), one("all", "all", "bright", t)
        lines.append(f"| {t} | {a.k_success} | {100*a.success_fraction:.2f}% | {100*a.cp95_lower_bonferroni_times:.2f}% | "
                     f"{b.k_success} | {100*b.success_fraction:.2f}% | {100*b.cp95_lower_bonferroni_times:.2f}% |")
    lines += ["", "## 首次达到目标的已测时间", "",
              "| 分组 | 夜空 | 目标 | 点估计 / s | 逐时间单侧下限 / s | 五时间校正后单侧下限 / s |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for gt, name, label in (("all", "all", "全部 r₀<20.5"),
                            ("magnitude", "20_lt_r_lt_20p5", "20<r₀<20.5")):
        for sky, sname in (("dark", "暗夜"), ("bright", "亮夜")):
            for target in (.90, .95):
                row = threshold(gt, name, sky, target)
                lines.append(f"| {label} | {sname} | {target:.0%} | {fmt_time(row.time_s_point)} | "
                             f"{fmt_time(row.time_s_pointwise_lower)} | {lower_time(row)} |")
    lines += ["", "主判据是单侧 95% Clopper–Pearson 统计下限达到目标，"
              "并在每个分组、每种天空内对五个时间点做 Bonferroni 校正。"
              "逐时间下限不做该校正；点估计达到目标不代表已按下限确认。"
              "若所有已测时间均未满足条件，记为“未确认”，这不等于证明真实最短时间大于 3600 s。"
              "各时间使用独立噪声实现及不同分次，成功率不保证单调，时间网格没有进行插值优化。",
              "", "较暗子样本在 2250 s 时暗夜为 417/441（94.56%），亮夜为 419/441（95.01%）；"
              "亮夜比暗夜多成功 2 个目标，这是本次有限样本和噪声实现下的结果，不能据此认定亮夜更有利。"
              "两种天空在这一子样本中均未达到五时间校正后的 95% 下限。",
              "", "## 3600 s 分组结果", "",
              "| 分组 | 夜空 | n | 成功数 | 成功率 | 调整后下限 | t₉₅ 点估计 / s | t₉₅ 调整下限 / s |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    groups = [("magnitude", "r_le_19", "r₀≤19"),
              ("magnitude", "19_lt_r_le_20", "19<r₀≤20"),
              ("magnitude", "20_lt_r_lt_20p5", "20<r₀<20.5"),
              ("redshift", "z_0p1_0p25", "0.1<z≤0.25"),
              ("redshift", "z_0p25_0p4", "0.25<z<0.4"),
              ("spectral_type", "bpt_star_forming", "BPT 星形成候选"),
              ("spectral_type", "weak_line_old", "弱线老族候选"),
              ("spectral_type", "other_unclassified", "其他／未归类")]
    for gt, name, label in groups:
        for sky, sname in (("dark", "暗夜"), ("bright", "亮夜")):
            row = one(gt, name, sky, 3600)
            tr = threshold(gt, name, sky, .95)
            lines.append(f"| {label} | {sname} | {row.n} | {row.k_success} | {100*row.success_fraction:.2f}% | "
                         f"{100*row.cp95_lower_bonferroni_times:.2f}% | {fmt_time(tr.time_s_point)} | {lower_time(tr)} |")
    lines += ["", "“样本不足”表示即使该组全部成功，五时间校正后的下限也不能达到 95%。"
              "不同分组、天空和阈值没有再进行联合多重检验校正；交叉分组及光纤星等／颜色分组见 CSV。",
              "", "## 3600 s 的质量通过及纯度", "",
              "| 夜空 | 质量通过 | 正确复原 | 纯度及逐情景双侧 95% 区间 | 质量不合格 | 质量通过但错误 | 无结果 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for sky, sname in (("dark", "暗夜"), ("bright", "亮夜")):
        row = one("all", "all", sky, 3600)
        purity = "无质量通过结果" if not math.isfinite(row.purity) else (
            f"{100*row.purity:.2f}% [{100*row.purity_cp95_lo:.2f}%, {100*row.purity_cp95_hi:.2f}%]")
        lines.append(f"| {sname} | {row.quality_pass} | {row.k_success} | {purity} | {row.quality_fail} | "
                     f"{row.wrong_redshift_quality_pass} | {row.no_result} |")
    lines += ["", "纯度分母为质量通过数；区间只表示该情景内有限样本的统计不确定性。",
              "", "## 模型与适用范围", "",
              "使用既有冻结的 4200 m 条件化天光外推和 Qiu+ESO 大气拼接，"
              "而非后来构建的平滑大气候选；天空光谱不是冷湖现场实测光谱。参考条件为 airmass=1.2、"
              "800 nm 处 seeing FWHM=0.8″；暗夜月亮在地平线下，亮夜月面照亮约 50%、"
              "月亮高度 45°、月目标角距 60°。",
              "",
              "沿用原 JUST ETC 仪器设定：175 μm 光纤入口直径、约 1.362″ 视场中心角直径，"
              "4 e⁻/pixel/read 读出噪声、7 空间像元等权抽取；这是工程基线，未确认为最终 JUST 设计。"
              "源形态近似圆形指数分布；Redrock 使用 JUST 理论 LSF 与逐源 DESI 参考分辨率矩的 Gaussian 近似组合的响应矩阵。"
              "模拟源谱按消光校正总星等归一，源响应再按各视线的 E(B−V) 单独施加银河系前景消光；"
              "几何吞吐量计算中的 ebv=0 不代表省略了这一逐源消光。",
              "",
              "750、1500、2250、3000 s 分别为 1、2、3、4 次 750 s；3600 s 为 4×750+600 s。"
              "表中均是快门总积分，不含读出及其他观测开销。统计下限依赖所选深场样本近似独立同分布，"
              "不包含仪器、天光、大气、源谱选择或场间差异的系统误差。新增 8 星系试跑已事先查看，"
              "故本扩展分析不能描述为完全盲测。",
              "", "## 产物与核验", "",
              "- [逐时间及分组曲线](success_curves.csv)、[曝光阈值](exposure_thresholds.csv)、"
              "[逐源结果](individual_results.csv)。",
              "- [独立统计核验](independent_audit.json)、[输入与结果清单](analysis_manifest.json)、"
              "[旧结果回归核对](old_analysis_reproduction.json)、[图表 PDF](recovery_20p5.pdf)。",
              "- 可复现代码：[扫描驱动](../../run_extension.py)、[合并分析](../../analyze_20p5.py)、"
              "[独立审计](../../audit_20p5.py)、[作图](../../plot_20p5.py)、[报告生成](../../report_20p5.py)；"
              "[批次运行状态](../run/progress.json)。",
              "- 图示为已测试的模拟网格点；圆点是样本成功率，空心倒三角是五时间点校正后的单侧 95% 下限。"
              "科学绘图检查参考 [Kassis et al. (2026), Scientific Agent Skills](https://arxiv.org/abs/2609.00065)。",
              "", "![4200 m 亮暗夜红移恢复](recovery_20p5.png)", ""]
    (OUT / "RESULTS_zh.md").write_text("\n".join(lines))
    print(OUT / "RESULTS_zh.md")


if __name__ == "__main__":
    main()
