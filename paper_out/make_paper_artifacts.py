"""Build the FAccT paper's figures and tables from committed result files (no new computation of results).

    python paper_out/make_paper_artifacts.py

Inputs (prereg/): holdout_confirmatory_results.json, h6_rate_units_results.json, materiality_results.json,
corrected_bootstrap_results.json, secondary_results.json, attributes_holdout_confirmatory_results.json,
attributes_descriptives.json, attributes_pilot_results.json; jev_pilot_out/repeat/scores.csv + jev_scores.csv (repeat noise).
Outputs: paper_out/figures/*.pdf|png, paper_out/tables/*.csv|tex.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

P = Path("prereg")
FIG, TAB = Path("paper_out/figures"), Path("paper_out/tables")
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# Validated categorical slots 1-3 (light surface); aqua < 3:1 contrast -> every figure is direct-labelled + tabled.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, INK2, SURFACE = "#9a9893", "#0b0b0b", "#52514e", "#fcfcfb"
FAM_COLOR = {"delivery": BLUE, "structure": ORANGE, "content": AQUA}
SYS_COLOR = {"GPT-4o-mini": BLUE, "GPT-5-mini": ORANGE, "Jev": AQUA}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "legend.frameon": False,
})


def load(name):
    return json.loads((P / name).read_text(encoding="utf-8"))


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_table(df: pd.DataFrame, name: str, caption: str):
    df.to_csv(TAB / f"{name}.csv", index=False)
    (TAB / f"{name}.tex").write_text(
        df.to_latex(index=False, escape=True, caption=caption, label=f"tab:{name}", float_format="%.3f"),
        encoding="utf-8")


# ---------------------------------------------------------------------------
# F1: where style variation comes from (TV, same complaint)
# ---------------------------------------------------------------------------

def repeat_noise_tv():
    r = pd.read_csv("jev_pilot_out/repeat/scores.csv").drop_duplicates("text_id").set_index("text_id")
    o = pd.read_csv("jev_pilot_out/jev_scores.csv").drop_duplicates("text_id").set_index("text_id")
    j = r.join(o, lsuffix="_r", rsuffix="_o", how="inner")
    a = j[[f"p{i}_fwd_r" for i in range(3)]].to_numpy()
    b = j[[f"p{i}_fwd_o" for i in range(3)]].to_numpy()
    return float(0.5 * np.abs(a - b).sum(1).mean()), float((a.argmax(1) != b.argmax(1)).mean())


def fig1(sec):
    h = sec["holdout_n1000"]["S1_hidden_variation"]
    rep_tv, rep_flip = repeat_noise_tv()
    rows = [("Same text, scored twice (exploratory)", rep_tv, GRAY),
            ("Same text, option order reversed", h["gpt-4o-mini"]["tv_same_text_option_reversal"], GRAY),
            ("Two paraphrases, same persona", h["cross_bank"]["tv_same_persona"], BLUE),
            ("Two paraphrases, different persona", h["cross_bank"]["tv_different_persona"], BLUE)]
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.55)
    for yi, (lab, v, _) in zip(y, rows):
        ax.text(v + 0.002, yi, f"{v:.3f}", va="center", color=INK, fontsize=8)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("Mean total-variation distance between Jev remedy distributions")
    share = h["cross_bank"]["persona_specific_share"]
    ax.set_title(f"Persona adds {100*share:.1f}% to paraphrase variation: arbitrariness, not profiling", loc="left")
    ax.set_xlim(0, 0.11)
    ax.grid(axis="y", visible=False)
    save(fig, "F1_variation_sources")
    pd.DataFrame([(r[0], r[1]) for r in rows], columns=["comparison", "mean_TV"]).to_csv(TAB / "F1_variation_sources.csv", index=False)
    return rep_tv, rep_flip


# ---------------------------------------------------------------------------
# F2: materiality - monetary-rate gaps with 95% CIs vs +/-1 pp (holdout)
# ---------------------------------------------------------------------------

def fig2(mat):
    rows = []
    m = mat["holdout_n1000"]
    for bank, gname in (("gpt-4o-mini", "GPT-4o-mini"), ("gpt-5-mini", "GPT-5-mini")):
        for sysname, key in ((gname, "GPT"), (f"Jev (matched to {gname})", "Jev_matched")):
            for dim, lab in (("ses", "SES high − low"), ("edu", "College − high school"), ("age", "Old − young")):
                s = m[bank][key][dim]
                rows.append({"system": sysname, "contrast": lab, "gap_pp": s["gap_pp"], "lo": s["gap_pp_ci95"][0],
                             "hi": s["gap_pp_ci95"][1], "AIR": s["AIR"], "cohen_h": s["cohen_h"],
                             "within_1pp_AIR>0.8": s["non_material_AIR_gt_0.80"]})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    systems = list(dict.fromkeys(df.system))
    colors = [BLUE, AQUA, ORANGE, AQUA]
    contrasts = list(dict.fromkeys(df.contrast))
    for ci_, c in enumerate(contrasts):
        for si, s in enumerate(systems):
            r = df[(df.system == s) & (df.contrast == c)].iloc[0]
            y = ci_ * (len(systems) + 1.2) + si
            ax.plot([r.lo, r.hi], [y, y], color=colors[si], lw=2, solid_capstyle="round")
            hollow = si == 3
            ax.plot(r.gap_pp, y, linestyle="none", ms=6, color=colors[si], mec=colors[si] if hollow else SURFACE,
                    mfc=SURFACE if hollow else colors[si], mew=1.5, marker="o" if si % 2 == 0 else "s")
    ax.axvspan(-1, 1, color="#eeede9", zorder=0)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks([ci_ * (len(systems) + 1.2) + 1.5 for ci_ in range(len(contrasts))], contrasts)
    ax.invert_yaxis()
    ax.set_xlabel("Within-complaint gap in monetary-remedy rate (percentage points), 95% CI")
    ax.set_title("Every persona gap and its 95% CI lies inside ±1 pp (shaded band)", loc="left")
    handles = [plt.Line2D([], [], color=colors[i], marker="o" if i % 2 == 0 else "s", lw=2, label=s,
                          mfc=SURFACE if i == 3 else colors[i], mec=colors[i]) for i, s in enumerate(systems)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=7)
    ax.grid(axis="y", visible=False)
    save(fig, "F2_materiality_monetary_gaps")
    write_table(df.round(3), "T2_materiality_holdout", "Monetary-remedy gaps, adverse impact ratios and Cohen's h (holdout, n = 1,000).")


# ---------------------------------------------------------------------------
# F3: decomposition of GPT-4o-mini tier gaps into delivery / structure / content
# ---------------------------------------------------------------------------

def fig3(att):
    a2 = att["H-A2"]
    fig, ax = plt.subplots(figsize=(6.2, 2.2))
    labels = {"edu": "College − high school", "ses": "SES high − low"}
    rows = []
    for i, dim in enumerate(("edu", "ses")):
        g = a2[dim]
        left = 0.0
        for fam in ("delivery", "structure", "content"):
            v = g[f"fam_{fam}"]
            ax.barh(i, v, left=left, color=FAM_COLOR[fam], height=0.5, edgecolor=SURFACE, linewidth=2)
            if v > 0.005:
                ax.text(left + v / 2, i, f"{fam}\n{v:+.3f}", ha="center", va="center", color="white", fontsize=7)
            else:
                ax.annotate(f"{fam} {v:+.3f}", (left + v / 2, i - 0.25), xytext=(left + v / 2, i - 0.42),
                            ha="center", fontsize=6.5, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
            left += v
        unexpl = g["gap"] - g["explained"]
        if unexpl > 0:
            ax.barh(i, unexpl, left=left, color="#d9d8d3", height=0.5, edgecolor=SURFACE, linewidth=2)
        ax.plot(g["gap_ci95"], [i + 0.38, i + 0.38], color=INK, lw=1)
        ax.plot(g["gap"], i + 0.38, "|", color=INK, ms=8)
        ax.text(g["gap_ci95"][1] + 0.001, i + 0.38, f"total gap {g['gap']:+.3f} ({100*g['share_explained']:.0f}% explained)",
                va="center", fontsize=7, color=INK)
        rows.append({"contrast": labels[dim], "gap": g["gap"], "gap_lo": g["gap_ci95"][0], "gap_hi": g["gap_ci95"][1],
                     "share_explained": g["share_explained"], **{f: g[f"fam_{f}"] for f in ("delivery", "structure", "content")},
                     **{f"{f}_lo": g[f"fam_{f}_ci95"][0] for f in ("delivery", "structure", "content")},
                     **{f"{f}_hi": g[f"fam_{f}_ci95"][1] for f in ("delivery", "structure", "content")}})
    ax.set_yticks([0, 1], [labels["edu"], labels["ses"]])
    ax.set_ylim(1.62, -0.72)                      # inverted, with headroom for the small-segment labels
    ax.set_xlabel("Contribution to GPT-4o-mini's within-complaint tier gap (Gelbach decomposition, holdout)")
    ax.set_title("GPT-4o-mini's education and SES gaps run mostly through delivery, not content", loc="left")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 0.05)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=FAM_COLOR[f], label=f) for f in ("delivery", "structure", "content")] +
              [Patch(color="#d9d8d3", label="unexplained")], loc="upper center", bbox_to_anchor=(0.5, -0.3),
              ncol=4, fontsize=7)
    save(fig, "F3_decomposition_delivery_structure_content")
    write_table(pd.DataFrame(rows).round(4), "T3_decomposition_holdout",
                "Decomposition of GPT-4o-mini's tier gaps into attribute families (holdout, n = 1,000; 95% block-bootstrap CIs).")


# ---------------------------------------------------------------------------
# F4: recognition moderation
# ---------------------------------------------------------------------------

def fig4(att):
    a3 = att["H-A3"]
    rows = [("Jev recognises the education cue", a3["recognised_gap"], a3["recognised_ci95"], BLUE),
            ("Education cue ambiguous", a3["ambiguous_gap"], a3["ambiguous_ci95"], GRAY)]
    fig, ax = plt.subplots(figsize=(6.2, 1.8))
    for i, (lab, v, c, col) in enumerate(rows):
        ax.barh(i, v, color=col, height=0.5)
        ax.plot(c, [i, i], color=INK, lw=1)
        ax.text(max(c[1], v) + 0.002, i, f"{v:+.3f} [{c[0]:+.3f}, {c[1]:+.3f}]", va="center", fontsize=8, color=INK)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks([0, 1], [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("GPT-4o-mini college − high-school tier gap (complaint-demeaned), 95% CI")
    ax.set_title(f"The education effect appears only where the style cue is recognisable (difference {a3['difference']:+.3f})", loc="left")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-0.01, 0.1)
    save(fig, "F4_recognition_moderation")


# ---------------------------------------------------------------------------
# F5: persona effects on attributes (heatmap, GPT-4o-mini bank)
# ---------------------------------------------------------------------------

def fig5(desc):
    eff = desc["D2_persona_effects"]["gpt-4o-mini"]
    order = sorted(eff, key=lambda k: ({"delivery": 0, "structure": 1, "content": 2}[eff[k]["family"]], -eff[k]["eta2_persona"]))
    cols = [("age:old-young_std", "Old − young"), ("edu:college-high_school_std", "College − HS"),
            ("ses:high-low_std", "High − low SES"), ("geo:urban-rural_std", "Urban − rural")]
    M = np.array([[eff[a][c] for c, _ in cols] for a in order])
    fig, ax = plt.subplots(figsize=(5.2, 6.0))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", [ORANGE, "#f2f1ed", BLUE])
    im = ax.imshow(M, cmap=cmap, vmin=-1.2, vmax=1.2, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(M[i, j]) > 0.7 else INK)
    ax.set_xticks(range(len(cols)), [c[1] for c in cols], fontsize=7)
    names = [a.replace("_E", "").replace("obj_", "objective: ").replace("_", " ") for a in order]
    ax.set_yticks(range(len(order)), [f"{n} ({eff[a]['family'][0].upper()})" for n, a in zip(names, order)], fontsize=7)
    ax.grid(False)
    fams = [eff[a]["family"] for a in order]
    for k in range(1, len(fams)):
        if fams[k] != fams[k - 1]:
            ax.axhline(k - 0.5, color=INK, lw=1.2)
    cb = fig.colorbar(im, ax=ax, shrink=0.5)
    cb.set_label("Standardised within-complaint effect", fontsize=7)
    ax.set_title("What persona rewrites change (GPT-4o-mini rewrites; D/S/C = family)", loc="left", fontsize=9)
    save(fig, "F5_persona_effects_on_attributes")
    pd.DataFrame(M, index=order, columns=[c[1] for c in cols]).round(3).to_csv(TAB / "F5_persona_effects_on_attributes.csv")


# ---------------------------------------------------------------------------
# F6: individual arbitrariness vs largest group gap (pooled)
# ---------------------------------------------------------------------------

def fig6(mat):
    m = mat["pooled_n3000"]
    rows = []
    for bank, gname in (("gpt-4o-mini", "GPT-4o-mini"), ("gpt-5-mini", "GPT-5-mini")):
        for key, name in (("GPT", gname), ("Jev_matched", f"Jev (matched to {gname})")):
            s = m[bank][key]
            maxgap = max(abs(s[d]["gap_pp"]) for d in ("age", "edu", "ses", "geo"))
            rows.append({"system": name, "p_two_guises_disagree_pct": 100 * s["arbitrariness"]["p_two_guises_disagree_monetary"],
                         "complaints_not_unanimous_pct": 100 * s["arbitrariness"]["share_complaints_monetary_not_unanimous"],
                         "largest_group_gap_pp": maxgap})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    y = np.arange(len(df))[::-1]
    ax.barh(y + 0.18, df.p_two_guises_disagree_pct, height=0.34, color=BLUE, label="Individual: two style versions get different monetary decisions (%)")
    ax.barh(y - 0.18, df.largest_group_gap_pp, height=0.34, color=ORANGE, label="Group: largest persona gap in monetary rate (pp)")
    for yi, a, g in zip(y, df.p_two_guises_disagree_pct, df.largest_group_gap_pp):
        ax.text(a + 0.1, yi + 0.18, f"{a:.1f}%", va="center", fontsize=7, color=INK)
        ax.text(g + 0.1, yi - 0.18, f"{g:.2f} pp", va="center", fontsize=7, color=INK)
    ax.set_yticks(y, df.system, fontsize=8)
    ax.set_xlabel("Percent / percentage points (pooled, n = 3,000)")
    ax.set_title("Individual-level arbitrariness dwarfs group-level profiling", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=1, fontsize=7)
    ax.grid(axis="y", visible=False)
    save(fig, "F6_arbitrariness_vs_group_gap")
    write_table(df.round(2), "T4_arbitrariness_vs_profiling", "Individual-level arbitrariness vs group-level gaps in monetary decisions (pooled, n = 3,000).")


# ---------------------------------------------------------------------------
# T1: hypothesis summary
# ---------------------------------------------------------------------------

def t1(hc, att, cbt, sec):
    s0 = sec["holdout_n1000"]["S0_H4_difference_scale"]
    rows = [
        ("H1", "Persona-specific share of style variation < 5%", f"{100*hc['H1']['estimate']:.1f}% [{100*hc['H1']['ci95'][0]:.1f}, {100*hc['H1']['ci95'][1]:.1f}]", "Supported"),
        ("H2", "Directional persona effects (12 contrasts)", f"{sum(v['supported'] for b in hc['H2'].values() for m in b.values() for k, v in m.items() if ':' in k)}/12 supported", "Mostly supported"),
        ("H3a", "E[tier] reduces arbitrariness (residual share)", f"4o: {hc['H3']['gpt-4o-mini']['H3a_residual_share_argmax']:.2f}→{hc['H3']['gpt-4o-mini']['H3a_residual_share_e_tier']:.2f}; 5: {hc['H3']['gpt-5-mini']['H3a_residual_share_argmax']:.2f}→{hc['H3']['gpt-5-mini']['H3a_residual_share_e_tier']:.2f}", "Supported"),
        ("H3b", "E[tier] effect on systematic disparity (estimation)", f"4o: {cbt['holdout_n1000']['gpt-4o-mini']['H3b_D_argmax_minus_D_etier']['estimate']:+.4f} {np.round(cbt['holdout_n1000']['gpt-4o-mini']['H3b_D_argmax_minus_D_etier']['corrected'], 4).tolist()}", "No reduction"),
        ("H4", "Probability averaging not superior to hard voting", f"paraphrase Δ {s0['paraphrase_2']['difference']:+.4f}; orders(4o) Δ {s0['orders_6_gpt-4o-mini']['difference']:+.4f}", "Supported"),
        ("H5", "Uncertainty referral creates disparity (5-mini bank)", f"p = {hc['H5']['gpt-5-mini']['persona_sd_perm_p']:.2f}", "Not supported"),
        ("H6", "Jev monetary disparity < GPT-4o-mini (matched volume)", f"Δ {hc['H6']['gpt-4o-mini']['delta_gpt_minus_jev']:+.4f} {np.round(cbt['holdout_n1000']['gpt-4o-mini']['H6_D_gpt_minus_D_jev']['corrected'], 4).tolist()}", "Inconclusive"),
        ("H-A1", "Rewrites preserve content (η² < 0.05)", f"max upper bound {max(v['upper95_one_sided'] for k, v in att['H-A1'].items() if k != 'supported'):.3f}", "Supported"),
        ("H-A2", "Delivery drives GPT-4o-mini edu/SES gaps", f"edu {100*att['H-A2']['edu']['share_explained']:.0f}%, SES {100*att['H-A2']['ses']['share_explained']:.0f}% explained", "Supported"),
        ("H-A3", "Effect only where cue is recognisable", f"difference {att['H-A3']['difference']:+.3f} {np.round(att['H-A3']['difference_ci95'], 3).tolist()}", "Supported"),
        ("H-A4", "Jev infers persona from delivery, not content", "6/6 differences > 0", "Supported"),
        ("H-A5", "Jev remedy judgment driven by content", f"R² content {att['H-A5']['gpt-4o-mini']['r2_content']:.2f} vs delivery {att['H-A5']['gpt-4o-mini']['r2_delivery']:.2f}", "Supported"),
    ]
    write_table(pd.DataFrame(rows, columns=["Hypothesis", "Claim", "Estimate [95% CI]", "Verdict"]),
                "T1_hypotheses", "Pre-registered hypotheses and confirmatory results (holdout, n = 1,000).")


def t5(desc, pilot):
    cert = desc["D3_certainty_holdout"]
    retest = pilot["QC_certainty"]
    rows = [{"attribute": a, "family": v["family"], "mean_norm_entropy_holdout": v["mean_norm_entropy"],
             "share_choice_p_ge_0.8": v["share_high_confidence_choice_p_ge_0.8"],
             "retest_spearman_pilot": retest[a].get("retest_spearman_E"), "retest_choice_agree_pilot": retest[a].get("retest_choice_agree")}
            for a, v in cert.items()]
    write_table(pd.DataFrame(rows).round(3), "T5_attribute_certainty_reliability",
                "Jev's certainty (holdout) and test-retest reliability (pilot) per attribute.")
    rows = []
    for dim, d in desc["D1_gelbach_by_attribute_gpt4o_tier"].items():
        for a, r in list(d.items())[:6]:
            rows.append({"contrast": dim, "attribute": a.replace("_E", ""), "family": r["family"], "contribution": r["contribution"],
                         "lo": r["ci95"][0], "hi": r["ci95"][1]})
    write_table(pd.DataFrame(rows).round(4), "T6_top_attribute_contributions",
                "Largest attribute contributions to GPT-4o-mini's tier gaps (holdout; 95% block-bootstrap CIs).")


def main():
    hc = load("holdout_confirmatory_results.json")
    mat = load("materiality_results.json")
    cbt = load("corrected_bootstrap_results.json")
    sec = load("secondary_results.json")
    att = load("attributes_holdout_confirmatory_results.json")
    desc = load("attributes_descriptives.json")
    pilot = load("attributes_pilot_results.json")
    rep_tv, rep_flip = fig1(sec)
    fig2(mat)
    fig3(att)
    fig4(att)
    fig5(desc)
    fig6(mat)
    t1(hc, att, cbt, sec)
    t5(desc, pilot)
    (TAB / "repeat_noise.json").write_text(json.dumps({"mean_tv_rescore": rep_tv, "tier_flip_rate_rescore": rep_flip}, indent=2))
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))
    print("tables:", sorted(p.name for p in TAB.glob("*.csv")))


if __name__ == "__main__":
    main()
