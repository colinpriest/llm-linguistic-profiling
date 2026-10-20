"""Frozen analysis for the attribute pilot (pre-registered in prereg/attributes_prereg.md).

    python prereg/attributes_pilot_analysis.py

Inputs: jev_pilot_out/attributes_pilot/scores.csv (Jev attribute scores, 200 complaints),
jev_pilot_out/attributes_repeat/scores.csv (test-retest), jev_pilot_out/jev_scores.csv
(Jev persona estimates and remedy probabilities), the frozen GPT persona decisions CSV,
GPT persona estimates from the LLM cache, and texts (jev_pilot_out/texts_exploratory.parquet).
Output: jev_pilot_out/attributes_pilot/pilot_results.json and pilot_report.md.

All analyses are within-complaint (complaint fixed effects) and per rewrite bank.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import jev_attributes as ja  # noqa: E402
from src.cache import ResponseCache  # noqa: E402

SEED = 20261020
N_BOOT = 500
BANKS = ["gpt-4o-mini", "gpt-5-mini"]
RUN_TS = "20260220_101746"
OUT = Path("jev_pilot_out/attributes_pilot")
ATTRS = list(ja.ATTRIBUTES)
OBJECTIVE = ["obj_sentence_len", "obj_fk_grade", "obj_emotion_abs"]   # objective delivery measures
FAMILY = dict(ja.FAMILY, **{k: "delivery" for k in OBJECTIVE})
CONTRASTS = {"age": ("old", "young"), "edu": ("college", "high_school"), "ses": ("high", "low"),
             "geo": ("urban", "rural")}
# Go / no-go thresholds (pre-registered)
G1_ENTROPY, G1_SD, G1_MIN_ATTRS = 0.90, 0.05, 10
G2_ETA, G2_MIN_ATTRS = 0.05, 3
G3_SHARE = 0.25
RELIABILITY_DROP = 0.50
# Amendment 2a (persona uncertainty; added 20 Oct 2026 before any substantive pilot analysis)
PERSONA_LEVELS = {"age": ["young", "old"], "geo": ["urban", "suburban", "rural"],
                  "edu": ["high_school", "college"], "ses": ["low", "middle", "high"]}
RECOGNISED = 0.60          # Jev P(intended persona level) at or above this = "recognised" guise
rng = np.random.default_rng(SEED)


def persona_parts(label: str) -> Dict[str, str]:
    return {"age": label.split("_")[0], "geo": label.split("_")[1],
            "edu": "high_school" if "_high_school_" in label else "college", "ses": label.split("_")[-1]}


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def objective_metrics(texts: pd.Series) -> pd.DataFrame:
    import textstat
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    rows = []
    for t in texts:
        sents = [s for s in re.split(r"[.!?]+\s+|\n+", t) if s.strip()]
        words = re.findall(r"\b\w+\b", t)
        rows.append({"obj_sentence_len": len(words) / max(1, len(sents)),
                     "obj_fk_grade": textstat.flesch_kincaid_grade(t),
                     "obj_emotion_abs": abs(sia.polarity_scores(t)["compound"])})
    return pd.DataFrame(rows, index=texts.index)


def gpt_persona(texts: pd.DataFrame) -> pd.DataFrame:
    cache = ResponseCache("./cache")
    age = {"young": 0.0, "old": 1.0}
    edu = {"high_school": 0.0, "college": 1.0}
    ses = {"low": 0.0, "middle": 0.5, "high": 1.0}
    rows = []
    for tx, m in zip(texts.text, texts.rewrite_model):
        r = cache.get(m, "persona_estimate", tx) if m else None
        r = r or {}
        rows.append({"gpt_age_old": age.get(r.get("age")), "gpt_edu_college": edu.get(r.get("education")),
                     "gpt_ses": ses.get(r.get("socioeconomic")),
                     "gpt_markers": "; ".join(r.get("linguistic_markers") or [])})
    return pd.DataFrame(rows, index=texts.index)


def assemble() -> pd.DataFrame:
    a = pd.read_csv(OUT / "scores.csv")
    a = a[a.source == "guise"].drop_duplicates(["case_id", "rewrite_model", "persona_label"])
    texts = pd.read_parquet("jev_pilot_out/texts_exploratory.parquet")[["text_id", "text"]].drop_duplicates("text_id")
    a = a.merge(texts, on="text_id", how="left")
    a = pd.concat([a, objective_metrics(a.text), gpt_persona(a)], axis=1)
    js = pd.read_csv("jev_pilot_out/jev_scores.csv")
    pcols = ["age_p_young", "age_p_old", "geo_p_urban", "geo_p_suburban", "geo_p_rural",
             "edu_p_high_school", "edu_p_college", "ses_p_low", "ses_p_middle", "ses_p_high"]
    js = js[js.source == "guise"][["text_id", "e_tier", "p2_fwd"] + pcols]
    js = js.drop_duplicates("text_id").assign(jev_ses=lambda d: d.ses_p_high - d.ses_p_low)
    for dim, levels in PERSONA_LEVELS.items():                     # Jev persona ambiguity (normalised entropy)
        p = js[[f"{dim}_p_{lv}" for lv in levels]].to_numpy().clip(1e-12)
        js[f"jev_{dim}_entropy"] = -(p * np.log(p)).sum(1) / math.log(len(levels))
    a = a.merge(js, on="text_id", how="left")
    dec = pd.read_csv(f"output/results_persona_decisions_{RUN_TS}.csv", usecols=["case_id", "model", "variant", "tier"])
    dec = dec.rename(columns={"model": "rewrite_model", "variant": "persona_label", "tier": "gpt_tier"})
    a = a.merge(dec.drop_duplicates(["case_id", "rewrite_model", "persona_label"]),
                on=["case_id", "rewrite_model", "persona_label"], how="left")
    a["gpt_monetary"] = (a.gpt_tier == 2).astype(float).where(a.gpt_tier.notna())
    parts = a.persona_label.map(persona_parts)
    for d in ("age", "geo", "edu", "ses"):
        a[f"p_{d}"] = parts.map(lambda x: x[d])
    return a


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def demean(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return df[cols] - df.groupby("case_id")[cols].transform("mean")


def ols_within(df: pd.DataFrame, y: str, xs: List[str]) -> Dict[str, Any]:
    d = df.dropna(subset=[y] + xs)
    Y = demean(d, [y])[y].to_numpy()
    X = demean(d, xs)
    sd = X.std(ddof=0).replace(0, np.nan)
    Xz = (X / sd).fillna(0).to_numpy()
    beta, *_ = np.linalg.lstsq(Xz, Y, rcond=None)
    r2 = 1 - ((Y - Xz @ beta) ** 2).sum() / (Y ** 2).sum() if (Y ** 2).sum() > 0 else float("nan")
    return {"n": len(d), "r2_within": float(r2), "beta_std": dict(zip(xs, map(float, beta))),
            "beta_raw": dict(zip(xs, map(float, beta / sd.fillna(np.inf).to_numpy())))}


def eta_persona(df: pd.DataFrame, col: str) -> float:
    w = df.pivot_table(index="case_id", columns="persona_label", values=col).dropna()
    X = w.to_numpy()
    rm, cm, gm = X.mean(1, keepdims=True), X.mean(0, keepdims=True), X.mean()
    ss_p = X.shape[0] * ((cm - gm) ** 2).sum()
    ss_r = ((X - rm - cm + gm) ** 2).sum()
    return float(ss_p / (ss_p + ss_r)) if ss_p + ss_r > 0 else float("nan")


def entropy3(df: pd.DataFrame, a: str) -> float:
    p = df[[f"{a}_p_{lv}" for lv in ja.LEVELS]].to_numpy().clip(1e-12)
    return float(np.mean(-(p * np.log(p)).sum(1) / math.log(3)))


def gelbach(df: pd.DataFrame, y: str, xs: List[str], dim: str) -> Dict[str, Any]:
    """Order-invariant decomposition of the within-complaint persona gap (hi - lo on `dim`)
    into attribute contributions: gap_k = (hi-lo difference in attribute k) * beta_k (full model)."""
    hi, lo = CONTRASTS[dim]
    d = df[df[f"p_{dim}"].isin([hi, lo])].dropna(subset=[y] + xs)
    def one(dd: pd.DataFrame) -> Dict[str, float]:
        fit = ols_within(dd, y, xs)
        ind = (dd[f"p_{dim}"] == hi)
        gap = float(dd.loc[ind, y].mean() - dd.loc[~ind, y].mean())
        contrib = {x: float((dd.loc[ind, x].mean() - dd.loc[~ind, x].mean()) * fit["beta_raw"][x]) for x in xs}
        fam = {f: sum(v for k, v in contrib.items() if FAMILY[k] == f) for f in ("delivery", "structure", "content")}
        return {"gap": gap, "explained": sum(contrib.values()), **{f"fam_{f}": v for f, v in fam.items()}, "contrib": contrib}
    est = one(d)
    d = d.reset_index(drop=True)
    groups = list(d.groupby("case_id").indices.values())        # row positions per complaint
    boots = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(groups), len(groups))
        pos = np.concatenate([groups[i] for i in pick])
        newid = np.concatenate([np.full(len(groups[i]), j) for j, i in enumerate(pick)])
        boots.append(one(d.iloc[pos].assign(case_id=newid)))
    def ci(key: str) -> List[float]:
        v = np.array([b[key] for b in boots])
        return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
    share = est["explained"] / est["gap"] if est["gap"] else float("nan")
    return {"contrast": f"{hi}-{lo}", "gap": est["gap"], "gap_ci95": ci("gap"), "explained": est["explained"],
            "explained_ci95": ci("explained"), "share_explained": share,
            **{f"fam_{f}": est[f"fam_{f}"] for f in ("delivery", "structure", "content")},
            **{f"fam_{f}_ci95": ci(f"fam_{f}") for f in ("delivery", "structure", "content")},
            "top_contrib": dict(sorted(est["contrib"].items(), key=lambda kv: -abs(kv[1]))[:6])}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = assemble()
    xs = ATTRS_E = [f"{a}_E" for a in ATTRS] + OBJECTIVE
    for a in ATTRS:
        FAMILY[f"{a}_E"] = ja.FAMILY[a]
    res: Dict[str, Any] = {"n_complaints": int(df.case_id.nunique()), "n_guises": len(df)}

    # Q-C certainty and reliability
    rep_path = Path("jev_pilot_out/attributes_repeat/scores.csv")
    rep = pd.read_csv(rep_path).drop_duplicates("text_id") if rep_path.exists() else None
    base = pd.read_csv(OUT / "scores.csv").drop_duplicates("text_id")
    cert = {}
    for a in ATTRS:
        r = {"family": ja.FAMILY[a], "mean_norm_entropy": entropy3(df, a),
             "within_complaint_sd_E": float(demean(df, [f"{a}_E"])[f"{a}_E"].std())}
        if rep is not None:
            j = rep.merge(base, on="text_id", suffixes=("_r", "_b"))
            r["retest_spearman_E"] = float(j[f"{a}_E_r"].rank().corr(j[f"{a}_E_b"].rank()))
            r["retest_choice_agree"] = float((j[f"{a}_choice_r"] == j[f"{a}_choice_b"]).mean())
        cert[a] = r
    res["QC_certainty"] = cert

    # Q-A manipulation and fidelity
    qa = {}
    for b in BANKS:
        d = df[df.rewrite_model == b]
        rows = {}
        for x in xs:
            sdw = float(demean(d, [x])[x].std()) or float("nan")
            effects = {}
            for dim, (hi, lo) in CONTRASTS.items():
                diff = d[d[f"p_{dim}"] == hi].groupby("case_id")[x].mean() - d[d[f"p_{dim}"] == lo].groupby("case_id")[x].mean()
                effects[f"{dim}:{hi}-{lo}_std"] = float(diff.mean() / sdw) if sdw else float("nan")
            rows[x] = {"family": FAMILY[x], "eta2_persona": eta_persona(d, x), **effects}
        qa[b] = rows
    res["QA_manipulation"] = qa

    # Q-B persona inference
    qb = {}
    targets = {"jev_age_old": "age_p_old", "jev_edu_college": "edu_p_college", "jev_ses": "jev_ses",
               "gpt_age_old": "gpt_age_old", "gpt_edu_college": "gpt_edu_college", "gpt_ses": "gpt_ses",
               # amendment 2a: which attributes make Jev UNSURE who is writing
               **{f"jev_{d}_ambiguity": f"jev_{d}_entropy" for d in PERSONA_LEVELS}}
    for b in BANKS:
        d = df[df.rewrite_model == b]
        out = {}
        for name, y in targets.items():
            full = ols_within(d, y, xs)
            fam_r2 = {f: ols_within(d, y, [x for x in xs if FAMILY[x] == f])["r2_within"]
                      for f in ("delivery", "structure", "content")}
            top = dict(sorted(full["beta_std"].items(), key=lambda kv: -abs(kv[1]))[:6])
            out[name] = {"r2_within": full["r2_within"], "r2_by_family_alone": fam_r2, "top_std_betas": top}
        qb[b] = out
    res["QB_persona_inference"] = qb

    # Q-D outcomes, decomposition and relevance
    qd = {}
    for b in BANKS:
        d = df[df.rewrite_model == b]
        out = {}
        for y in ("gpt_tier", "gpt_monetary", "e_tier"):
            full = ols_within(d, y, xs)
            fam_r2 = {f: ols_within(d, y, [x for x in xs if FAMILY[x] == f])["r2_within"]
                      for f in ("delivery", "structure", "content")}
            out[y] = {"r2_within": full["r2_within"], "r2_by_family_alone": fam_r2,
                      "top_std_betas": dict(sorted(full["beta_std"].items(), key=lambda kv: -abs(kv[1]))[:8]),
                      "decomposition": {dim: gelbach(d, y, xs, dim) for dim in ("age", "edu", "ses")}}
        qd[b] = out
    res["QD_outcomes"] = qd

    # Q-E (amendment 2a): allowing for Jev's uncertainty about which persona applies
    qe = {}
    for b in BANKS:
        d = df[df.rewrite_model == b]
        out_b = {}
        for dim in ("age", "edu", "ses"):
            hi, lo = CONTRASTS[dim]
            dd = d[d[f"p_{dim}"].isin([hi, lo])].copy()
            dd["intended_hi"] = (dd[f"p_{dim}"] == hi).astype(float)
            dd["perceived_hi"] = dd[f"{dim}_p_{hi}"] / (dd[f"{dim}_p_{hi}"] + dd[f"{dim}_p_{lo}"])  # Jev soft label
            dd["recognition"] = np.where(dd.intended_hi == 1, dd[f"{dim}_p_{hi}"], dd[f"{dim}_p_{lo}"])
            dd["recognised"] = dd.recognition >= RECOGNISED
            r = {"mean_recognition": float(dd.recognition.mean()),
                 "share_recognised": float(dd.recognised.mean()),
                 "mean_recognition_by_level": {hi: float(dd.loc[dd.intended_hi == 1, "recognition"].mean()),
                                               lo: float(dd.loc[dd.intended_hi == 0, "recognition"].mean())}}
            for y in ("gpt_tier", "gpt_monetary"):
                m_int = ols_within(dd, y, ["intended_hi"])
                m_per = ols_within(dd, y, ["perceived_hi"])
                m_both = ols_within(dd, y, ["intended_hi", "perceived_hi"])
                yy = demean(dd.dropna(subset=[y]), [y])[y]
                strata = {}
                for name, mask in (("recognised", dd.recognised), ("ambiguous", ~dd.recognised)):
                    s = dd.loc[mask].assign(_y=yy)
                    vals = s.dropna(subset=["_y"])
                    gap = float(vals.loc[vals.intended_hi == 1, "_y"].mean() - vals.loc[vals.intended_hi == 0, "_y"].mean())
                    groups = list(vals.reset_index(drop=True).groupby("case_id").indices.values())
                    v2 = vals.reset_index(drop=True)
                    bs = [float("nan")]
                    for _ in range(N_BOOT if groups else 0):
                        pos = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
                        t = v2.iloc[pos]
                        bs.append(t.loc[t.intended_hi == 1, "_y"].mean() - t.loc[t.intended_hi == 0, "_y"].mean())
                    strata[name] = {"n_guises": int(len(vals)), "gap_demeaned": gap,
                                    "gap_ci95": [float(np.nanquantile(bs, .025)), float(np.nanquantile(bs, .975))]}
                r[y] = {"raw_beta_intended": m_int["beta_raw"]["intended_hi"],
                        "raw_beta_perceived": m_per["beta_raw"]["perceived_hi"],
                        "joint_raw_betas": m_both["beta_raw"], "joint_r2_within": m_both["r2_within"],
                        "gap_by_recognition": strata}
            out_b[dim] = r
        qe[b] = out_b
    res["QE_persona_uncertainty"] = qe

    # Go / no-go
    g1_ok = [a for a, r in cert.items() if r["mean_norm_entropy"] < G1_ENTROPY and r["within_complaint_sd_E"] >= G1_SD]
    g2_ok = sorted({x for b in BANKS for x, r in qa[b].items()
                    if r["family"] in ("delivery", "structure") and x.endswith("_E") and r["eta2_persona"] >= G2_ETA})
    shares = {f"{b}:{dim}": qd[b]["gpt_tier"]["decomposition"][dim]["share_explained"]
              for b in ("gpt-4o-mini",) for dim in ("edu", "ses")}
    res["go_no_go"] = {
        "G1_discriminating_attributes": g1_ok, "G1_pass": len(g1_ok) >= G1_MIN_ATTRS,
        "G2_manipulated_delivery_structure": g2_ok, "G2_pass": len(g2_ok) >= G2_MIN_ATTRS,
        "G3_shares_gpt4o_tier": shares, "G3_pass": any(s >= G3_SHARE for s in shares.values() if s == s),
        "drop_for_confirmatory_unreliable": [a for a, r in cert.items() if r.get("retest_spearman_E", 1) < RELIABILITY_DROP],
    }
    res["go_no_go"]["GO"] = bool(res["go_no_go"]["G1_pass"] and res["go_no_go"]["G2_pass"] and res["go_no_go"]["G3_pass"])
    (OUT / "pilot_results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    md = ["# Attribute pilot results", "", f"Complaints {res['n_complaints']}, guises {res['n_guises']}", "",
          "## Go / no-go", "```", json.dumps(res["go_no_go"], indent=2, default=float), "```",
          "## Certainty and reliability", "```", json.dumps(cert, indent=2, default=float), "```"]
    (OUT / "pilot_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
