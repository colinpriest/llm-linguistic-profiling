"""Descriptive detail for the attribute findings (holdout, n = 1,000; not confirmatory).

    python prereg/attributes_descriptives.py

D1  Per-attribute Gelbach contributions to GPT-4o-mini's tier gaps (age, education, SES), with block-bootstrap CIs
D2  Per-attribute persona effects (standardised within-complaint contrasts) and persona eta^2, both banks
D3  Jev certainty per attribute on the holdout (mean normalised entropy)
D4  GPT's self-reported linguistic markers (cached Step 4 persona estimates; exploratory panel only), coded by
    keyword into attribute themes, by intended persona level
Output: prereg/attributes_descriptives.json
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).parent))
import attributes_confirmatory as ac  # noqa: E402
import jev_attributes as ja  # noqa: E402

N_BOOT = 1000
ac.rng = np.random.default_rng(20261023)
THEMES = {
    "formality/register": r"formal|informal|colloquial|casual|register|slang|conversational",
    "vocabulary/technical terms": r"vocabular|technical|jargon|terminolog|sophisticat|lexic|word choice",
    "legal/regulatory references": r"legal|\blaws?\b|regulat|statut|fcra|fdcpa|\breg(ulation)?\.? ?[a-z]\b|cfpb|\bact\b|\brights?\b",
    "grammar/spelling/punctuation": r"grammar|grammat|spelling|typo|punctuat|capitali|\bmistakes?\b",
    "structure/organisation": r"structur|organi[sz]|chronolog|coheren|logical|clear|clarity|concise",
    "confidence/assertiveness": r"assertive|confiden|hedg|direct|firm|demand",
    "emotion/frustration": r"emotion|frustrat|anger|angry|upset|distress|exasperat|exclamat",
    "politeness": r"polite|courte|respectful|deferen",
    "specific details/evidence": r"specific|detail|dates?\b|amounts?|numbers|reference|document|evidence",
    "age cues": r"retire|\bage\b|\baged\b|elder|senior|\byoung|student|generation",
    "regional/dialect cues": r"rural|urban|region|dialect|southern|country",
}


def d1_contributions(df):
    d0 = df[df.rewrite_model == "gpt-4o-mini"]
    out = {}
    for dim in ("age", "edu", "ses"):
        hi, lo = ac.CONTRASTS[dim]
        d = d0[d0[f"p_{dim}"].isin([hi, lo])].copy()
        d["_hi"] = (d[f"p_{dim}"] == hi).astype(float)
        blk = ac.Blocks(d, "gpt_tier", ac.XS + ["_hi"])
        Xd = blk.X[:, :-1]
        Xraw = blk.raw[ac.XS].to_numpy()
        H = blk.raw["_hi"].to_numpy() == 1

        def contrib(pos):
            beta = ac.raw_betas(blk.Y[pos], Xd[pos])
            x = Xraw[pos]
            h = H[pos]
            return (x[h].mean(0) - x[~h].mean(0)) * beta
        est = contrib(np.arange(len(H)))
        bs = np.array([contrib(blk.sample()) for _ in range(N_BOOT)])
        rows = {a: {"family": ac.FAMILY[a], "contribution": float(est[i]),
                    "ci95": [float(np.quantile(bs[:, i], .025)), float(np.quantile(bs[:, i], .975))]}
                for i, a in enumerate(ac.XS)}
        out[dim] = dict(sorted(rows.items(), key=lambda kv: -abs(kv[1]["contribution"])))
    return out


def d2_effects(df):
    out = {}
    for b in ac.BANKS:
        d = df[df.rewrite_model == b]
        rows = {}
        for x in ac.XS:
            dm = d[x] - d.groupby("case_id")[x].transform("mean")
            sdw = float(dm.std())
            r = {"family": ac.FAMILY[x]}
            W = d.pivot_table(index="case_id", columns="persona_label", values=x).dropna().to_numpy()
            r["eta2_persona"] = ac.eta2(W)
            for dim, (hi, lo) in {**ac.CONTRASTS, "geo": ("urban", "rural")}.items():
                diff = d[d[f"p_{dim}"] == hi].groupby("case_id")[x].mean() - d[d[f"p_{dim}"] == lo].groupby("case_id")[x].mean()
                r[f"{dim}:{hi}-{lo}_std"] = float(diff.mean() / sdw) if sdw else float("nan")
            rows[x] = r
        out[b] = rows
    return out


def d3_certainty():
    s = pd.read_csv("holdout_out/attributes/scores.csv")
    s = s[s.source == "guise"]
    out = {}
    for a in ja.ATTRIBUTES:
        p = s[[f"{a}_p_{lv}" for lv in ja.LEVELS]].to_numpy().clip(1e-12)
        out[a] = {"family": ja.FAMILY[a], "mean_norm_entropy": float(np.mean(-(p * np.log(p)).sum(1) / math.log(3))),
                  "share_high_confidence_choice_p_ge_0.8": float((p.max(1) >= 0.8).mean())}
    return dict(sorted(out.items(), key=lambda kv: kv[1]["mean_norm_entropy"]))


def d4_gpt_markers():
    from src.cache import ResponseCache
    t = pd.read_parquet("jev_pilot_out/texts_exploratory.parquet")
    t = t[t.source == "guise"]
    t = t.sample(min(len(t), 20000), random_state=7)
    cache = ResponseCache("./cache")
    rows = []
    for tx, m, pl in zip(t.text, t.rewrite_model, t.persona_label):
        r = cache.get(m, "persona_estimate", tx) or {}
        marks = " | ".join(r.get("linguistic_markers") or []).lower()
        rows.append({"model": m, "persona_label": pl, "markers": marks})
    d = pd.DataFrame(rows)
    for th, pat in THEMES.items():
        d[th] = d.markers.str.contains(pat, regex=True)
    parts = d.persona_label.str.extract(r"^(young|old)_(urban|suburban|rural)_(high_school|college)_(low|middle|high)$")
    d["age"], d["geo"], d["edu"], d["ses"] = parts[0], parts[1], parts[2], parts[3]
    out = {"n_guises": len(d), "share_with_any_marker": float((d.markers != "").mean())}
    for m in ac.BANKS:
        dm = d[d.model == m]
        out[m] = {"overall_theme_share": {th: float(dm[th].mean()) for th in THEMES},
                  "by_edu": {lv: {th: float(g[th].mean()) for th in THEMES} for lv, g in dm.groupby("edu")},
                  "by_ses": {lv: {th: float(g[th].mean()) for th in THEMES} for lv, g in dm.groupby("ses")},
                  "by_age": {lv: {th: float(g[th].mean()) for th in THEMES} for lv, g in dm.groupby("age")}}
    return out


def main():
    df = ac.assemble("holdout")
    res = {"D1_gelbach_by_attribute_gpt4o_tier": d1_contributions(df), "D2_persona_effects": d2_effects(df),
           "D3_certainty_holdout": d3_certainty(), "D4_gpt_linguistic_markers": d4_gpt_markers()}
    Path("prereg/attributes_descriptives.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    for dim, rows in res["D1_gelbach_by_attribute_gpt4o_tier"].items():
        print(f"\n{dim}: top contributions")
        for a, r in list(rows.items())[:7]:
            print(f"  {a:20s} {r['family']:9s} {r['contribution']:+.4f} [{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}]")
    print("\nD3 certainty:", {a: round(v["mean_norm_entropy"], 2) for a, v in res["D3_certainty_holdout"].items()})
    for m in ac.BANKS:
        print(f"\nD4 {m} markers by edu:", json.dumps({lv: {k: round(v, 2) for k, v in th.items()} for lv, th in res["D4_gpt_linguistic_markers"][m]["by_edu"].items()}))


if __name__ == "__main__":
    main()
