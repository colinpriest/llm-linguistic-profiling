"""Frozen confirmatory test of the attribute hypotheses H-A1..H-A5 (prereg/attributes_confirmatory.md).

    python prereg/attributes_confirmatory.py --panel holdout   # the one confirmatory run (1,000 complaints)
    python prereg/attributes_confirmatory.py --panel pilot     # code check on the already-seen pilot

Bootstrap: complaints are resampled as blocks. Within-complaint demeaning is computed once per
analysis subset, because demeaned rows are invariant to which complaints are drawn.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).parent))
import jev_attributes as ja  # noqa: E402
from attributes_pilot_analysis import objective_metrics, persona_parts  # noqa: E402

SEED = 20261021
N_BOOT = 2000
BANKS = ["gpt-4o-mini", "gpt-5-mini"]
ATTRS = [f"{a}_E" for a in ja.ATTRIBUTES]
OBJECTIVE = ["obj_sentence_len", "obj_fk_grade", "obj_emotion_abs"]
XS = ATTRS + OBJECTIVE
FAMILY = {**{f"{a}_E": f for a, f in ja.FAMILY.items()}, **{k: "delivery" for k in OBJECTIVE}}
CONTRASTS = {"age": ("old", "young"), "edu": ("college", "high_school"), "ses": ("high", "low")}
RECOGNISED = 0.60
# Pre-registered thresholds
HA1_ETA_BOUND = 0.05
HA2_MIN_SHARE = 0.50
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble(panel: str) -> pd.DataFrame:
    if panel == "holdout":
        a = pd.read_csv("holdout_out/attributes/scores.csv")
        rw = pd.read_csv("holdout_out/rewrites_n1000.csv")[["case_id", "rewrite_model", "persona_label", "text"]]
        js = pd.read_csv("holdout_out/jev/scores.csv")
        js["e_tier"] = js.p1_o012 + 2 * js.p2_o012
        dec = pd.read_csv("holdout_out/decisions_n1000.csv")[["case_id", "model", "variant", "tier"]]
    else:
        a = pd.read_csv("jev_pilot_out/attributes_pilot/scores.csv")
        t = pd.read_parquet("jev_pilot_out/texts_exploratory.parquet")
        rw = t[t.source == "guise"][["case_id", "rewrite_model", "persona_label", "text"]]
        js = pd.read_csv("jev_pilot_out/jev_scores.csv")
        dec = pd.read_csv("output/results_persona_decisions_20260220_101746.csv", usecols=["case_id", "model", "variant", "tier"])
    key = ["case_id", "rewrite_model", "persona_label"]
    a = a[a.source == "guise"].drop_duplicates(key)
    a = a.merge(rw.drop_duplicates(key), on=key, how="left")
    a = pd.concat([a.reset_index(drop=True), objective_metrics(a.text.reset_index(drop=True))], axis=1)
    pcols = ["age_p_young", "age_p_old", "edu_p_high_school", "edu_p_college", "ses_p_low", "ses_p_middle", "ses_p_high"]
    js = js[js.source == "guise"][["text_id", "e_tier"] + pcols].drop_duplicates("text_id")
    js["jev_ses"] = js.ses_p_high - js.ses_p_low
    a = a.merge(js, on="text_id", how="left")
    dec = dec.rename(columns={"model": "rewrite_model", "variant": "persona_label", "tier": "gpt_tier"})
    a = a.merge(dec.drop_duplicates(key), on=key, how="left")
    a["gpt_monetary"] = (a.gpt_tier == 2).astype(float).where(a.gpt_tier.notna())
    parts = a.persona_label.map(persona_parts)
    for d in ("age", "geo", "edu", "ses"):
        a[f"p_{d}"] = parts.map(lambda x: x[d])
    return a


# ---------------------------------------------------------------------------
# Block-bootstrap machinery on within-complaint-demeaned arrays
# ---------------------------------------------------------------------------

class Blocks:
    """Demeaned Y and X for a subset, plus the row positions of each complaint."""

    def __init__(self, d: pd.DataFrame, y: str, xs: List[str], extra: Dict[str, np.ndarray] | None = None):
        d = d.dropna(subset=[y] + xs).reset_index(drop=True)
        g = d.groupby("case_id")
        self.Y = (d[y] - g[y].transform("mean")).to_numpy()
        self.X = (d[xs] - g[xs].transform("mean")).to_numpy()
        self.raw = d
        self.groups = list(g.indices.values())
        self.xs = xs

    def sample(self):
        pick = rng.integers(0, len(self.groups), len(self.groups))
        return np.concatenate([self.groups[i] for i in pick])


def r2(Y, X):
    sd = X.std(0)
    keep = sd > 0
    if not keep.any() or (Y ** 2).sum() == 0:
        return float("nan")
    Xz = X[:, keep] / sd[keep]
    beta, *_ = np.linalg.lstsq(Xz, Y, rcond=None)
    return float(1 - ((Y - Xz @ beta) ** 2).sum() / (Y ** 2).sum())


def raw_betas(Y, X):
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return beta


def ci(vals, lo=0.025, hi=0.975):
    v = np.asarray(vals, float)
    return [float(np.nanquantile(v, lo)), float(np.nanquantile(v, hi))]


# ---------------------------------------------------------------------------
# Hypotheses
# ---------------------------------------------------------------------------

def eta2(X: np.ndarray) -> float:
    rm, cm, gm = X.mean(1, keepdims=True), X.mean(0, keepdims=True), X.mean()
    ss_p = X.shape[0] * ((cm - gm) ** 2).sum()
    ss_r = ((X - rm - cm + gm) ** 2).sum()
    return float(ss_p / (ss_p + ss_r))


def h_a1(df):
    """Content invariance: persona eta^2 for every content attribute, both banks, upper 95% bound < 0.05."""
    out, ok = {}, True
    for b in BANKS:
        d = df[df.rewrite_model == b]
        for a in [x for x in ATTRS if FAMILY[x] == "content"]:
            W = d.pivot_table(index="case_id", columns="persona_label", values=a).dropna().to_numpy()
            est = eta2(W)
            bs = [eta2(W[rng.integers(0, len(W), len(W))]) for _ in range(N_BOOT)]
            upper = float(np.quantile(bs, 0.95))
            out[f"{b}:{a}"] = {"eta2": est, "upper95_one_sided": upper, "below_bound": upper < HA1_ETA_BOUND}
            ok &= upper < HA1_ETA_BOUND
    out["supported"] = bool(ok)
    return out


def gelbach_blocks(d: pd.DataFrame, y: str, dim: str):
    hi, lo = CONTRASTS[dim]
    d = d[d[f"p_{dim}"].isin([hi, lo])].copy()
    d["_hi"] = (d[f"p_{dim}"] == hi).astype(float)
    blk = Blocks(d, y, XS + ["_hi"])
    Yraw = blk.raw[y].to_numpy()
    Xraw = blk.raw[XS].to_numpy()
    H = blk.raw["_hi"].to_numpy() == 1
    Xd = blk.X[:, :-1]                                        # demeaned attributes (exclude _hi)
    fam_idx = {f: np.array([FAMILY[x] == f for x in XS]) for f in ("delivery", "structure", "content")}

    def stats(pos):
        Yd, Xdd, yr, xr, h = blk.Y[pos], Xd[pos], Yraw[pos], Xraw[pos], H[pos]
        beta = raw_betas(Yd, Xdd)
        gap = yr[h].mean() - yr[~h].mean()
        contrib = (xr[h].mean(0) - xr[~h].mean(0)) * beta
        fam = {f: float(contrib[m].sum()) for f, m in fam_idx.items()}
        return gap, float(contrib.sum()), fam
    allpos = np.arange(len(Yraw))
    gap, expl, fam = stats(allpos)
    bs = [stats(blk.sample()) for _ in range(N_BOOT)]
    return {"gap": gap, "gap_ci95": ci([b[0] for b in bs]), "explained": expl, "explained_ci95": ci([b[1] for b in bs]),
            "share_explained": expl / gap if gap else float("nan"),
            "share_ci95": ci([b[1] / b[0] if b[0] else np.nan for b in bs]),
            **{f"fam_{f}": v for f, v in fam.items()},
            **{f"fam_{f}_ci95": ci([b[2][f] for b in bs]) for f in fam},
            "delivery_minus_content_ci95": ci([b[2]["delivery"] - b[2]["content"] for b in bs])}


def h_a2(df):
    """GPT-4o-mini tier: edu and SES gaps >= 50% explained; delivery contribution > 0 and > content."""
    d = df[df.rewrite_model == "gpt-4o-mini"]
    out, ok = {}, True
    for dim in ("edu", "ses"):
        g = gelbach_blocks(d, "gpt_tier", dim)
        g["supported"] = bool(g["share_explained"] >= HA2_MIN_SHARE and g["fam_delivery_ci95"][0] > 0
                              and g["delivery_minus_content_ci95"][0] > 0)
        out[dim] = g
        ok &= g["supported"]
    out["supported"] = bool(ok)
    out["secondary_monetary"] = {dim: gelbach_blocks(d, "gpt_monetary", dim) for dim in ("edu", "ses")}
    return out


def h_a3(df):
    """GPT-4o-mini tier, education: gap in recognised guises > gap in ambiguous guises."""
    d = df[(df.rewrite_model == "gpt-4o-mini") & df.p_edu.isin(["college", "high_school"])].dropna(subset=["gpt_tier"]).copy()
    d["hi"] = (d.p_edu == "college")
    d["recog"] = np.where(d.hi, d.edu_p_college, d.edu_p_high_school) >= RECOGNISED
    d["yd"] = d.gpt_tier - d.groupby("case_id").gpt_tier.transform("mean")
    d = d.reset_index(drop=True)
    groups = list(d.groupby("case_id").indices.values())
    yd, hi, rc = d.yd.to_numpy(), d.hi.to_numpy(), d.recog.to_numpy()

    def gaps(pos):
        y, h, r = yd[pos], hi[pos], rc[pos]
        gr = y[h & r].mean() - y[~h & r].mean()
        ga = y[h & ~r].mean() - y[~h & ~r].mean()
        return gr, ga, gr - ga
    est = gaps(np.arange(len(d)))
    bs = [gaps(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])) for _ in range(N_BOOT)]
    diff_ci = ci([b[2] for b in bs])
    return {"recognised_gap": est[0], "recognised_ci95": ci([b[0] for b in bs]),
            "ambiguous_gap": est[1], "ambiguous_ci95": ci([b[1] for b in bs]),
            "difference": est[2], "difference_ci95": diff_ci, "share_recognised": float(rc.mean()),
            "supported": bool(diff_ci[0] > 0)}


def family_r2_diff(d, y, fam_a, fam_b):
    blk = Blocks(d, y, XS)
    ia = np.array([FAMILY[x] == fam_a for x in XS])
    ib = np.array([FAMILY[x] == fam_b for x in XS])
    def diff(pos):
        return r2(blk.Y[pos], blk.X[pos][:, ia]), r2(blk.Y[pos], blk.X[pos][:, ib])
    ra, rb = diff(np.arange(len(blk.Y)))
    bs = [diff(blk.sample()) for _ in range(N_BOOT)]
    dci = ci([x - z for x, z in bs])
    return {f"r2_{fam_a}": ra, f"r2_{fam_b}": rb, "difference": ra - rb, "difference_ci95": dci, "supported": bool(dci[0] > 0)}


def h_a4(df):
    """Jev persona inference is delivery-dominated: R2(delivery) > R2(content), 3 targets x 2 banks."""
    out, ok = {}, True
    for b in BANKS:
        d = df[df.rewrite_model == b]
        for t in ("age_p_old", "edu_p_college", "jev_ses"):
            r = family_r2_diff(d, t, "delivery", "content")
            out[f"{b}:{t}"] = r
            ok &= r["supported"]
    out["supported"] = bool(ok)
    return out


def h_a5(df):
    """Jev's own remedy judgment is content-dominated: R2(content) > R2(delivery) for E[tier], both banks."""
    out, ok = {}, True
    for b in BANKS:
        r = family_r2_diff(df[df.rewrite_model == b], "e_tier", "content", "delivery")
        out[b] = r
        ok &= r["supported"]
    out["supported"] = bool(ok)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=["holdout", "pilot"], required=True)
    args = ap.parse_args()
    df = assemble(args.panel)
    res = {"panel": args.panel, "n_complaints": int(df.case_id.nunique()), "n_guises": len(df),
           "missing_gpt_tier": int(df.gpt_tier.isna().sum()), "missing_attr": int(df[ATTRS].isna().any(axis=1).sum()),
           "H-A1": h_a1(df), "H-A2": h_a2(df), "H-A3": h_a3(df), "H-A4": h_a4(df), "H-A5": h_a5(df)}
    out = Path("holdout_out/attributes" if args.panel == "holdout" else "jev_pilot_out/attributes_pilot")
    out.mkdir(parents=True, exist_ok=True)
    (out / "confirmatory_results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: (v["supported"] if isinstance(v, dict) and "supported" in v else v) for k, v in res.items()}, indent=2))


if __name__ == "__main__":
    main()
