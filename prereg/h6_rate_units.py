"""Post hoc: H6 (Jev vs GPT monetary disparity) in rate units, with 90%/95% CIs,
a minimum-effect test and TOST equivalence against a 1 percentage-point bound.

NOT pre-registered (added 20 Oct 2026 at the author's request, after the n = 1,000
confirmatory run).  Quantities, each for GPT and for Jev with its p2 threshold matched to
that GPT model's monetary volume, and their difference (GPT - Jev), in percentage points:
  * sys_sd_pp : systematic persona SD of the monetary rate (two-way ANOVA estimator,
                noise-corrected) - the rate-unit version of the pre-registered D numerator
  * gap_<dim> : paired within-complaint monetary-rate gap for the pre-specified contrasts
Tests (alpha 0.05, bound B = 1 pp), using complaint-bootstrap SE (normal approximation)
and percentile CIs:
  * minimum-effect test  H0: |delta| <= B  vs  H1: |delta| > B   (p = min one-sided p)
  * TOST equivalence     H0: |delta| >= B  vs  H1: |delta| < B   (p = max one-sided p)
    -> equivalent iff the 90% CI lies inside (-B, B)

CAVEAT (found on first run): percentile-bootstrap CIs for sys_sd_pp are NOT valid. Resampling
complaints with replacement duplicates them, so the noise correction (s2/C) under-corrects within
each resample and the truncated variance-component estimate is biased upward; several point
estimates fall outside their own CIs.  Use sys_sd_pp point estimates descriptively only; the paired
gap_<dim> contrasts are linear and their CIs/tests are valid.  The same bias affects bootstrap CIs
of the pre-registered D-based quantities (H3b, H4, H6) and is reported as a limitation.

    python prereg/h6_rate_units.py
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
from scipy.stats import norm
sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).parent))
import confirmatory_analysis as ca  # noqa: E402

B = 1.0
N_BOOT = 4000
rng = np.random.default_rng(20261020)
CONTRASTS = {"age": ("old", "young"), "edu": ("college", "high_school"), "ses": ("high", "low")}


def mats(P, G, b):
    g = G[b]
    gm = (g == 2).astype(float)
    gm[np.isnan(g)] = np.nan
    p2 = P[f"{b}|{ca.PRIMARY_ORDER}"][..., 2]
    jm = (p2 >= np.quantile(p2, 1 - np.nanmean(gm))).astype(float)
    keep = ~np.isnan(gm).any(1)
    return gm[keep], jm[keep]


def sys_sd_pp(X):
    C, K = X.shape
    rm, cm, gmn = X.mean(1, keepdims=True), X.mean(0, keepdims=True), X.mean()
    s2 = ((X - rm - cm + gmn) ** 2).sum() / ((C - 1) * (K - 1))
    return 100 * math.sqrt(max(float(np.var(cm, ddof=1)) - s2 / C, 0.0))


def gap_pp(X, d):
    hi, lo = CONTRASTS[d]
    lv = ca.DIM_OF[d]
    return 100 * (X[:, lv == hi].mean() - X[:, lv == lo].mean())


def stats(g, j):
    C = len(g)
    fns = {"sys_sd_pp": sys_sd_pp, **{f"gap_{d}": (lambda X, d=d: gap_pp(X, d)) for d in CONTRASTS}}
    out = {}
    idx = [rng.integers(0, C, C) for _ in range(N_BOOT)]
    for name, f in fns.items():
        est = {"gpt": f(g), "jev": f(j)}
        est["delta_gpt_minus_jev"] = est["gpt"] - est["jev"]
        bs = np.array([[f(g[ix]), f(j[ix])] for ix in idx])
        bs = np.column_stack([bs, bs[:, 0] - bs[:, 1]])
        rec = {}
        for k, col in (("gpt", 0), ("jev", 1), ("delta_gpt_minus_jev", 2)):
            v, s = est[k], float(np.std(bs[:, col], ddof=1))
            ci90 = [float(np.quantile(bs[:, col], .05)), float(np.quantile(bs[:, col], .95))]
            ci95 = [float(np.quantile(bs[:, col], .025)), float(np.quantile(bs[:, col], .975))]
            p_hi_met, p_lo_met = 1 - norm.cdf((v - B) / s), norm.cdf((v + B) / s)      # delta > B ; delta < -B
            p_tost = max(1 - norm.cdf((v + B) / s), norm.cdf((v - B) / s))             # delta > -B and delta < B
            rec[k] = {"estimate_pp": v, "se_pp": s, "ci90_pp": ci90, "ci95_pp": ci95,
                      "min_effect_p": min(p_hi_met, p_lo_met), "exceeds_1pp": bool(ci90[0] > B or ci90[1] < -B),
                      "tost_p": p_tost, "equivalent_within_1pp": bool(ci90[0] > -B and ci90[1] < B)}
        out[name] = rec
    return out


def main():
    Pe, Ge, _, _ = ca.load_panel("exploratory", "")
    Ph, Gh, _, _ = ca.load_panel("holdout", "n1000")
    res = {}
    for b in ca.BANKS:
        ge, je = mats(Pe, Ge, b)
        gh, jh = mats(Ph, Gh, b)
        res[b] = {"holdout_n1000": stats(gh, jh), "pooled_n3000": stats(np.vstack([ge, gh]), np.vstack([je, jh]))}
    Path("prereg/h6_rate_units_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    for b, r in res.items():
        for panel, s in r.items():
            print(f"\n### {b} | {panel}")
            for name, rec in s.items():
                for k in ("gpt", "jev", "delta_gpt_minus_jev"):
                    x = rec[k]
                    print(f"  {name:10s} {k:20s} {x['estimate_pp']:+.3f} pp  90% [{x['ci90_pp'][0]:+.3f},{x['ci90_pp'][1]:+.3f}]  95% [{x['ci95_pp'][0]:+.3f},{x['ci95_pp'][1]:+.3f}]  MET p={x['min_effect_p']:.3g} (>1pp: {x['exceeds_1pp']})  TOST p={x['tost_p']:.3g} (within 1pp: {x['equivalent_within_1pp']})")


if __name__ == "__main__":
    main()
