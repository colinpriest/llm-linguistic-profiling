"""Sensitivity analysis: corrected complaint bootstrap for the noise-corrected disparity metric D.

Post hoc (20 Oct 2026). The pre-registered analysis bootstrapped D by resampling complaints
with replacement and re-applying the noise correction var_b = var(persona means) - s2/C.
In a resample where complaint i appears w_i times, the noise variance of a persona mean is
s2 * sum(w_i^2) / C^2, not s2 / C, so the old correction under-corrects and every bootstrap
D is biased upward (point estimates can fall outside their own CIs). Here each resample
subtracts the multiplicity-correct term. Point estimates are unchanged; only the CIs change.

Recomputes the D-based quantities H3b, H4 and H6, plus the H6 rate-unit systematic SD, on
the holdout (n = 1,000) and pooled (n = 3,000) panels, and reports both the original
(uncorrected) and the corrected percentile CIs.

    python prereg/corrected_bootstrap.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).parent))
import confirmatory_analysis as ca  # noqa: E402

N_BOOT = 2000
SEED = 20261020


def var_components(X: np.ndarray, w: np.ndarray | None = None):
    """(var_persona, var_complaint) with noise correction; w = complaint multiplicities in a resample."""
    X = X[~np.isnan(X).any(axis=1)] if w is None else X
    C, K = X.shape
    rm, cm, gm = X.mean(1, keepdims=True), X.mean(0, keepdims=True), X.mean()
    s2 = float(((X - rm - cm + gm) ** 2).sum() / ((C - 1) * (K - 1)))
    noise_cm = s2 / C if w is None else s2 * float((w ** 2).sum()) / C ** 2
    var_b = float(np.var(cm, ddof=1)) - noise_cm
    var_a = float(np.var(rm, ddof=1)) - s2 / K
    return var_b, var_a


def D_point(X):
    vb, va = var_components(X)
    return math.sqrt(max(vb, 0) / va) if va > 0 else float("nan")


def D_resample(X, ix, corrected: bool):
    Xb = X[ix]
    if not corrected:
        return ca.D(Xb)
    w = np.bincount(ix, minlength=len(X))
    w = w[w > 0]
    vb, va = var_components(Xb, w)
    return math.sqrt(max(vb, 0) / va) if va > 0 else float("nan")


def sd_pp_resample(X, ix, corrected: bool):
    Xb = X[ix]
    if corrected:
        w = np.bincount(ix, minlength=len(X))
        vb, _ = var_components(Xb, w[w > 0])
    else:
        vb, _ = var_components(Xb)
    return 100 * math.sqrt(max(vb, 0))


def boot(stat, X_list, C, rng, level=0.95):
    out = {}
    for corrected in (False, True):
        vals = []
        for _ in range(N_BOOT):
            ix = rng.integers(0, C, C)
            vals.append(stat(X_list, ix, corrected))
        vals = np.array(vals, float)
        a = (1 - level) / 2
        out["corrected" if corrected else "original"] = [float(np.nanquantile(vals, a)), float(np.nanquantile(vals, 1 - a))]
    return out


def complete(*Xs):
    keep = np.ones(len(Xs[0]), bool)
    for X in Xs:
        keep &= ~np.isnan(X).any(axis=1)
    return [X[keep] for X in Xs]


def monetary(P, G, b):
    g = G[b]
    gm = (g == 2).astype(float)
    gm[np.isnan(g)] = np.nan
    p2 = P[f"{b}|{ca.PRIMARY_ORDER}"][..., 2]
    jm = (p2 >= np.quantile(p2, 1 - np.nanmean(gm))).astype(float)
    return complete(gm, jm)


def main():
    rng = np.random.default_rng(SEED)
    Pe, Ge, _, _ = ca.load_panel("exploratory", "")
    Ph, Gh, _, _ = ca.load_panel("holdout", "n1000")
    res = {}
    for panel in ("holdout_n1000", "pooled_n3000"):
        r = {}
        for b in ca.BANKS:
            def stack(key):
                return Ph[key] if panel == "holdout_n1000" else np.concatenate([Pe[key], Ph[key]])
            p = stack(f"{b}|{ca.PRIMARY_ORDER}")
            am, et = p.argmax(-1).astype(float), ca.etier(p)
            C = len(am)
            # H3b: D(argmax) - D(E[tier])
            h3b = boot(lambda L, ix, c: D_resample(L[0], ix, c) - D_resample(L[1], ix, c), [am, et], C, rng)
            h3b["estimate"] = D_point(am) - D_point(et)
            # H4 (orders, per bank): D(prob-avg argmax over 6 orders) / D(hard vote)
            ps = [stack(f"{b}|{o}") for o in ca.ALL_ORDERS]
            xa = ca.norm(sum(ps)).argmax(-1).astype(float)
            xv = ca.hard_vote([x.argmax(-1) for x in ps])
            h4 = boot(lambda L, ix, c: D_resample(L[0], ix, c) / max(D_resample(L[1], ix, c), 1e-12), [xa, xv], C, rng, level=0.90)
            h4["estimate"] = D_point(xa) / D_point(xv)
            # H6: D(GPT monetary) - D(Jev matched), and rate-unit systematic SDs
            if panel == "holdout_n1000":
                gm, jm = monetary(Ph, Gh, b)
            else:
                ge, je = monetary(Pe, Ge, b)
                gh, jh = monetary(Ph, Gh, b)
                gm, jm = np.vstack([ge, gh]), np.vstack([je, jh])
            Cm = len(gm)
            h6 = boot(lambda L, ix, c: D_resample(L[0], ix, c) - D_resample(L[1], ix, c), [gm, jm], Cm, rng)
            h6["estimate"] = D_point(gm) - D_point(jm)
            sd = {}
            for name, X in (("gpt", gm), ("jev", jm)):
                sd[name] = boot(lambda L, ix, c: sd_pp_resample(L[0], ix, c), [X], Cm, rng)
                sd[name]["estimate_pp"] = 100 * math.sqrt(max(var_components(X)[0], 0))
            sd["delta"] = boot(lambda L, ix, c: sd_pp_resample(L[0], ix, c) - sd_pp_resample(L[1], ix, c), [gm, jm], Cm, rng)
            sd["delta"]["estimate_pp"] = sd["gpt"]["estimate_pp"] - sd["jev"]["estimate_pp"]
            r[b] = {"H3b_D_argmax_minus_D_etier": h3b, "H4_orders6_ratio_ci90": h4, "H6_D_gpt_minus_D_jev": h6,
                    "H6_sys_sd_pp": sd}
        res[panel] = r
    Path("prereg/corrected_bootstrap_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    for panel, r in res.items():
        print(f"\n### {panel}")
        for b, q in r.items():
            for k in ("H3b_D_argmax_minus_D_etier", "H4_orders6_ratio_ci90", "H6_D_gpt_minus_D_jev"):
                v = q[k]
                print(f"  {b:12s} {k:28s} est {v['estimate']:+.4f}  original CI [{v['original'][0]:+.4f},{v['original'][1]:+.4f}]  corrected CI [{v['corrected'][0]:+.4f},{v['corrected'][1]:+.4f}]")
            for k in ("gpt", "jev", "delta"):
                v = q["H6_sys_sd_pp"][k]
                print(f"  {b:12s} sys_sd_pp {k:5s}               est {v['estimate_pp']:+.3f}pp  original CI [{v['original'][0]:+.3f},{v['original'][1]:+.3f}]  corrected CI [{v['corrected'][0]:+.3f},{v['corrected'][1]:+.3f}]")


if __name__ == "__main__":
    main()
