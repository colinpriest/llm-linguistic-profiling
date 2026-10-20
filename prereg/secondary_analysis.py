"""Secondary (pre-registered as reported, not confirmatory; prereg/jev_facct_prereg.md section 5) and
sensitivity analyses on the holdout (n = 1,000), with the exploratory panel (n = 2,000) for reference.

    python prereg/secondary_analysis.py

S0  H4 on a difference scale: D(probability-average) - D(hard vote), with the corrected bootstrap
S1  Hidden variation: complaints with a constant tier across the 36 guises, the range of p2 and entropy within them,
    TV decomposition (option order, paraphrase/same persona, different persona)
S2  Directional persona effects on p2 and normalised entropy (within-complaint contrasts)
S3  Option order x persona interaction: the persona contrast in E[tier] under each of the 6 option orders, and a
    Wald test of equality across orders (bootstrap covariance)
S4  Persona recovery (manipulation check): Jev argmax = target and mean P(target), by bank and dimension
Output: prereg/secondary_results.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).parent))
import confirmatory_analysis as ca  # noqa: E402
import corrected_bootstrap as cb  # noqa: E402

N_BOOT = 1000
rng = np.random.default_rng(20261022)
DIRS = {"age": ("old", "young"), "edu": ("college", "high_school"), "ses": ("high", "low")}


def ci(v, a=0.05):
    v = np.asarray(v, float)
    return [float(np.nanquantile(v, a / 2)), float(np.nanquantile(v, 1 - a / 2))]


def boot_idx(C):
    return [rng.integers(0, C, C) for _ in range(N_BOOT)]


def s0_h4_difference(P, C):
    out = {}
    idx = boot_idx(C)
    A, B = P[f"{ca.BANKS[0]}|{ca.PRIMARY_ORDER}"], P[f"{ca.BANKS[1]}|{ca.PRIMARY_ORDER}"]
    comps = {"paraphrase_2": (ca.norm(A + B).argmax(-1).astype(float), ca.hard_vote([A.argmax(-1), B.argmax(-1)]))}
    for b in ca.BANKS:
        ps = [P[f"{b}|{o}"] for o in ca.ALL_ORDERS]
        comps[f"orders_6_{b}"] = (ca.norm(sum(ps)).argmax(-1).astype(float), ca.hard_vote([x.argmax(-1) for x in ps]))
    for name, (xa, xv) in comps.items():
        est = cb.D_point(xa) - cb.D_point(xv)
        vals = [cb.D_resample(xa, ix, True) - cb.D_resample(xv, ix, True) for ix in idx]
        out[name] = {"D_prob_avg": cb.D_point(xa), "D_hard_vote": cb.D_point(xv), "difference": est,
                     "ci90_corrected": ci(vals, 0.10), "ci95_corrected": ci(vals),
                     "prob_avg_superior": bool(np.nanquantile(vals, 0.95) < 0)}
    return out


def s1_hidden(P):
    out = {}
    for b in ca.BANKS:
        F, R = P[f"{b}|o012"], P[f"{b}|o210"]
        am = F.argmax(-1)
        const = (am == am[:, :1]).all(1)
        H = ca.entropy(F)
        p2r = F[..., 2].max(1) - F[..., 2].min(1)
        iu = np.triu_indices(36, 1)
        pair_tv = ca.tv(F[:, iu[0]], F[:, iu[1]])
        out[b] = {"share_complaints_tier_constant": float(const.mean()),
                  "mean_p2_range_in_constant": float(p2r[const].mean()),
                  "share_constant_with_p2_range_gt_0.10": float((p2r[const] > 0.10).mean()),
                  "mean_entropy_range_in_constant": float((H.max(1) - H.min(1))[const].mean()),
                  "mean_pairwise_tv_between_guises": float(pair_tv.mean()),
                  "tv_same_text_option_reversal": float(ca.tv(F, R).mean())}
    A, B = P[f"{ca.BANKS[0]}|o012"], P[f"{ca.BANKS[1]}|o012"]
    same = ca.tv(A, B).mean()
    diff = np.mean([ca.tv(A[:, i][:, None, :], np.delete(B, i, axis=1)).mean() for i in range(36)])
    out["cross_bank"] = {"tv_same_persona": float(same), "tv_different_persona": float(diff),
                         "persona_specific_share": float((diff - same) / same)}
    return out


def contrast_matrix(X, dim):
    hi, lo = DIRS[dim]
    lv = ca.DIM_OF[dim]
    return X[:, lv == hi].mean(1) - X[:, lv == lo].mean(1)            # per-complaint contrast


def s2_p2_entropy(P, C):
    out = {}
    idx = boot_idx(C)
    for b in ca.BANKS:
        F = P[f"{b}|o012"]
        for name, X in (("p2", F[..., 2]), ("entropy", ca.entropy(F)), ("max_prob", F.max(-1))):
            for dim in DIRS:
                c = contrast_matrix(X, dim)
                out[f"{b}:{name}:{dim}"] = {"estimate": float(c.mean()), "ci95": ci([c[ix].mean() for ix in idx])}
    return out


def s3_order_interaction(P, C):
    out = {}
    idx = boot_idx(C)
    L = np.eye(6)[1:] - np.eye(6)[:1]                                  # differences from o012
    for b in ca.BANKS:
        for dim in DIRS:
            cm = np.stack([contrast_matrix(ca.etier(P[f"{b}|{o}"]), dim) for o in ca.ALL_ORDERS], 1)   # C x 6
            est = cm.mean(0)
            bs = np.array([cm[ix].mean(0) for ix in idx])
            S = np.cov(bs, rowvar=False)
            d = L @ est
            W = float(d @ np.linalg.pinv(L @ S @ L.T) @ d)
            out[f"{b}:{dim}"] = {"contrast_by_order": dict(zip(ca.ALL_ORDERS, map(float, est))),
                                 "ci95_by_order": {o: ci(bs[:, i]) for i, o in enumerate(ca.ALL_ORDERS)},
                                 "range": float(est.max() - est.min()),
                                 "wald_equal_across_orders": W, "df": 5, "p": float(chi2.sf(W, 5))}
    return out


def s4_recovery(panel_scores: pd.DataFrame):
    g = panel_scores[panel_scores.source == "guise"]
    out = {}
    for b in ca.BANKS:
        gb = g[g.rewrite_model == b]
        parts = gb.persona_label.str.extract(r"^(young|old)_(urban|suburban|rural)_(high_school|college)_(low|middle|high)$")
        r = {}
        for i, (dim, levels) in enumerate((("age", 2), ("geo", 3), ("edu", 2), ("ses", 3))):
            target = parts[i].to_numpy()
            p_t = np.array([gb[f"{dim}_p_{t}"].iloc[j] for j, t in enumerate(target)])
            r[dim] = {"chance": 1 / levels, "argmax_matches_target": float((gb[f"{dim}_choice"].to_numpy() == target).mean()),
                      "mean_p_target": float(p_t.mean())}
        out[b] = r
    return out


def run(panel):
    P, G, cases, base = ca.load_panel(panel, "n1000")
    C = len(cases)
    scores = pd.read_csv("holdout_out/jev/scores.csv" if panel == "holdout" else "jev_pilot_out/jev_scores.csv")
    res = {"n_complaints": C, "orders": list(P["_orders"]), "S1_hidden_variation": s1_hidden(P),
           "S2_p2_entropy_effects": s2_p2_entropy(P, C), "S4_persona_recovery": s4_recovery(scores)}
    if len(P["_orders"]) == 6:
        res["S0_H4_difference_scale"] = s0_h4_difference(P, C)
        res["S3_order_x_persona"] = s3_order_interaction(P, C)
    return res


def main():
    res = {"holdout_n1000": run("holdout"), "exploratory_n2000": run("exploratory")}
    Path("prereg/secondary_results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    h = res["holdout_n1000"]
    print(json.dumps({"S0": h.get("S0_H4_difference_scale"), "S1": h["S1_hidden_variation"],
                      "S3_p": {k: round(v["p"], 4) for k, v in h.get("S3_order_x_persona", {}).items()},
                      "S4": h["S4_persona_recovery"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
