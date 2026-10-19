"""Frozen confirmatory analysis for the Jev holdout panel (pre-registered).

Implements hypotheses H1-H6 in prereg/jev_facct_prereg.md exactly.  It must not
be edited after the pre-registration commit; any later analysis goes in a
separate, clearly exploratory script.

    python prereg/confirmatory_analysis.py --panel holdout       # the one confirmatory run
    python prereg/confirmatory_analysis.py --panel exploratory   # dry run on the already-seen panel

Inputs
  holdout:     holdout_out/jev/scores.csv  (+ holdout_out/decisions_<tag>.csv for H6)
  exploratory: jev_pilot_out/jev_scores.csv (+ jev_pilot_out/orders/scores.csv if present)
               and the frozen GPT decisions CSV
Output: <panel dir>/confirmatory_results.json and .md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Frozen parameters
# ---------------------------------------------------------------------------
SEED = 20261019
N_BOOT = 2000
N_PERM = 2000
ALPHA = 0.05
BANKS = ["gpt-4o-mini", "gpt-5-mini"]
PRIMARY_ORDER = "o012"                     # options listed no_action, non_monetary, monetary
ALL_ORDERS = ["o012", "o021", "o102", "o120", "o201", "o210"]
H1_BOUND = 0.05                            # persona-specific share of cross-bank TV
H4_MARGIN = 0.20                           # secondary equivalence margin on the ratio D_avg / D_vote
H5_PRIMARY_BANK = "gpt-5-mini"             # bank where the exploratory referral disparity appeared
REFER_SHARE = 0.20                         # H5 referral budget
DIRECTIONAL = {                            # H2 pre-specified contrasts (higher - lower)
    "age": ("old", "young"),
    "edu": ("college", "high_school"),
    "ses": ("high", "low"),
}
LEVELS = {"age": ["young", "old"], "geo": ["urban", "suburban", "rural"],
          "edu": ["high_school", "college"], "ses": ["low", "middle", "high"]}
PERSONAS = [f"{a}_{g}_{e}_{s}" for a in LEVELS["age"] for g in LEVELS["geo"]
            for e in LEVELS["edu"] for s in LEVELS["ses"]]
# "high_school" is the only multi-token level, so age/geo are the first two tokens and ses the last
DIM_OF = {
    "age": np.array([p.split("_")[0] for p in PERSONAS]),
    "geo": np.array([p.split("_")[1] for p in PERSONAS]),
    "edu": np.array(["high_school" if "_high_school_" in p else "college" for p in PERSONAS]),
    "ses": np.array([p.split("_")[-1] for p in PERSONAS]),
}

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_panel(panel: str, tag: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str], Path]:
    if panel == "holdout":
        base = Path("holdout_out")
        s = pd.read_csv(base / "jev" / "scores.csv")
        dec_path = base / f"decisions_{tag}.csv"
        dec = pd.read_csv(dec_path) if dec_path.exists() else None
    else:
        base = Path("jev_pilot_out")
        s = pd.read_csv(base / "jev_scores.csv")
        s = s.rename(columns={f"p{i}_fwd": f"p{i}_o012" for i in range(3)} |
                     {f"p{i}_rev": f"p{i}_o210" for i in range(3)})
        extra = base / "orders" / "scores.csv"
        if extra.exists():
            e = pd.read_csv(extra)
            keep = ["text_id"] + [c for c in e.columns if c[:2] in ("p0", "p1", "p2") and c[3:] in ALL_ORDERS]
            s = s.merge(e[keep].drop_duplicates("text_id"), on="text_id", how="left")
        dec = pd.read_csv("output/results_persona_decisions_20260220_101746.csv",
                          usecols=["case_id", "model", "variant", "tier"])
    g = s[s.source == "guise"]
    orders = [o for o in ALL_ORDERS if f"p0_{o}" in g.columns and g[f"p0_{o}"].notna().all()]
    # complete cases: all 36 personas in both banks, all available orders
    counts = g.groupby(["case_id", "rewrite_model"]).persona_label.nunique().unstack()
    cases = sorted(counts.index[(counts.reindex(columns=BANKS) == 36).all(axis=1)])
    P: Dict[str, np.ndarray] = {}
    for b in BANKS:
        gb = g[g.rewrite_model == b].drop_duplicates(["case_id", "persona_label"]).set_index(["case_id", "persona_label"])
        gb = gb.reindex(pd.MultiIndex.from_product([cases, PERSONAS]))
        for o in orders:
            P[f"{b}|{o}"] = gb[[f"p0_{o}", f"p1_{o}", f"p2_{o}"]].to_numpy(float).reshape(len(cases), 36, 3)
    G: Dict[str, np.ndarray] = {}
    if dec is not None:
        for b in BANKS:
            d = dec[dec.model == b].drop_duplicates(["case_id", "variant"]).set_index(["case_id", "variant"]).tier
            G[b] = d.reindex(pd.MultiIndex.from_product([cases, PERSONAS])).to_numpy(float).reshape(len(cases), 36)
    P["_orders"] = np.array(orders)
    return P, G, cases, base


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def norm(p: np.ndarray) -> np.ndarray:
    return p / p.sum(-1, keepdims=True)


def etier(p: np.ndarray) -> np.ndarray:
    return p[..., 1] + 2 * p[..., 2]


def entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-12, 1)
    return -(q * np.log(q)).sum(-1) / math.log(3)


def tv(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(p - q).sum(-1)


def hard_vote(labels: List[np.ndarray]) -> np.ndarray:
    L = np.stack(labels, -1)
    return np.stack([(L == k).sum(-1) for k in range(3)], -1).argmax(-1).astype(float)  # ties -> lower tier


def D(X: np.ndarray) -> float:
    """Standardised systematic persona disparity, corrected for sampling noise.

    Two-way additive decomposition X[c,k] = a_c + b_k + e_ck (complaint x persona).
    The raw variance of the 36 persona means is var(b) + var(e)/C, so a noisier
    rule would look more disparate purely through its noise floor.  We use the
    unbiased ANOVA estimators
        var_b = var(persona means) - s2_e / C
        var_a = var(complaint means) - s2_e / K
    with s2_e the residual mean square, and return sqrt(max(var_b, 0) / var_a).
    """
    X = X[~np.isnan(X).any(axis=1)]
    C, K = X.shape
    rm, cm, gm = X.mean(1, keepdims=True), X.mean(0, keepdims=True), X.mean()
    s2_e = float(((X - rm - cm + gm) ** 2).sum() / ((C - 1) * (K - 1)))
    var_b = float(np.var(cm, ddof=1)) - s2_e / C
    var_a = float(np.var(rm, ddof=1)) - s2_e / K
    return float(math.sqrt(max(var_b, 0.0) / var_a)) if var_a > 0 else float("nan")


def perm_p_persona_sd(X: np.ndarray) -> float:
    obs = np.std(X.mean(0), ddof=1)
    ge = sum(np.std(rng.permuted(X, axis=1).mean(0), ddof=1) >= obs for _ in range(N_PERM))
    return (ge + 1) / (N_PERM + 1)


def contrast(X: np.ndarray, dim: str, hi: str, lo: str) -> np.ndarray:
    """Per-complaint within-complaint contrast (mean over hi-level guises - lo-level guises)."""
    lv = DIM_OF[dim]
    return np.nanmean(X[:, lv == hi], axis=1) - np.nanmean(X[:, lv == lo], axis=1)


def boot_ci(stat_fn, C: int, level: float = 0.95) -> Tuple[float, float, np.ndarray]:
    idx = rng.integers(0, C, size=(N_BOOT, C))
    vals = np.array([stat_fn(i) for i in idx])
    a = (1 - level) / 2
    return float(np.nanquantile(vals, a)), float(np.nanquantile(vals, 1 - a)), vals


def boot_p_two_sided(vals: np.ndarray) -> float:
    v = vals[~np.isnan(vals)]
    return float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()) + 1 / (len(v) + 1)))


def bh(ps: List[float]) -> List[float]:
    p = np.array(ps, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        prev = min(prev, p[i] * n / rank)
        q[i] = prev
    return q.tolist()


# ---------------------------------------------------------------------------
# Hypotheses
# ---------------------------------------------------------------------------

def h1(P: Dict[str, np.ndarray], C: int) -> Dict[str, Any]:
    A, B = P[f"{BANKS[0]}|{PRIMARY_ORDER}"], P[f"{BANKS[1]}|{PRIMARY_ORDER}"]
    same = tv(A, B)                                                     # C x 36
    diff = np.stack([tv(A[:, i][:, None, :], np.delete(B, i, axis=1)).mean(1) for i in range(36)], 1)
    def ratio(ix):
        s_, d_ = same[ix].mean(), diff[ix].mean()
        return (d_ - s_) / s_
    lo, hi, _ = boot_ci(ratio, C)
    est = ratio(np.arange(C))
    return {"estimate": est, "ci95": [lo, hi], "bound": H1_BOUND, "supported": hi < H1_BOUND,
            "tv_same_persona": float(same.mean()), "tv_different_persona": float(diff.mean())}


def h2(P: Dict[str, np.ndarray], C: int) -> Dict[str, Any]:
    out, ps, keys = {}, [], []
    for b in BANKS:
        p = P[f"{b}|{PRIMARY_ORDER}"]
        for metric, X in (("e_tier", etier(p)), ("argmax_tier", p.argmax(-1).astype(float))):
            r = {"persona_sd_perm_p": perm_p_persona_sd(X)}
            for dim, (hi_l, lo_l) in DIRECTIONAL.items():
                c = contrast(X, dim, hi_l, lo_l)
                lo, hi, vals = boot_ci(lambda ix: float(np.nanmean(c[ix])), C)
                pv = boot_p_two_sided(vals)
                r[f"{dim}:{hi_l}-{lo_l}"] = {"estimate": float(np.nanmean(c)), "ci95": [lo, hi], "p": pv}
                ps.append(pv)
                keys.append((b, metric, f"{dim}:{hi_l}-{lo_l}"))
            lv = DIM_OF["geo"]
            means = {l: float(np.nanmean(X[:, lv == l] - np.nanmean(X, 1, keepdims=True))) for l in LEVELS["geo"]}
            r["geo_level_effects_two_sided"] = means
            out.setdefault(b, {})[metric] = r
    q = bh(ps)
    for (b, metric, k), qv in zip(keys, q):
        e = out[b][metric][k]
        e["q_bh"] = qv
        e["supported"] = bool(qv < ALPHA and e["estimate"] > 0)
    return out


def residual_share(X: np.ndarray) -> float:
    """Arbitrariness: residual (complaint x persona interaction) variance as a share of
    residual + between-complaint variance."""
    rm, cm, gm = X.mean(1, keepdims=True), X.mean(0, keepdims=True), X.mean()
    e = float(((X - rm - cm + gm) ** 2).mean())
    return e / (e + float(np.var(rm)))


def h3(P: Dict[str, np.ndarray], C: int) -> Dict[str, Any]:
    """H3a (confirmatory): E[tier] has a lower residual share than argmax in each bank.
    H3b (estimation only): difference in systematic disparity D, no directional claim."""
    out = {}
    for b in BANKS:
        p = P[f"{b}|{PRIMARY_ORDER}"]
        am, et = p.argmax(-1).astype(float), etier(p)
        ra = lambda ix: residual_share(am[ix]) - residual_share(et[ix])
        lo_a, hi_a, _ = boot_ci(ra, C)
        rb = lambda ix: D(am[ix]) - D(et[ix])
        lo_b, hi_b, _ = boot_ci(rb, C)
        out[b] = {"H3a_residual_share_argmax": residual_share(am), "H3a_residual_share_e_tier": residual_share(et),
                  "H3a_delta": ra(np.arange(C)), "H3a_ci95": [lo_a, hi_a], "H3a_supported": bool(lo_a > 0),
                  "H3b_D_argmax": D(am), "H3b_D_e_tier": D(et), "H3b_delta": rb(np.arange(C)),
                  "H3b_ci95": [lo_b, hi_b]}
    out["H3a_supported"] = bool(all(out[b]["H3a_supported"] for b in BANKS))
    return out


def h4(P: Dict[str, np.ndarray], C: int) -> Dict[str, Any]:
    out = {}
    lvl = 1 - 2 * ALPHA                                      # 90% CI for TOST-style equivalence
    # (a) two same-style paraphrases (cross-bank), primary order
    A, B = P[f"{BANKS[0]}|{PRIMARY_ORDER}"], P[f"{BANKS[1]}|{PRIMARY_ORDER}"]
    avg = norm(A + B).argmax(-1).astype(float)
    vote = hard_vote([A.argmax(-1), B.argmax(-1)])
    comps = {"paraphrase_2": (avg, vote)}
    # (b) all available option orders, per bank
    orders = list(P["_orders"])
    if len(orders) >= 3:
        for b in BANKS:
            ps = [P[f"{b}|{o}"] for o in orders]
            comps[f"orders_{len(orders)}_{b}"] = (norm(sum(ps)).argmax(-1).astype(float),
                                                 hard_vote([x.argmax(-1) for x in ps]))
    # Prediction: probability averaging is NOT superior to equal-budget hard voting.
    # Refuted if the one-sided 95% upper bound of D_avg / D_vote is below 1 (i.e. 90% CI upper < 1).
    for name, (xa, xv) in comps.items():
        ratio = lambda ix: D(xa[ix]) / D(xv[ix]) if D(xv[ix]) > 0 else float("nan")
        lo, hi, _ = boot_ci(ratio, C, level=lvl)
        est = ratio(np.arange(C))
        out[name] = {"D_prob_avg": D(xa), "D_hard_vote": D(xv), "ratio": est, "ci90": [lo, hi],
                     "prediction_refuted_prob_avg_superior": bool(hi < 1.0),
                     "secondary_equivalent_within_margin": bool(lo > 1 - H4_MARGIN and hi < 1 + H4_MARGIN)}
    out["supported"] = bool(not any(v["prediction_refuted_prob_avg_superior"] for v in out.values()
                                    if isinstance(v, dict)))
    return out


def h5(P: Dict[str, np.ndarray], C: int) -> Dict[str, Any]:
    out = {}
    for b in BANKS:
        H = entropy(P[f"{b}|{PRIMARY_ORDER}"])
        refer = (H >= np.quantile(H, 1 - REFER_SHARE)).astype(float)
        r = {"referral_rate": float(refer.mean()), "persona_sd_perm_p": perm_p_persona_sd(refer)}
        for dim, (hi_l, lo_l) in DIRECTIONAL.items():
            c = contrast(refer, dim, hi_l, lo_l)
            lo, hi, vals = boot_ci(lambda ix: float(np.nanmean(c[ix])), C)
            r[f"{dim}:{hi_l}-{lo_l}"] = {"estimate": float(np.nanmean(c)), "ci95": [lo, hi], "p": boot_p_two_sided(vals)}
        out[b] = r
    # Confirmatory test in the primary bank only; the other bank is reported descriptively
    out["supported"] = bool(out[H5_PRIMARY_BANK]["persona_sd_perm_p"] < ALPHA)
    return out


def h6(P: Dict[str, np.ndarray], G: Dict[str, np.ndarray], C: int) -> Optional[Dict[str, Any]]:
    if not G:
        return None
    out = {}
    for b in BANKS:
        g = G[b]
        gm = (g == 2).astype(float)
        gm[np.isnan(g)] = np.nan
        p2 = P[f"{b}|{PRIMARY_ORDER}"][..., 2]
        rate = float(np.nanmean(gm))
        jm = (p2 >= np.quantile(p2, 1 - rate)).astype(float)
        diff = lambda ix: D(gm[ix]) - D(jm[ix])
        lo, hi, _ = boot_ci(diff, C)
        out[b] = {"gpt_monetary_rate": rate, "jev_monetary_rate": float(jm.mean()),
                  "D_gpt": D(gm), "D_jev": D(jm), "delta_gpt_minus_jev": diff(np.arange(C)), "ci95": [lo, hi],
                  "jev_lower": bool(lo > 0)}
    out["supported"] = bool(out[BANKS[0]]["jev_lower"] and not out[BANKS[1]]["jev_lower"])
    return out


def to_md(res: Dict[str, Any]) -> str:
    lines = [f"# Confirmatory results ({res['panel']})", "",
             f"Complaints: {res['n_complaints']}; orders available: {', '.join(res['orders'])}; "
             f"seed {SEED}; bootstrap {N_BOOT}; permutations {N_PERM}", ""]
    for h in ("H1", "H2", "H3", "H4", "H5", "H6"):
        lines += [f"## {h}", "```", json.dumps(res[h], indent=2, default=float), "```", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", choices=["holdout", "exploratory"], required=True)
    ap.add_argument("--tag", default="n1000")
    args = ap.parse_args()
    P, G, cases, base = load_panel(args.panel, args.tag)
    C = len(cases)
    res = {"panel": args.panel, "n_complaints": C, "orders": list(P["_orders"]),
           "H1": h1(P, C), "H2": h2(P, C), "H3": h3(P, C), "H4": h4(P, C), "H5": h5(P, C), "H6": h6(P, G, C)}
    (base / "confirmatory_results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    (base / "confirmatory_results.md").write_text(to_md(res), encoding="utf-8")
    print(to_md(res))


if __name__ == "__main__":
    main()
