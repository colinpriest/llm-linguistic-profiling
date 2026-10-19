"""Exploratory distributional audit of the Jev matched-guise scores.

Uses only existing outputs (jev_pilot_out/jev_scores.csv and the frozen GPT
decision CSV); makes no API calls.  Run from the repository root:

    python jev_pilot_out/jev_distributional_audit.py

Questions:
  Q1  Does using probabilities (rather than a single prediction) reveal or
      create new forms of style-linked disparity?
  Q2  Can rules that use the full distribution reduce style sensitivity while
      keeping between-complaint (substantive) signal?

All results are exploratory: the full panel was already inspected.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.schemas import PERSONA_GRID, persona_label  # noqa: E402

OUT = Path("jev_pilot_out/dist_audit")
RUN_TS = "20260220_101746"
BANKS = ["gpt-4o-mini", "gpt-5-mini"]
LABELS = [persona_label(**p) for p in PERSONA_GRID]
DIMS = {d: np.array([p[d] for p in PERSONA_GRID]) for d in ("age", "geo", "edu", "ses")}
N_BOOT = 1000
N_PERM = 1000
SEED = 7
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load() -> Tuple[Dict[str, np.ndarray], List[str], pd.DataFrame]:
    s = pd.read_csv("jev_pilot_out/jev_scores.csv")
    g = s[s.source == "guise"]
    cases = sorted(g.case_id.unique())
    arrays: Dict[str, np.ndarray] = {}
    for b in BANKS:
        gb = g[g.rewrite_model == b].set_index(["case_id", "persona_label"])
        idx = pd.MultiIndex.from_product([cases, LABELS])
        gb = gb.reindex(idx)
        assert gb.p0_fwd.notna().all(), f"missing guises in {b}"
        C, K = len(cases), len(LABELS)
        arrays[f"{b}|fwd"] = gb[["p0_fwd", "p1_fwd", "p2_fwd"]].to_numpy().reshape(C, K, 3)
        arrays[f"{b}|rev"] = gb[["p0_rev", "p1_rev", "p2_rev"]].to_numpy().reshape(C, K, 3)
    gpt = pd.read_csv(f"output/results_persona_decisions_{RUN_TS}.csv",
                      usecols=["case_id", "model", "variant", "tier"])
    for b in BANKS:
        gg = gpt[gpt.model == b].set_index(["case_id", "variant"]).tier
        gg = gg[~gg.index.duplicated()].reindex(pd.MultiIndex.from_product([cases, LABELS]))
        arrays[f"{b}|gpt_tier"] = gg.to_numpy(dtype=float).reshape(len(cases), len(LABELS))
    return arrays, cases, s


def norm(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0, None)
    return p / p.sum(axis=-1, keepdims=True)


def entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-12, 1)
    return -(q * np.log(q)).sum(axis=-1) / math.log(3)


def tv(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(p - q).sum(axis=-1)


# ---------------------------------------------------------------------------
# Generic statistics on a complaint x persona matrix X (C x 36)
# ---------------------------------------------------------------------------

def dim_gaps(X: np.ndarray) -> Dict[str, Any]:
    """Within-complaint level effects per dimension, with complaint bootstrap CI."""
    dev = X - np.nanmean(X, axis=1, keepdims=True)
    out: Dict[str, Any] = {}
    C = X.shape[0]
    boot_idx = rng.integers(0, C, size=(N_BOOT, C))
    for d, lv in DIMS.items():
        levels = list(dict.fromkeys(lv))
        M = np.stack([np.nanmean(dev[:, lv == l], axis=1) for l in levels], axis=1)  # C x L
        means = M.mean(axis=0)
        gap = float(means.max() - means.min())
        bm = M[boot_idx].mean(axis=1)  # B x L
        bgap = bm.max(axis=1) - bm.min(axis=1)
        # CI for the fixed contrast hi-lo chosen on the full sample (less biased than max-min)
        hi, lo = int(np.argmax(means)), int(np.argmin(means))
        bcon = bm[:, hi] - bm[:, lo]
        out[d] = {
            "level_effects": {l: round(float(m), 5) for l, m in zip(levels, means)},
            "gap": round(gap, 5),
            "contrast": f"{levels[hi]}-{levels[lo]}",
            "contrast_ci95": [round(float(np.quantile(bcon, 0.025)), 5),
                              round(float(np.quantile(bcon, 0.975)), 5)],
            "gap_boot_median": round(float(np.median(bgap)), 5),
        }
    return out


def persona_sd(X: np.ndarray) -> Tuple[float, float]:
    """SD of the 36 persona means and within-complaint permutation p."""
    obs = float(np.std(X.mean(axis=0), ddof=1))
    ge = 0
    for _ in range(N_PERM):
        if np.std(rng.permuted(X, axis=1).mean(axis=0), ddof=1) >= obs:
            ge += 1
    return obs, (ge + 1) / (N_PERM + 1)


def signal_noise(X: np.ndarray) -> Dict[str, float]:
    """Between-complaint variance (signal kept) vs within-complaint variance
    across guises (style + paraphrase sensitivity)."""
    within = float(np.mean(np.var(X, axis=1, ddof=1)))
    between = float(np.var(X.mean(axis=1), ddof=1))
    return {"within_var": within, "between_var": between,
            "within_share": within / (within + between) if within + between > 0 else float("nan")}


def describe_rule(X: np.ndarray, with_perm: bool = True) -> Dict[str, Any]:
    r: Dict[str, Any] = {"mean": float(np.mean(X))}
    r.update(signal_noise(X))
    if with_perm:
        sd, p = persona_sd(X)
        r["persona_sd"], r["persona_perm_p"] = sd, p
    r["dim_gaps"] = dim_gaps(X)
    return r


# ---------------------------------------------------------------------------
# Q1a: what single predictions hide
# ---------------------------------------------------------------------------

def hidden_variation(P: np.ndarray, Prev: np.ndarray) -> Dict[str, Any]:
    C, K, _ = P.shape
    am = P.argmax(-1)
    et = P[..., 1] + 2 * P[..., 2]
    H = entropy(P)
    iu = np.triu_indices(K, 1)
    pair_tv = tv(P[:, iu[0]], P[:, iu[1]])               # C x 630
    pair_same = am[:, iu[0]] == am[:, iu[1]]
    pair_det = np.abs(et[:, iu[0]] - et[:, iu[1]])
    const = (am == am[:, :1]).all(axis=1)
    order_tv = tv(P, Prev)                               # same text, options reversed
    res = {
        "complaints_argmax_constant_across_36": float(const.mean()),
        "mean_pairwise_tv_all": float(pair_tv.mean()),
        "mean_pairwise_tv_same_argmax_pairs": float(pair_tv[pair_same].mean()),
        "share_pairs_same_argmax": float(pair_same.mean()),
        "share_same_argmax_pairs_tv_gt_0.10": float((pair_tv[pair_same] > 0.10).mean()),
        "share_same_argmax_pairs_tv_gt_0.20": float((pair_tv[pair_same] > 0.20).mean()),
        "share_pairs_mean_invisible": float(((pair_det < 0.05) & (pair_tv > 0.15)).mean()),
        "in_argmax_constant_complaints": {
            "mean_pairwise_tv": float(pair_tv[const].mean()) if const.any() else None,
            "mean_range_p2": float((P[const, :, 2].max(1) - P[const, :, 2].min(1)).mean()) if const.any() else None,
            "mean_range_entropy": float((H[const].max(1) - H[const].min(1)).mean()) if const.any() else None,
            "share_with_p2_range_gt_0.10": float(((P[const, :, 2].max(1) - P[const, :, 2].min(1)) > 0.10).mean()) if const.any() else None,
        },
        "baseline_order_reversal_tv_same_text": float(order_tv.mean()),
    }
    return res


def paraphrase_vs_style(Pa: np.ndarray, Pb: np.ndarray) -> Dict[str, Any]:
    """Cross-bank comparisons: same persona (two paraphrases of the same style)
    vs different personas.  Both comparisons cross the rewrite generators, so the
    difference isolates the style-specific contribution."""
    C, K, _ = Pa.shape
    same = tv(Pa, Pb).mean()
    # different-persona cross-bank pairs
    diff = np.mean([tv(Pa[:, i][:, None, :], np.delete(Pb, i, axis=1)).mean() for i in range(K)])
    # pairs differing on exactly one dimension, per dimension
    per_dim = {}
    keys = list(DIMS)
    for d in keys:
        vals = []
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                diffs = [k for k in keys if DIMS[k][i] != DIMS[k][j]]
                if diffs == [d]:
                    vals.append(tv(Pa[:, i], Pb[:, j]).mean())
        per_dim[d] = float(np.mean(vals))
    return {"cross_bank_tv_same_persona": float(same),
            "cross_bank_tv_different_persona": float(diff),
            "cross_bank_tv_one_dim_changed": per_dim}


# ---------------------------------------------------------------------------
# Q1b/c: policies applied to the same probabilities
# ---------------------------------------------------------------------------

def policies(P: np.ndarray, gpt_tier: np.ndarray) -> Dict[str, Any]:
    am = P.argmax(-1)
    p2 = P[..., 2]
    H = entropy(P)
    base_rate = float((am == 2).mean())
    t_eq = float(np.quantile(p2, 1 - base_rate))     # equal monetary budget
    gpt_rate = float(np.nanmean(gpt_tier == 2))
    t_gpt = float(np.quantile(p2, 1 - gpt_rate))      # monetary budget matched to GPT
    out: Dict[str, Any] = {"argmax_monetary_rate": base_rate, "p2_threshold_equal_budget": t_eq,
                           "gpt_monetary_rate": gpt_rate, "p2_threshold_gpt_budget": t_gpt}
    rules: Dict[str, np.ndarray] = {
        "monetary_argmax": (am == 2).astype(float),
        "monetary_p2_threshold_equal_budget": (p2 >= t_eq).astype(float),
        "monetary_p2_threshold_gpt_budget": (p2 >= t_gpt).astype(float),
        "gpt_monetary": (gpt_tier == 2).astype(float),
        "any_action_argmax": (am >= 1).astype(float),
        "any_action_p0_threshold_equal_budget": (P[..., 0] <= np.quantile(P[..., 0], (am >= 1).mean())).astype(float),
        "gpt_any_action": (gpt_tier >= 1).astype(float),
    }
    for q in (0.10, 0.20):
        thr = np.quantile(H, 1 - q)
        refer = (H >= thr).astype(float)
        rules[f"refer_top{int(q*100)}pct_entropy"] = refer
        # automated monetary decisions among those not referred
        auto_mon = np.where(refer == 1, np.nan, (am == 2).astype(float))
        out[f"refer_top{int(q*100)}pct_auto_monetary_rate"] = float(np.nanmean(auto_mon))
    out["continuous"] = {
        "p2": describe_rule(p2),
        "entropy": describe_rule(H),
        "max_prob": describe_rule(P.max(-1)),
    }
    out["rules"] = {k: describe_rule(v) for k, v in rules.items()}
    return out


# ---------------------------------------------------------------------------
# Q2: mitigation rules at equal inference budgets
# ---------------------------------------------------------------------------

def hard_vote(*labels: np.ndarray) -> np.ndarray:
    """Majority over hard labels; ties broken toward the lower tier."""
    L = np.stack(labels, -1)
    counts = np.stack([(L == k).sum(-1) for k in range(3)], -1)
    return counts.argmax(-1).astype(float)   # argmax picks lowest index on ties


def mitigation(arr: Dict[str, np.ndarray]) -> Dict[str, Any]:
    a_f, a_r = arr["gpt-4o-mini|fwd"], arr["gpt-4o-mini|rev"]
    b_f, b_r = arr["gpt-5-mini|fwd"], arr["gpt-5-mini|rev"]
    rules: Dict[str, Tuple[int, Callable[[], np.ndarray]]] = {}
    for tag, F, R in (("bank4o", a_f, a_r), ("bank5", b_f, b_r)):
        rules[f"{tag}: argmax fwd (1 call)"] = (1, lambda F=F: F.argmax(-1).astype(float))
        rules[f"{tag}: E[tier] fwd (1 call)"] = (1, lambda F=F: F[..., 1] + 2 * F[..., 2])
        rules[f"{tag}: argmax of order-averaged probs (2 calls)"] = (2, lambda F=F, R=R: norm(F + R).argmax(-1).astype(float))
        rules[f"{tag}: hard vote fwd/rev, tie->lower (2 calls)"] = (2, lambda F=F, R=R: hard_vote(F.argmax(-1), R.argmax(-1)))
        rules[f"{tag}: mean of fwd/rev argmax (2 calls)"] = (2, lambda F=F, R=R: (F.argmax(-1) + R.argmax(-1)) / 2.0)
        rules[f"{tag}: E[tier] order-averaged (2 calls)"] = (2, lambda F=F, R=R: (lambda P: P[..., 1] + 2 * P[..., 2])(norm(F + R)))
    # Paraphrase ensembles: two independent same-style rewrites of each guise
    rules["both banks: argmax of paraphrase-averaged probs (2 calls)"] = (2, lambda: norm(a_f + b_f).argmax(-1).astype(float))
    rules["both banks: hard vote of 2 paraphrases, tie->lower (2 calls)"] = (2, lambda: hard_vote(a_f.argmax(-1), b_f.argmax(-1)))
    rules["both banks: mean of 2 paraphrase argmaxes (2 calls)"] = (2, lambda: (a_f.argmax(-1) + b_f.argmax(-1)) / 2.0)
    rules["both banks: argmax of 4-way averaged probs (4 calls)"] = (4, lambda: norm(a_f + a_r + b_f + b_r).argmax(-1).astype(float))
    rules["both banks: hard vote of 4 labels, tie->lower (4 calls)"] = (4, lambda: hard_vote(a_f.argmax(-1), a_r.argmax(-1), b_f.argmax(-1), b_r.argmax(-1)))
    rules["both banks: E[tier] 4-way averaged (4 calls)"] = (4, lambda: (lambda P: P[..., 1] + 2 * P[..., 2])(norm(a_f + a_r + b_f + b_r)))
    out: Dict[str, Any] = {}
    for name, (calls, fn) in rules.items():
        X = fn()
        d = describe_rule(X)
        d["calls"] = calls
        d["dist"] = {str(k): float(np.mean(X == k)) for k in (0, 1, 2)} if np.all(np.isin(X, [0, 1, 2])) else None
        out[name] = d
    # monetary decisions under the same ensembles (budget comparisons)
    mon = {
        "bank4o argmax fwd": (a_f.argmax(-1) == 2),
        "bank4o order-avg argmax": (norm(a_f + a_r).argmax(-1) == 2),
        "bank4o hard vote fwd/rev (tie->lower)": (hard_vote(a_f.argmax(-1), a_r.argmax(-1)) == 2),
        "paraphrase-avg argmax": (norm(a_f + b_f).argmax(-1) == 2),
        "paraphrase hard vote (tie->lower)": (hard_vote(a_f.argmax(-1), b_f.argmax(-1)) == 2),
        "4-way avg argmax": (norm(a_f + a_r + b_f + b_r).argmax(-1) == 2),
        "4-way hard vote (tie->lower)": (hard_vote(a_f.argmax(-1), a_r.argmax(-1), b_f.argmax(-1), b_r.argmax(-1)) == 2),
    }
    out["_monetary"] = {k: describe_rule(v.astype(float)) for k, v in mon.items()}
    return out


# ---------------------------------------------------------------------------
# Neutralisation (originals only; guises were never neutralised)
# ---------------------------------------------------------------------------

def neutralisation(s: pd.DataFrame) -> Dict[str, Any]:
    o = s[s.source == "original"].set_index("case_id")[["p0_fwd", "p1_fwd", "p2_fwd"]]
    out = {}
    for b in BANKS:
        n = s[(s.source == "neutralized") & (s.rewrite_model == b)].set_index("case_id")[["p0_fwd", "p1_fwd", "p2_fwd"]]
        j = o.join(n, rsuffix="_n", how="inner")
        P, Q = j.iloc[:, :3].to_numpy(), j.iloc[:, 3:].to_numpy()
        out[b] = {"n": len(j), "mean_tv_original_vs_neutralised": float(tv(P, Q).mean()),
                  "argmax_change_rate": float((P.argmax(1) != Q.argmax(1)).mean()),
                  "mean_p2_original": float(P[:, 2].mean()), "mean_p2_neutralised": float(Q[:, 2].mean()),
                  "mean_entropy_original": float(entropy(P).mean()), "mean_entropy_neutralised": float(entropy(Q).mean())}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arr, cases, s = load()
    res: Dict[str, Any] = {"n_complaints": len(cases), "n_personas": len(LABELS),
                           "n_boot": N_BOOT, "n_perm": N_PERM}
    res["q1_hidden_variation"] = {b: hidden_variation(arr[f"{b}|fwd"], arr[f"{b}|rev"]) for b in BANKS}
    for b in BANKS:
        g = arr[f"{b}|gpt_tier"]
        res["q1_hidden_variation"][b]["gpt_complaints_tier_constant_across_36"] = float(
            np.mean([(np.unique(r[~np.isnan(r)]).size == 1) for r in g]))
    res["q1_paraphrase_vs_style"] = paraphrase_vs_style(arr["gpt-4o-mini|fwd"], arr["gpt-5-mini|fwd"])
    res["q1_policies"] = {b: policies(arr[f"{b}|fwd"], arr[f"{b}|gpt_tier"]) for b in BANKS}
    res["q2_mitigation"] = mitigation(arr)
    res["neutralisation_originals"] = neutralisation(s)
    (OUT / "dist_audit.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"wrote {OUT / 'dist_audit.json'}")


if __name__ == "__main__":
    main()
