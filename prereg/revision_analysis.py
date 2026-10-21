"""POST HOC analyses responding to the 21 Oct adversarial review (no new API calls; existing outputs only).

R1  Variance components for Jev on the holdout (replaces the "99% arbitrariness" reading of H1).
    Fully crossed random-effects ANOVA, complaint (C) x persona (K) x rewrite bank (B) x option order (O),
    one observation per cell, solved by expected mean squares (Cornfield-Tukey). Components are grouped into:
      complaint; persona main effect; complaint-specific persona effect that replicates across both
      independent paraphrases (C x K); paraphrase-specific (any term with B but not O); option order and
      inference noise (any term with O, including the residual).
R2  Remedy rules on the holdout: what the complainant would actually receive (monetary vs not), for GPT
    single labels and for decision rules on Jev's distribution, at matched monetary volume.
R3  The mitigation arms under explicit remedy rules (M0, M1, M2 with majority vote or a volume-matched
    threshold on the averaged probability).

    python prereg/revision_analysis.py            -> prereg/revision_analysis_results.json
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "prereg")
import confirmatory_analysis as ca  # noqa: E402

SEED = 20261026
N_BOOT = 1000
N_BOOT_VC = 300
rng = np.random.default_rng(SEED)
BANK = "gpt-4o-mini"
CONTR = {"age": ("old", "young"), "edu": ("college", "high_school"), "ses": ("high", "low")}


def lvl(lbl: str, dim: str) -> str:
    return {"age": lbl.split("_")[0], "edu": "high_school" if "_high_school_" in lbl else "college", "ses": lbl.split("_")[-1]}[dim]


# ---------------------------------------------------------------------------
# R1: variance components
# ---------------------------------------------------------------------------

FACTORS = "CKBO"


def ems_components(Y: np.ndarray) -> Dict[str, float]:
    """Random-effects variance components for a fully crossed design with one observation per cell."""
    n = Y.shape
    nf = len(n)
    subsets = [s for r in range(1, nf + 1) for s in itertools.combinations(range(nf), r)]
    means = {(): Y.mean()}
    for s in subsets:
        other = tuple(i for i in range(nf) if i not in s)
        means[s] = Y.mean(axis=other) if other else Y
    ms = {}
    for s in subsets:
        eff = np.zeros(tuple(n[i] for i in s))
        for r in range(len(s) + 1):
            for t in itertools.combinations(s, r):
                m = means[t]
                # broadcast m (over dims t) into dims s
                shape = [n[i] if i in t else 1 for i in s]
                eff = eff + (-1) ** (len(s) - len(t)) * np.reshape(m, shape)
        rest = int(np.prod([n[i] for i in range(nf) if i not in s])) if len(s) < nf else 1
        ss = rest * float((eff ** 2).sum())
        df = int(np.prod([n[i] - 1 for i in s]))
        ms[s] = ss / df
    sig = {}
    for s in sorted(subsets, key=len, reverse=True):
        acc = ms[s]
        for u in subsets:
            if set(u) > set(s):
                acc -= int(np.prod([n[i] for i in range(nf) if i not in u])) * sig[u]
        sig[s] = acc / int(np.prod([n[i] for i in range(nf) if i not in s]))
    return {"".join(FACTORS[i] for i in s): v for s, v in sig.items()}


def group_components(sig: Dict[str, float]) -> Dict[str, float]:
    s = {k: max(v, 0.0) for k, v in sig.items()}
    g = {"complaint": s["C"], "persona_main": s["K"], "complaint_x_persona_replicated": s["CK"],
         "rewrite_model_systematic": s["B"] + s["CB"] + s["KB"],      # same shift for every guise of a complaint in a bank
         "guise_paraphrase": s["CKB"],                                # this guise's paraphrase, not shared across banks
         "order_and_inference_noise": sum(v for k, v in s.items() if "O" in k)}
    within = sum(v for k, v in g.items() if k != "complaint")
    shares = {f"share_within_{k}": v / within for k, v in g.items() if k != "complaint"}
    return {**g, "within_complaint_total": within, **shares}


def r1() -> Dict[str, Any]:
    P, G, cases, _ = ca.load_panel("holdout", "n1000")
    orders = list(P["_orders"])
    out: Dict[str, Any] = {"n_complaints": len(cases), "orders": orders}
    for name, f in (("p_monetary", lambda p: p[..., 2]), ("e_tier", lambda p: p[..., 1] + 2 * p[..., 2])):
        Y = np.stack([np.stack([f(P[f"{b}|{o}"]) for o in orders], -1) for b in ca.BANKS], 2)  # C x 36 x B x O
        Y = Y[~np.isnan(Y).any(axis=(1, 2, 3))]
        base = group_components(ems_components(Y))
        C = len(Y)
        boots = [group_components(ems_components(Y[rng.integers(0, C, C)])) for _ in range(N_BOOT_VC)]
        out[name] = {k: {"estimate": v, "ci95_approx": [float(np.quantile([b[k] for b in boots], .025)),
                                                       float(np.quantile([b[k] for b in boots], .975))]}
                     for k, v in base.items()}
        out[name]["raw_components"] = ems_components(Y)
    return out


# ---------------------------------------------------------------------------
# Helpers for decision rules
# ---------------------------------------------------------------------------

def pairwise_disagree_rate(X: np.ndarray) -> np.ndarray:
    """Per complaint: share of guise pairs with different binary outcomes."""
    k = X.shape[1]
    m = np.nansum(X, 1)
    return (m * (k - m)) / (k * (k - 1) / 2)


def gaps_pp(X: np.ndarray, personas: List[str]) -> Dict[str, Any]:
    out = {}
    C = len(X)
    for dim, (hi, lo) in CONTR.items():
        lv = np.array([lvl(p, dim) for p in personas])
        c = X[:, lv == hi].mean(1) - X[:, lv == lo].mean(1)
        b = [c[rng.integers(0, C, C)].mean() for _ in range(N_BOOT)]
        out[dim] = {"gap_pp": 100 * float(c.mean()), "ci95_pp": [100 * float(np.quantile(b, .025)), 100 * float(np.quantile(b, .975))]}
    return out


def summarise_rule(X: np.ndarray, personas: List[str]) -> Dict[str, Any]:
    d = pairwise_disagree_rate(X)
    C = len(X)
    b = [d[rng.integers(0, C, C)].mean() for _ in range(N_BOOT)]
    return {"monetary_rate": float(np.nanmean(X)),
            "p_two_versions_differ": float(d.mean()), "p_two_versions_differ_ci95": [float(np.quantile(b, .025)), float(np.quantile(b, .975))],
            "share_complaints_not_unanimous": float(np.mean((X.min(1) != X.max(1)))),
            "gaps": gaps_pp(X, personas)}


def paired_diff(X1: np.ndarray, X2: np.ndarray) -> Dict[str, Any]:
    """Difference in P(two versions differ) between two rules on the same complaints, complaint bootstrap."""
    d = pairwise_disagree_rate(X1) - pairwise_disagree_rate(X2)
    C = len(d)
    b = [d[rng.integers(0, C, C)].mean() for _ in range(N_BOOT)]
    return {"estimate": float(d.mean()), "ci95": [float(np.quantile(b, .025)), float(np.quantile(b, .975))]}


def matched_threshold(score: np.ndarray, rate: float) -> float:
    return float(np.quantile(score[~np.isnan(score)], 1 - rate))


def fact_span(mon: Dict[str, np.ndarray]) -> Dict[str, Any]:
    s, w = mon["strengthen"], mon["weaken"]
    d = s - w
    C = len(d)
    b = [d[rng.integers(0, C, C)].mean() for _ in range(N_BOOT)]
    return {"pp": 100 * float(d.mean()), "ci95_pp": [100 * float(np.quantile(b, .025)), 100 * float(np.quantile(b, .975))], "n": int(C)}


# ---------------------------------------------------------------------------
# R2: remedy rules on the holdout
# ---------------------------------------------------------------------------

def r2() -> Dict[str, Any]:
    P, G, cases, _ = ca.load_panel("holdout", "n1000")
    orders = list(P["_orders"])
    personas = ca.PERSONAS
    pcj = pd.read_csv("holdout_out/positive_controls/jev/scores.csv")
    pcd = pd.read_csv("holdout_out/positive_controls/decisions.csv")
    out: Dict[str, Any] = {}
    for b in ca.BANKS:
        g = G[b]
        keep = ~np.isnan(g).any(1)
        gm = (g[keep] == 2).astype(float)
        rate = float(gm.mean())
        p = {o: P[f"{b}|{o}"][keep] for o in orders}
        p2_mean = np.mean([p[o][..., 2] for o in orders], 0)
        t1 = matched_threshold(p[ca.PRIMARY_ORDER][..., 2], rate)
        t2 = matched_threshold(p2_mean, rate)
        votes = np.sum([p[o].argmax(-1) == 2 for o in orders], 0)
        rules = {
            "gpt_label": gm,
            "jev_argmax_o012": (p[ca.PRIMARY_ORDER].argmax(-1) == 2).astype(float),
            "jev_p2_threshold_matched": (p[ca.PRIMARY_ORDER][..., 2] >= t1).astype(float),
            "jev_p2_mean6orders_threshold_matched": (p2_mean >= t2).astype(float),
            "jev_majority_6orders": (votes > len(orders) / 2).astype(float),
        }
        res = {k: summarise_rule(v, personas) for k, v in rules.items()}
        res["paired_diff_p_differ"] = {f"{a}_minus_{b_}": paired_diff(rules[a], rules[b_]) for a, b_ in (
            ("gpt_label", "jev_p2_threshold_matched"), ("jev_p2_threshold_matched", "jev_p2_mean6orders_threshold_matched"),
            ("jev_p2_threshold_matched", "jev_majority_6orders"), ("jev_argmax_o012", "jev_majority_6orders"))}
        res["thresholds"] = {"p2_o012": t1, "p2_mean6": t2, "gpt_monetary_rate": rate}
        # fact responsiveness of each rule (positive-control edits; first 200 holdout complaints)
        fs = {}
        d = pcd[pcd.model == b].pivot_table(index="case_id", columns="edit", values="tier")
        fs["gpt_label"] = fact_span({e: (d[e] == 2).astype(float).to_numpy() for e in ("strengthen", "weaken")})
        j = pcj[pcj.persona_label.isin(["strengthen", "weaken"])].drop_duplicates(["case_id", "persona_label"])
        piv = {e: j[j.persona_label == e].set_index("case_id").sort_index() for e in ("strengthen", "weaken")}
        common = piv["strengthen"].index.intersection(piv["weaken"].index)
        for rule, fn in (("jev_argmax_o012", lambda x: (x["argmax_o012"] == 2)),
                         ("jev_p2_threshold_matched", lambda x: x["p2_o012"] >= t1),
                         ("jev_p2_mean6orders_threshold_matched", lambda x: x[[f"p2_{o}" for o in orders]].mean(1) >= t2),
                         ("jev_majority_6orders", lambda x: sum((x[[f"p0_{o}", f"p1_{o}", f"p2_{o}"]].to_numpy().argmax(1) == 2) for o in orders) > len(orders) / 2)):
            fs[rule] = fact_span({e: np.asarray(fn(piv[e].loc[common]), float) for e in ("strengthen", "weaken")})
        res["fact_span_monetary"] = fs
        out[b] = res
    return out


# ---------------------------------------------------------------------------
# R3: mitigation arms under remedy rules
# ---------------------------------------------------------------------------

def r3() -> Dict[str, Any]:
    ids = pd.read_csv("holdout_out/holdout_sample_full.csv").case_id.tolist()
    main, pcs = ids[:500], ids[:200]
    personas = ca.PERSONAS
    styles = ["A_plain", "B_formal", "C_simple"]
    d0 = pd.read_csv("holdout_out/decisions_n1000.csv")
    d0 = d0[(d0.model == BANK) & d0.case_id.isin(main)].rename(columns={"variant": "persona_label"})
    dm = pd.read_csv("holdout_out/style_mitigation/decisions.csv")
    j0 = pd.read_csv("holdout_out/jev/scores.csv")
    j0 = j0[(j0.source == "guise") & (j0.rewrite_model == BANK) & j0.case_id.isin(main)].drop_duplicates(["case_id", "persona_label"])
    jm = pd.read_csv("holdout_out/style_mitigation/jev/scores.csv").drop_duplicates(["case_id", "persona_label", "rewrite_model"])

    def mat(f: pd.DataFrame, val: str) -> np.ndarray:
        return f.pivot_table(index="case_id", columns="persona_label", values=val).reindex(index=main, columns=personas).to_numpy(float)

    gm = {"M0": mat(d0.assign(m=(d0.tier == 2).astype(float)), "m")}
    gstyle = {s: mat(dm[(dm["style"] == s) & dm.case_id.isin(main)].assign(m=lambda x: (x.tier == 2).astype(float)), "m") for s in styles}
    gm["M1"] = gstyle["A_plain"]
    gm["M2_majority"] = (np.sum([gstyle[s] for s in styles], 0) >= 2).astype(float)
    jg = jm[jm.source == "guise"]
    jp2 = {s: mat(jg[jg.rewrite_model == s], "p2_o012") for s in styles}
    jarg = {s: mat(jg[jg.rewrite_model == s].assign(m=lambda x: (x.argmax_o012 == 2).astype(float)), "m") for s in styles}
    j0p2 = mat(j0, "p2_o012")
    jmm = {"M0": mat(j0.assign(m=(j0.argmax_o012 == 2).astype(float)), "m"), "M1": jarg["A_plain"],
           "M2_majority": (np.sum([jarg[s] for s in styles], 0) >= 2).astype(float)}
    rate0 = float(np.nanmean(jmm["M0"]))
    p2avg = np.mean([jp2[s] for s in styles], 0)
    t = matched_threshold(p2avg, rate0)
    jmm["M2_mean_p2_threshold_matched_to_M0"] = (p2avg >= t).astype(float)

    def rows_ok(X):
        return ~np.isnan(X).any(1)

    out: Dict[str, Any] = {"jev_M2_threshold": t}
    for sysn, arms in (("gpt", gm), ("jev", jmm)):
        keep = np.ones(len(main), bool)
        for X in arms.values():
            keep &= rows_ok(X)
        out[sysn] = {k: summarise_rule(X[keep], personas) for k, X in arms.items()}
        names = list(arms)
        out[sysn]["paired_diff_p_differ"] = {f"M0_minus_{k}": paired_diff(arms["M0"][keep], arms[k][keep]) for k in names[1:]}
        out[sysn]["n_complaints"] = int(keep.sum())
    # fact responsiveness on monetary decisions under each arm
    pcd = pd.read_csv("holdout_out/positive_controls/decisions.csv")
    pcd = pcd[(pcd.model == BANK) & pcd.case_id.isin(pcs)]
    dpc = dm[dm.case_id.isin(pcs) & dm.persona_label.isin(["strengthen", "weaken"])]

    def span_from(frame, val_fn, styles_=None, combine="single"):
        res = {}
        for e in ("strengthen", "weaken"):
            f = frame[frame.persona_label == e] if "persona_label" in frame else frame[frame.edit == e]
            if styles_ is None:
                res[e] = f.set_index("case_id").reindex(pcs).pipe(val_fn).to_numpy(float)
            else:
                mats = [f[f["style"] == s].set_index("case_id").reindex(pcs).pipe(val_fn).to_numpy(float) for s in styles_]
                res[e] = (np.sum(mats, 0) >= 2).astype(float) if combine == "majority" else mats[0]
        ok = ~np.isnan(res["strengthen"]) & ~np.isnan(res["weaken"])
        return fact_span({k: v[ok] for k, v in res.items()})

    mon = lambda f: (f.tier == 2).astype(float)  # noqa: E731
    out["fact_span_monetary_gpt"] = {"M0": span_from(pcd, mon), "M1": span_from(dpc, mon, ["A_plain"]),
                                     "M2_majority": span_from(dpc, mon, styles, "majority")}
    return out


def main():
    res = {"label": "POST HOC (responding to the 21 Oct review); no new API calls",
           "R1_variance_components_jev": r1(), "R2_remedy_rules_holdout": r2(), "R3_mitigation_remedy_rules": r3()}
    Path("prereg/revision_analysis_results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    v = res["R1_variance_components_jev"]
    for name in ("p_monetary", "e_tier"):
        print(name, {k: round(x["estimate"], 4) for k, x in v[name].items() if k.startswith("share")})


if __name__ == "__main__":
    main()
