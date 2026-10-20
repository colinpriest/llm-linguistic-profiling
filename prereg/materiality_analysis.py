"""Post hoc materiality (effect-size) analysis of monetary-remedy disparities.

NOT pre-registered: added 20 Oct 2026 after the n = 1,000 confirmatory run, at the
author's request, to assess whether disparities are material.  Reported as such.

For each system (GPT-4o-mini, GPT-5-mini, Jev with p2 threshold matched to each GPT
model's monetary volume) and persona dimension:
  * paired within-complaint monetary-rate gap (pp) and adverse impact ratio (AIR),
  * Cohen's h, with complaint-bootstrap CIs,
  * equivalence (non-materiality) tests: AIR > 0.80 (four-fifths rule) and |h| < 0.20
    ("small"), each one-sided at alpha 0.05 (90% two-sided CI inside the bound).
Plus individual-level arbitrariness: probability that two random guises of the same
complaint receive different monetary decisions.

    python prereg/materiality_analysis.py            # holdout (n=1,000) and pooled (n=3,000)
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).parent))
import confirmatory_analysis as ca  # noqa: E402

N_BOOT = 2000
rng = np.random.default_rng(20261020)
CONTRASTS = {"age": ("old", "young"), "edu": ("college", "high_school"), "ses": ("high", "low"),
             "geo": ("urban", "rural")}


def mats(P, G, b):
    g = G[b]
    gm = (g == 2).astype(float)
    gm[np.isnan(g)] = np.nan
    p2 = P[f"{b}|{ca.PRIMARY_ORDER}"][..., 2]
    jm = (p2 >= np.quantile(p2, 1 - np.nanmean(gm))).astype(float)
    return gm, jm


def cohen_h(p1, p2):
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def contrast_stats(X):
    X = X[~np.isnan(X).any(1)]
    C = len(X)
    out = {}
    for d, (hi, lo) in CONTRASTS.items():
        lv = ca.DIM_OF[d]
        rh, rl = X[:, lv == hi].mean(1), X[:, lv == lo].mean(1)       # per-complaint rates
        def stat(ix):
            a, b = rh[ix].mean(), rl[ix].mean()
            return a - b, b / a, cohen_h(a, b)   # AIR directional: rate(lower-predicted level) / rate(higher-predicted level)
        est = stat(np.arange(C))
        bs = np.array([stat(rng.integers(0, C, C)) for _ in range(N_BOOT)])
        q = lambda j, lo_, hi_: [float(np.quantile(bs[:, j], lo_)), float(np.quantile(bs[:, j], hi_))]
        out[d] = {"contrast": f"{hi}-{lo}", "rate_hi": float(rh.mean()), "rate_lo": float(rl.mean()),
                  "gap_pp": 100 * est[0], "gap_pp_ci95": [100 * v for v in q(0, .025, .975)],
                  "AIR": est[1], "AIR_ci90": q(1, .05, .95), "cohen_h": est[2], "h_ci90": q(2, .05, .95),
                  "non_material_AIR_gt_0.80": bool(np.quantile(bs[:, 1], .05) > 0.80),
                  "non_material_AIR_gt_0.90": bool(np.quantile(bs[:, 1], .05) > 0.90),
                  "non_material_abs_h_lt_0.20": bool(np.quantile(bs[:, 2], .05) > -0.20 and np.quantile(bs[:, 2], .95) < 0.20)}
    # individual-level arbitrariness: P(two random distinct guises disagree on monetary decision)
    m = X.mean(1)
    K = X.shape[1]
    pair_disagree = (2 * m * (1 - m) * K / (K - 1)).mean()
    out["arbitrariness"] = {"p_two_guises_disagree_monetary": float(pair_disagree),
                            "share_complaints_monetary_not_unanimous": float(((m > 0) & (m < 1)).mean()),
                            "monetary_rate": float(X.mean())}
    return out


def main():
    Pe, Ge, _, _ = ca.load_panel("exploratory", "")
    Ph, Gh, _, _ = ca.load_panel("holdout", "n1000")
    res = {}
    for b in ca.BANKS:
        ge, je = mats(Pe, Ge, b)
        gh, jh = mats(Ph, Gh, b)
        for label, (g, j) in {"holdout_n1000": (gh, jh), "pooled_n3000": (np.vstack([ge, gh]), np.vstack([je, jh]))}.items():
            res.setdefault(label, {})[b] = {"GPT": contrast_stats(g), "Jev_matched": contrast_stats(j)}
    out = Path("prereg/materiality_results.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    for label, r in res.items():
        print(f"\n### {label}")
        for b, sysd in r.items():
            for sname, st in sysd.items():
                a = st["arbitrariness"]
                print(f"{b} / {sname}: monetary {100*a['monetary_rate']:.1f}%; P(2 guises disagree) {100*a['p_two_guises_disagree_monetary']:.1f}%; complaints not unanimous {100*a['share_complaints_monetary_not_unanimous']:.1f}%")
                for d in CONTRASTS:
                    s = st[d]
                    print(f"   {s['contrast']:22s} gap {s['gap_pp']:+.2f}pp [{s['gap_pp_ci95'][0]:+.2f},{s['gap_pp_ci95'][1]:+.2f}]  AIR {s['AIR']:.3f} (90%CI {s['AIR_ci90'][0]:.3f}-{s['AIR_ci90'][1]:.3f})  h {s['cohen_h']:+.3f} [{s['h_ci90'][0]:+.3f},{s['h_ci90'][1]:+.3f}]  immaterial: AIR>.8 {s['non_material_AIR_gt_0.80']}, AIR>.9 {s['non_material_AIR_gt_0.90']}, |h|<.2 {s['non_material_abs_h_lt_0.20']}")


if __name__ == "__main__":
    main()
