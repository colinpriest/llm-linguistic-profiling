"""Frozen confirmatory test of H6 (Jev vs GPT-4o-mini) on the extended 4,000-complaint holdout.

Pre-registered in prereg/amendment_1_extension.md before any of complaints 1,001-4,000 were
generated.  Two-look group-sequential design, Lan-DeMets O'Brien-Fleming spending:
  look 1: n = 1,000 (information fraction 0.25), spent alpha 8.86e-05, not crossed
  look 2: n = 4,000, nominal two-sided alpha 0.04996  ->  99.95..% replaced by a 95.004% CI

    python prereg/extension_h6.py

Output: holdout_out/extension_h6_results.json and .md.  H1-H5 are recomputed on n = 4,000 as
secondary, descriptive updates; their confirmatory results remain those recorded at n = 1,000.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import confirmatory_analysis as ca  # noqa: E402

TAG = "n4000"
EXPECTED_COMPLAINTS = 4000
ALPHA_FINAL = 0.04996
N_BOOT_H6 = 5000
SEED = 20261020


def h6_final(P, G, C):
    rng = np.random.default_rng(SEED)
    out = {}
    for b in ca.BANKS:
        g = G[b]
        gm = (g == 2).astype(float)
        gm[np.isnan(g)] = np.nan
        p2 = P[f"{b}|{ca.PRIMARY_ORDER}"][..., 2]
        rate = float(np.nanmean(gm))
        jm = (p2 >= np.quantile(p2, 1 - rate)).astype(float)
        diff = lambda ix: ca.D(gm[ix]) - ca.D(jm[ix])
        vals = np.array([diff(rng.integers(0, C, C)) for _ in range(N_BOOT_H6)])
        lo, hi = np.nanquantile(vals, [ALPHA_FINAL / 2, 1 - ALPHA_FINAL / 2])
        out[b] = {"gpt_monetary_rate": rate, "jev_monetary_rate": float(jm.mean()),
                  "D_gpt": ca.D(gm), "D_jev": ca.D(jm), "delta_gpt_minus_jev": diff(np.arange(C)),
                  "ci_level": 1 - ALPHA_FINAL, "ci": [float(lo), float(hi)], "jev_lower": bool(lo > 0)}
    out["confirmatory_test"] = "gpt-4o-mini only"
    out["supported"] = out["gpt-4o-mini"]["jev_lower"]
    out["note"] = "gpt-5-mini comparison reported descriptively (amendment 1, section 3)"
    return out


def main() -> None:
    P, G, cases, base = ca.load_panel("holdout", TAG)
    C = len(cases)
    res = {"tag": TAG, "n_complaints_complete": C, "expected": EXPECTED_COMPLAINTS,
           "orders": list(P["_orders"]), "alpha_final": ALPHA_FINAL,
           "H6_final": h6_final(P, G, C),
           "secondary_updates_n4000": {"H1": ca.h1(P, C), "H2": ca.h2(P, C), "H3": ca.h3(P, C),
                                       "H4": ca.h4(P, C), "H5": ca.h5(P, C)}}
    (base / "extension_h6_results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    md = ["# Extension: final H6 test (n = 4,000)", "",
          f"Complete complaints: {C} (expected {EXPECTED_COMPLAINTS}); alpha_final {ALPHA_FINAL}", "",
          "```", json.dumps(res["H6_final"], indent=2, default=float), "```", "",
          "## Secondary updates (descriptive)", "```",
          json.dumps(res["secondary_updates_n4000"], indent=2, default=float), "```"]
    (base / "extension_h6_results.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
