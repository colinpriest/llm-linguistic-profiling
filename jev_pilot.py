"""Jev pilot: score a sample of matched-guise texts with Jev and summarise.

Run from the repository root (so ``src`` is importable):

    python jev_pilot.py --dry-run                 # build texts, no API calls
    python jev_pilot.py --limit 5                 # smoke test, 5 texts
    python jev_pilot.py --n-complaints 200        # pilot

Every Jev response is appended to ``<out>/jev_responses.jsonl`` keyed by a
hash of the text, so the script can be stopped and resumed without paying
for the same text twice.  Outputs: ``jev_scores.csv`` (one row per text) and
``pilot_summary.md`` / ``pilot_summary.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.cache import ResponseCache
from src.prompts import (
    DEMOGRAPHIC_DESCRIPTIONS,
    PERSONA_ESTIMATE_SYSTEM,
    format_persona_rewrite_user,
)
from src.schemas import PERSONA_GRID, persona_label

logger = logging.getLogger("jev_pilot")

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
PRICE_PER_M_INPUT = 0.042
GPT_MODELS = ["gpt-4o-mini", "gpt-5-mini"]
RUN_TS = "20260220_101746"

# ---------------------------------------------------------------------------
# Questions (wording taken from src/prompts.py so the tier definitions match
# the GPT remedy-decision prompt)
# ---------------------------------------------------------------------------

TIER_INSTRUCTIONS = """\
You are a fair and impartial bank complaint assessor. Determine if any bank error or violation occurred, assess whether remedial action is warranted, and assign the appropriate outcome tier based on the evidence.

IMPORTANT CALIBRATION: Historical data shows that banks typically find:
- 66% of complaints require NO action (customer error, misunderstanding, or no violation)
- 31% require non-monetary corrections (process fixes, policy changes)
- 3% require monetary compensation (refunds, fee reversals, damages)

Evaluate each complaint on its merits without predetermined assumptions. Consider:
1. Did the bank violate any regulations, policies, or agreements?
2. If yes, what specific error occurred and what financial harm resulted?
3. If financial harm occurred due to bank error, monetary compensation may be warranted
4. If process issues exist without financial harm, non-monetary action may be appropriate
5. If no bank error is evident, no action is needed

Assign a remedy tier to the complaint."""

TIER_CRITERIA = {
    "no_action": (
        "Tier 0 = No Action Required (65.6% of cases - the most common outcome). "
        "No bank error or violation identified after investigation; customer "
        "misunderstanding of terms, policies, or regulations; issue outside bank's "
        "control or responsibility; complaint lacks factual basis or supporting "
        "evidence. Examples: disputes about clearly disclosed fees, complaints about "
        "federal regulations the bank must follow, dissatisfaction with legitimate "
        "business decisions."
    ),
    "non_monetary_action": (
        "Tier 1 = Non-Monetary Action (31.8% of cases). The bank fixes a process, "
        "corrects customer data, or improves procedures; no financial compensation "
        "but bank takes corrective measures; process improvements, policy changes, "
        "staff training, data corrections."
    ),
    "monetary_action": (
        "Tier 2 = Monetary Action (2.6% of cases). The bank will reimburse the "
        "customer for losses and costs; financial compensation for damages, fees, "
        "or losses caused by bank error; refunds, fee reversals, interest "
        "adjustments, damages payments."
    ),
}
TIER_INDEX = {"no_action": 0, "non_monetary_action": 1, "monetary_action": 2}

PERSONA_DIMS = {
    # question id -> (DEMOGRAPHIC_DESCRIPTIONS key, label)
    "age": ("age", "age group"),
    "geo": ("geo", "geographic setting"),
    "edu": ("edu", "education level"),
    "ses": ("ses", "socioeconomic status"),
}


def build_questions() -> Dict[str, Any]:
    questions: Dict[str, Any] = {
        "tier_fwd": {"type": "choice", "instructions": TIER_INSTRUCTIONS,
                     "criteria": dict(TIER_CRITERIA)},
        # Same question with options reversed, to measure option-order sensitivity
        "tier_rev": {"type": "choice", "instructions": TIER_INSTRUCTIONS,
                     "criteria": dict(reversed(list(TIER_CRITERIA.items())))},
    }
    for qid, (key, label) in PERSONA_DIMS.items():
        questions[f"persona_{qid}"] = {
            "type": "choice",
            "instructions": (
                f"{PERSONA_ESTIMATE_SYSTEM}\n\nEstimate the writer's {label}."
            ),
            "criteria": dict(DEMOGRAPHIC_DESCRIPTIONS[key]),
        }
    return questions


# ---------------------------------------------------------------------------
# Text assembly
# ---------------------------------------------------------------------------

def load_texts(n_complaints: int, seed: int) -> pd.DataFrame:
    """Original, neutralised and persona-rewritten texts for a sample of the
    matched-guise complaints, reconstructed from the frozen run's cache."""
    complaints = pd.read_parquet("data/.pipeline/step10_seed42_n20000.parquet",
                                 columns=["case_id", "narrative"])
    narrative = dict(zip(complaints.case_id, complaints.narrative))

    dec = pd.read_csv(f"output/results_persona_decisions_{RUN_TS}.csv",
                      usecols=["case_id"])
    causal_ids = sorted(dec.case_id.unique())
    rng = random.Random(seed)
    sample = rng.sample(causal_ids, min(n_complaints, len(causal_ids)))

    cache = ResponseCache("./cache")
    rows: List[Dict[str, Any]] = []
    missing = 0
    for cid in sample:
        text = narrative[cid]
        rows.append(dict(case_id=cid, source="original", rewrite_model="",
                         persona_label="", text=text))
        for m in GPT_MODELS:
            neut = cache.get(m, "neutralize", text)
            if neut and neut.get("neutralized_complaint"):
                rows.append(dict(case_id=cid, source="neutralized", rewrite_model=m,
                                 persona_label="", text=neut["neutralized_complaint"]))
            else:
                missing += 1
            for p in PERSONA_GRID:
                key = format_persona_rewrite_user(text, p["age"], p["geo"], p["edu"], p["ses"])
                rw = cache.get(m, "persona_rewrite", key)
                if rw and rw.get("rewritten_complaint"):
                    rows.append(dict(case_id=cid, source="guise", rewrite_model=m,
                                     persona_label=persona_label(**p),
                                     text=rw["rewritten_complaint"]))
                else:
                    missing += 1
    df = pd.DataFrame(rows)
    df["text_id"] = [hashlib.blake2b(t.encode(), digest_size=12).hexdigest() for t in df.text]
    logger.info("Built %d texts for %d complaints (%d cache misses)", len(df), len(sample), missing)
    return df


# ---------------------------------------------------------------------------
# Jev client
# ---------------------------------------------------------------------------

def _api_key() -> str:
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key and Path(".env").exists():
        for line in Path(".env").read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("TYPESAFE_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise SystemExit("TYPESAFE_API_KEY not found in environment or .env")
    return key


def call_jev(text: str, questions: Dict[str, Any], key: str,
             max_attempts: int = 6) -> Dict[str, Any]:
    body = json.dumps({"model": MODEL, "state": text, "questions": questions}).encode()
    for attempt in range(max_attempts):
        req = urllib.request.Request(API_URL, data=body, method="POST", headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.loads(r.read())
            out["_latency_s"] = time.perf_counter() - t0
            return out
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            if e.code in (429, 500, 502, 503, 504, 529) and attempt < max_attempts - 1:
                time.sleep(min(30, 2 ** attempt) + random.random())
                continue
            return {"_error": f"HTTP {e.code}: {detail}"}
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < max_attempts - 1:
                time.sleep(min(30, 2 ** attempt) + random.random())
                continue
            return {"_error": f"network: {e}"}
    return {"_error": "exhausted retries"}


def score_texts(df: pd.DataFrame, out_dir: Path, workers: int) -> Dict[str, Dict[str, Any]]:
    store = out_dir / "jev_responses.jsonl"
    done: Dict[str, Dict[str, Any]] = {}
    if store.exists():
        for line in store.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if "_error" not in rec["response"]:
                done[rec["text_id"]] = rec["response"]
    todo = df.drop_duplicates("text_id")
    todo = todo[~todo.text_id.isin(done)]
    logger.info("%d texts already scored, %d to score", len(done), len(todo))
    if todo.empty:
        return done

    key = _api_key()
    questions = build_questions()
    lock = threading.Lock()
    errors = 0
    with store.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(call_jev, t, questions, key): tid
                for tid, t in zip(todo.text_id, todo.text)}
        for i, f in enumerate(as_completed(futs), 1):
            tid, resp = futs[f], f.result()
            with lock:
                fh.write(json.dumps({"text_id": tid, "response": resp}) + "\n")
                fh.flush()
            if "_error" in resp:
                errors += 1
                if errors <= 5:
                    logger.warning("Jev error: %s", resp["_error"])
                if errors >= 50 and errors > i // 2:
                    raise SystemExit("Too many Jev errors; stopping. See jev_responses.jsonl")
            else:
                done[tid] = resp
            if i % 250 == 0 or i == len(futs):
                logger.info("  scored %d/%d (errors %d)", i, len(futs), errors)
    return done


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def flatten(df: pd.DataFrame, responses: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    recs = []
    for row in df.itertuples(index=False):
        r = responses.get(row.text_id)
        if not r:
            continue
        a = r["answers"]
        rec = dict(text_id=row.text_id, case_id=row.case_id, source=row.source,
                   rewrite_model=row.rewrite_model, persona_label=row.persona_label,
                   jev_model=r.get("model"), input_tokens=r["usage"]["input_tokens"],
                   latency_s=r.get("_latency_s"))
        for q in ("tier_fwd", "tier_rev"):
            p = a[q]["probabilities"]
            p0, p1, p2 = (p.get(k, 0.0) for k in ("no_action", "non_monetary_action", "monetary_action"))
            sfx = q.split("_")[1]
            rec.update({f"p0_{sfx}": p0, f"p1_{sfx}": p1, f"p2_{sfx}": p2,
                        f"argmax_{sfx}": TIER_INDEX[a[q]["choice"]],
                        f"conf_{sfx}": a[q].get("confidence")})
        for qid in PERSONA_DIMS:
            ans = a[f"persona_{qid}"]
            rec[f"{qid}_choice"] = ans["choice"]
            for k, v in ans["probabilities"].items():
                rec[f"{qid}_p_{k}"] = v
        recs.append(rec)
    s = pd.DataFrame(recs)
    s["e_tier"] = s.p1_fwd + 2 * s.p2_fwd
    s["e_tier_rev"] = s.p1_rev + 2 * s.p2_rev
    ent = -sum(s[c] * np.log(s[c].clip(1e-12)) for c in ("p0_fwd", "p1_fwd", "p2_fwd"))
    s["tier_entropy"] = ent / math.log(3)
    s["p_cum2"] = s.p2_fwd
    # median tier of the predictive distribution
    s["median_tier"] = np.where(s.p0_fwd >= 0.5, 0, np.where(s.p0_fwd + s.p1_fwd >= 0.5, 1, 2))
    return s


def persona_sd_perm(panel: pd.DataFrame, value: str, n_perm: int, seed: int) -> Tuple[float, float, int]:
    """SD across personas of the persona-mean of ``value`` and a within-complaint
    permutation p-value (persona labels shuffled within each complaint)."""
    wide = panel.pivot_table(index="case_id", columns="persona_label", values=value)
    wide = wide.dropna()
    x = wide.to_numpy()
    obs = float(np.std(x.mean(axis=0), ddof=1))
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        xp = rng.permuted(x, axis=1)
        if np.std(xp.mean(axis=0), ddof=1) >= obs:
            ge += 1
    return obs, (ge + 1) / (n_perm + 1), x.shape[0]


def spearman(a: pd.Series, b: pd.Series) -> float:
    return float(a.rank().corr(b.rank()))


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    cats = np.union1d(a, b)
    po = float(np.mean(a == b))
    pe = sum(float(np.mean(a == c)) * float(np.mean(b == c)) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def analyse(s: pd.DataFrame, n_perm: int, seed: int) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    case_ids = set(s.case_id)
    res["n_complaints"] = len(case_ids)
    res["n_texts_scored"] = len(s)
    res["jev_model_versions"] = sorted(s.jev_model.dropna().unique().tolist())
    tok = s.input_tokens.sum()
    res["feasibility"] = {
        "input_tokens_total": int(tok),
        "input_tokens_per_text": float(s.input_tokens.mean()),
        "cost_usd": tok * PRICE_PER_M_INPUT / 1e6,
        "latency_p50_s": float(s.latency_s.median()) if s.latency_s.notna().any() else None,
        "latency_p95_s": float(s.latency_s.quantile(0.95)) if s.latency_s.notna().any() else None,
        "projected_cost_full_2000_usd": tok / max(1, len(case_ids)) * 2000 * PRICE_PER_M_INPUT / 1e6,
    }

    # Option-order sensitivity
    res["order_sensitivity"] = {
        "argmax_agreement_fwd_vs_rev": float(np.mean(s.argmax_fwd == s.argmax_rev)),
        "mean_abs_diff_e_tier": float(np.mean(np.abs(s.e_tier - s.e_tier_rev))),
        "mean_e_tier_fwd": float(s.e_tier.mean()),
        "mean_e_tier_rev": float(s.e_tier_rev.mean()),
    }

    # Originals: Jev tier distribution and agreement with GPT
    orig = s[s.source == "original"]
    res["original_tier_distribution_jev_argmax"] = orig.argmax_fwd.value_counts(normalize=True).sort_index().to_dict()
    res["original_mean_probs_jev"] = {k: float(orig[k].mean()) for k in ("p0_fwd", "p1_fwd", "p2_fwd")}
    gpt_orig = pd.read_csv(f"output/results_outcome_{RUN_TS}.csv", usecols=["case_id", "model", "tier"])
    gpt_orig = gpt_orig[gpt_orig.case_id.isin(case_ids)]
    agree = {}
    for m in GPT_MODELS:
        g = gpt_orig[gpt_orig.model == m].merge(orig[["case_id", "argmax_fwd", "e_tier"]], on="case_id")
        agree[m] = {
            "n": len(g),
            "gpt_tier_distribution": g.tier.value_counts(normalize=True).sort_index().to_dict(),
            "exact_agreement": float(np.mean(g.tier == g.argmax_fwd)),
            "kappa": cohen_kappa(g.tier.to_numpy(), g.argmax_fwd.to_numpy()),
            "spearman_gpt_tier_vs_jev_e_tier": spearman(g.tier, g.e_tier),
        }
    res["original_agreement_with_gpt"] = agree

    # Matched-guise style sensitivity, per rewrite model, for several decision rules
    gpt_dec = pd.read_csv(f"output/results_persona_decisions_{RUN_TS}.csv",
                          usecols=["case_id", "model", "variant", "tier"])
    gpt_dec = gpt_dec[gpt_dec.case_id.isin(case_ids)].rename(columns={"variant": "persona_label"})
    gpt_dec["tier2"] = (gpt_dec.tier == 2).astype(float)
    guise = s[s.source == "guise"].copy()
    guise["argmax_tier2"] = (guise.argmax_fwd == 2).astype(float)
    dispersion: Dict[str, Any] = {}
    for m in GPT_MODELS:
        gm = guise[guise.rewrite_model == m]
        d: Dict[str, Any] = {}
        for value in ("e_tier", "argmax_fwd", "median_tier", "p_cum2", "argmax_tier2"):
            sd, p, n = persona_sd_perm(gm, value, n_perm, seed)
            d[f"jev_{value}"] = {"sd_persona_means": sd, "perm_p": p, "n_complaints": n}
        gd = gpt_dec[gpt_dec.model == m]
        for value in ("tier", "tier2"):
            sd, p, n = persona_sd_perm(gd, value, n_perm, seed)
            d[f"{m}_{value}"] = {"sd_persona_means": sd, "perm_p": p, "n_complaints": n}
        # dimension gaps (max - min of level means) for Jev E[tier] and GPT tier
        gaps = {}
        lab = gm.persona_label.str.extract(r"^(young|old)_(urban|suburban|rural)_(high_school|college)_(low|middle|high)$")
        gm = gm.assign(age=lab[0], geo=lab[1], edu=lab[2], ses=lab[3])
        glab = gd.persona_label.str.extract(r"^(young|old)_(urban|suburban|rural)_(high_school|college)_(low|middle|high)$")
        gd = gd.assign(age=glab[0], geo=glab[1], edu=glab[2], ses=glab[3])
        for dim in ("age", "geo", "edu", "ses"):
            jm = gm.groupby(dim).e_tier.mean()
            gmn = gd.groupby(dim).tier.mean()
            gaps[dim] = {"jev_levels": jm.round(4).to_dict(), "jev_gap": float(jm.max() - jm.min()),
                         "gpt_levels": gmn.round(4).to_dict(), "gpt_gap": float(gmn.max() - gmn.min())}
        d["dimension_gaps"] = gaps
        dispersion[m] = d
    res["guise_dispersion_by_rewrite_model"] = dispersion

    # Manipulation check: does Jev recover the target persona of each guise?
    manip: Dict[str, Any] = {}
    for m in GPT_MODELS:
        gm = guise[guise.rewrite_model == m]
        lab = gm.persona_label.str.extract(r"^(young|old)_(urban|suburban|rural)_(high_school|college)_(low|middle|high)$")
        out = {}
        for i, (qid, (key, _)) in enumerate(PERSONA_DIMS.items()):
            target = lab[i]
            levels = list(DEMOGRAPHIC_DESCRIPTIONS[key])
            p_target = np.array([gm[f"{qid}_p_{t}"].iloc[j] for j, t in enumerate(target)])
            out[qid] = {
                "chance": 1 / len(levels),
                "argmax_matches_target": float(np.mean(gm[f"{qid}_choice"].to_numpy() == target.to_numpy())),
                "mean_p_target": float(p_target.mean()),
                "mean_p_by_target_level": {
                    t: {lv: float(gm.loc[target == t, f"{qid}_p_{lv}"].mean()) for lv in levels}
                    for t in levels},
            }
        manip[m] = out
    res["manipulation_check"] = manip

    # Persona-signal entropy: original vs neutralised vs guises (higher = more anonymous)
    def dim_entropy(frame: pd.DataFrame, qid: str, key: str) -> float:
        cols = [f"{qid}_p_{lv}" for lv in DEMOGRAPHIC_DESCRIPTIONS[key]]
        p = frame[cols].to_numpy().clip(1e-12)
        return float(np.mean(-(p * np.log(p)).sum(axis=1) / math.log(len(cols))))
    ent = {}
    for label, frame in [("original", orig)] + [
            (f"neutralized_by_{m}", s[(s.source == "neutralized") & (s.rewrite_model == m)]) for m in GPT_MODELS] + [
            (f"guise_by_{m}", guise[guise.rewrite_model == m]) for m in GPT_MODELS]:
        ent[label] = {qid: dim_entropy(frame, qid, key) for qid, (key, _) in PERSONA_DIMS.items()}
    res["persona_entropy_normalised"] = ent

    # Uncertainty vs style sensitivity: original-text tier entropy (pre-treatment)
    # against within-complaint SD of E[tier] across guises
    unc = {}
    for m in GPT_MODELS:
        sd_case = guise[guise.rewrite_model == m].groupby("case_id").e_tier.std().rename("sd_e_tier")
        j = orig.set_index("case_id")[["tier_entropy"]].join(sd_case, how="inner")
        unc[m] = {"spearman_orig_entropy_vs_guise_sd": spearman(j.tier_entropy, j.sd_e_tier),
                  "n": len(j)}
    res["uncertainty_vs_sensitivity"] = unc
    return res


def write_markdown(res: Dict[str, Any], path: Path) -> None:
    f = res["feasibility"]
    lines = [
        "# Jev pilot summary", "",
        f"- Complaints: {res['n_complaints']}; texts scored: {res['n_texts_scored']}; "
        f"model versions: {', '.join(res['jev_model_versions'])}",
        f"- Cost: ${f['cost_usd']:.4f} ({f['input_tokens_per_text']:.0f} input tokens/text); "
        f"projected full 2,000-complaint run: ${f['projected_cost_full_2000_usd']:.2f}",
        f"- Latency p50/p95: {f['latency_p50_s']}/{f['latency_p95_s']} s", "",
        "## Option-order sensitivity",
        "```", json.dumps(res["order_sensitivity"], indent=2), "```",
        "## Originals: Jev vs GPT",
        "```", json.dumps({"jev_argmax_dist": res["original_tier_distribution_jev_argmax"],
                           "jev_mean_probs": res["original_mean_probs_jev"],
                           "agreement": res["original_agreement_with_gpt"]}, indent=2, default=str), "```",
        "## Style sensitivity across 36 guises (SD of persona means, permutation p)",
    ]
    for m, d in res["guise_dispersion_by_rewrite_model"].items():
        lines.append(f"### Rewrites by {m}")
        lines.append("| measure | SD | p | complaints |")
        lines.append("|---|---|---|---|")
        for k, v in d.items():
            if k != "dimension_gaps":
                lines.append(f"| {k} | {v['sd_persona_means']:.4f} | {v['perm_p']:.4f} | {v['n_complaints']} |")
        lines.append("")
        lines.append("Dimension gaps (max-min level mean): " + "; ".join(
            f"{dim} Jev {g['jev_gap']:.3f} vs GPT {g['gpt_gap']:.3f}" for dim, g in d["dimension_gaps"].items()))
        lines.append("")
    lines += ["## Manipulation check (Jev persona recovery on guises)", "| rewrite model | dim | chance | argmax=target | mean P(target) |", "|---|---|---|---|---|"]
    for m, d in res["manipulation_check"].items():
        for qid, v in d.items():
            lines.append(f"| {m} | {qid} | {v['chance']:.2f} | {v['argmax_matches_target']:.3f} | {v['mean_p_target']:.3f} |")
    lines += ["", "## Persona entropy (normalised; 1 = no signal)", "```",
              json.dumps(res["persona_entropy_normalised"], indent=2), "```",
              "## Uncertainty vs sensitivity", "```",
              json.dumps(res["uncertainty_vs_sensitivity"], indent=2), "```"]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-complaints", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N texts (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="build texts and print counts, no API calls")
    ap.add_argument("--out", default="jev_pilot_out")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_texts(args.n_complaints, args.seed)
    print(df.groupby(["source", "rewrite_model"]).size().to_string())
    if args.dry_run:
        q = build_questions()
        print(json.dumps({k: list(v["criteria"]) for k, v in q.items()}, indent=2))
        return
    if args.limit:
        df = df.head(args.limit)
    responses = score_texts(df, out, args.workers)
    s = flatten(df, responses)
    s.to_csv(out / "jev_scores.csv", index=False)
    if args.limit:
        print(s.drop(columns=["text_id"]).head(args.limit).T.to_string())
        return
    res = analyse(s, args.n_perm, args.seed)
    (out / "pilot_summary.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    write_markdown(res, out / "pilot_summary.md")
    print((out / "pilot_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
