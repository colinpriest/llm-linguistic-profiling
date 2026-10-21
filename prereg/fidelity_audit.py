"""Independent fact-fidelity audit and same-day baseline (prereg/fidelity_audit.md, pre-registration 5).

Answers the review's points 2 and 8:
  * Do the persona rewrites (and the standardising rewrites) preserve the facts, judged by two auditors from
    outside the audited systems plus exact code checks?  Does individual decision instability persist among
    guises verified as fact-preserving?
  * Did GPT-4o-mini or Jev drift between the 19 Oct M0 decisions and the 21 Oct M1/M2 decisions?

    python prereg/fidelity_audit.py sample            # draw the audit sample (no API calls)
    python prereg/fidelity_audit.py audit  [--smoke]  # two LLM auditors (resumable; cost-capped)
    python prereg/fidelity_audit.py baseline [--smoke]  # fresh GPT-4o-mini decisions + Jev scores, M0 and M1 interleaved
    python prereg/fidelity_audit.py analyse            # frozen analysis -> prereg/fidelity_audit_results.json
--smoke uses holdout complaints 501-505 (outside every analysis set) and writes to a separate folder.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, ".")
import holdout_panel as hp  # noqa: E402
import jev_arms as jarms  # noqa: E402
import jev_pilot as jp  # noqa: E402
from src.cache import ResponseCache  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402

logger = logging.getLogger("fidelity_audit")
SEED = 20261022
N_BOOT = 2000
BANK = "gpt-4o-mini"
N_MAIN = 500                       # mitigation analysis set (first 500 holdout complaints)
N_GUISE_PER_STRATUM = 20           # complaints per stratum, all 36 guises audited
N_STD = {"unchanged": 100, "changed": 50}
N_EDIT = {"strengthen": 50, "weaken": 50}
N_BASELINE = 100
COST_CAP_AUDIT = 45.0              # USD; the auditor step stops if its running cost reaches this
PRICES = {"claude-opus-5-5": (4.00, 20.00), "gpt-5.2": (1.75, 14.00)}   # USD per 1M input / output tokens
AUDITORS = ["claude-opus-5-5", "gpt-5.2"]

AUDIT_SYSTEM = (
    "You audit whether a rewritten US consumer financial complaint preserves the FACTS of the original. "
    "TEXT A is the original complaint; TEXT B is a rewrite that was instructed to change only the writing style. "
    "List every difference in factual content between A and B. Factual content means: amounts of money, dates and "
    "times, the order in which events happened, which party did what and who is responsible, whether something "
    "did or did not happen (negation), how certain or how strong a claim is (for example 'the bank made an error' "
    "versus 'I think the bank may have made an error'), what the writer asks for, and any event, action, document, "
    "account or institution that is added, omitted or changed. "
    "IGNORE differences in style: vocabulary, grammar, spelling, tone, politeness, formality, emotion, length, "
    "paragraphing, and reordering that does not change what happened when. Redaction tokens such as XXXX or "
    "XX/XX/XXXX are placeholders; ignore whether they are kept, but do compare any real values next to them. "
    "Mark a difference as material if a careful complaints handler choosing between no action, non-monetary action "
    "and monetary relief could reasonably decide differently because of it. "
    "Overall verdict: 'preserved' if there are no factual differences, 'minor' if all differences are non-material, "
    "'material' if at least one difference is material."
)
CATEGORIES = ["amount", "date_time", "chronology", "responsibility", "negation", "certainty",
              "requested_relief", "event_or_entity", "other"]
AUDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "discrepancies": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": CATEGORIES},
                "kind": {"type": "string", "enum": ["added", "omitted", "changed"]},
                "material": {"type": "boolean"},
                "description": {"type": "string"},
            },
            "required": ["category", "kind", "material", "description"], "additionalProperties": False}},
        "overall": {"type": "string", "enum": ["preserved", "minor", "material"]},
    },
    "required": ["discrepancies", "overall"], "additionalProperties": False,
}


def out_dir(smoke: bool) -> Path:
    return Path("holdout_out/fidelity_audit_smoke" if smoke else "holdout_out/fidelity_audit")


def holdout_ids() -> List[str]:
    return pd.read_csv("holdout_out/holdout_sample_full.csv").case_id.tolist()


# ---------------------------------------------------------------------------
# Sampling (deterministic; no API calls)
# ---------------------------------------------------------------------------

def cmd_sample(smoke: bool) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    ids = holdout_ids()
    orig = pd.read_csv("holdout_out/holdout_sample_full.csv").set_index("case_id").narrative
    rw = pd.read_csv("holdout_out/rewrites_n1000.csv")
    rw = rw[(rw.rewrite_model == BANK) & rw.text.notna()]
    d0 = pd.read_csv("holdout_out/decisions_n1000.csv")
    d0 = d0[d0.model == BANK]
    rows: List[Dict[str, Any]] = []

    if smoke:
        main = ids[500:505]
        for cid in main[:2]:
            for r in rw[rw.case_id == cid].head(3).itertuples():
                rows.append(dict(set="guise", stratum="smoke", weight=1.0, case_id=cid, label=r.persona_label,
                                 text_a=orig[cid], text_b=r.text))
        df = pd.DataFrame(rows)
    else:
        main = ids[:N_MAIN]
        # (1) Persona guises: 20 complaints with unanimous and 20 with non-unanimous GPT-4o-mini monetary decisions
        mon = d0[d0.case_id.isin(main)].assign(m=lambda x: (x.tier == 2).astype(float)).groupby("case_id").m.mean()
        strata = {"unanimous": sorted(mon[(mon == 0) | (mon == 1)].index), "mixed": sorted(mon[(mon > 0) & (mon < 1)].index)}
        for s, pool in strata.items():
            pick = rng.choice(pool, N_GUISE_PER_STRATUM, replace=False)
            w = len(pool) / len(mon) / N_GUISE_PER_STRATUM          # complaint weight (population share per sampled complaint)
            for cid in sorted(pick):
                for r in rw[rw.case_id == cid].itertuples():
                    rows.append(dict(set="guise", stratum=s, weight=w, case_id=cid, label=r.persona_label,
                                     text_a=orig[cid], text_b=r.text))
        # (2) Standardisation: guise (M0) vs its A_plain rewrite; strata by whether GPT-4o-mini's tier changed
        st = pd.read_csv("holdout_out/style_mitigation/rewrites.csv")
        st = st[(st.kind == "guise") & (st["style"] == "A_plain") & st.case_id.isin(main) & st.text.notna()]
        dm = pd.read_csv("holdout_out/style_mitigation/decisions.csv")
        dm = dm[(dm["style"] == "A_plain") & dm.case_id.isin(main)][["case_id", "persona_label", "tier"]]
        g = rw[rw.case_id.isin(main)][["case_id", "persona_label", "text"]].rename(columns={"text": "m0_text"})
        x = (st.merge(g, on=["case_id", "persona_label"])
               .merge(d0.rename(columns={"variant": "persona_label", "tier": "tier_m0"})[["case_id", "persona_label", "tier_m0"]],
                      on=["case_id", "persona_label"])
               .merge(dm.rename(columns={"tier": "tier_m1"}), on=["case_id", "persona_label"]))
        x["stratum"] = np.where(x.tier_m0 != x.tier_m1, "changed", "unchanged")
        for s, n in N_STD.items():
            pool = x[x.stratum == s].sort_values(["case_id", "persona_label"])
            pick = pool.iloc[rng.choice(len(pool), n, replace=False)]
            w = len(pool) / len(x) / n
            for r in pick.itertuples():
                rows.append(dict(set="std", stratum=s, weight=w, case_id=r.case_id, label=r.persona_label,
                                 text_a=r.m0_text, text_b=r.text))
        # (3) Auditor sensitivity: original vs fact edit (positive controls)
        e = pd.read_csv("holdout_out/positive_controls/edits.csv").dropna(subset=["text"])
        e = e[e.case_id.isin(ids[:200])]
        for s, n in N_EDIT.items():
            pool = e[e.edit == s].sort_values("case_id")
            pick = pool.iloc[rng.choice(len(pool), n, replace=False)]
            for r in pick.itertuples():
                rows.append(dict(set="edit", stratum=s, weight=1.0 / n, case_id=r.case_id, label=s,
                                 text_a=orig[r.case_id], text_b=r.text))
        df = pd.DataFrame(rows)
    df.insert(0, "pair_id", [f"p{i:05d}" for i in range(len(df))])
    o = out_dir(smoke)
    o.mkdir(parents=True, exist_ok=True)
    df.to_csv(o / "audit_pairs.csv", index=False)
    logger.info("audit pairs: %s", df.groupby(["set", "stratum"]).size().to_dict())
    return df


# ---------------------------------------------------------------------------
# Exact code checks (amounts, numbers, years)
# ---------------------------------------------------------------------------

_MONEY = re.compile(r"\$\s?\{?\s?\$?([0-9][0-9,]*(?:\.[0-9]{1,2})?)")
_NUM = re.compile(r"(?<![\w.$])([0-9][0-9,]*(?:\.[0-9]+)?)(?![\w])")
_YEAR = re.compile(r"\b(19[5-9][0-9]|20[0-4][0-9])\b")


def _money(t: str) -> List[float]:
    return sorted(round(float(m.replace(",", "")), 2) for m in _MONEY.findall(t or ""))


def code_check(a: str, b: str) -> Dict[str, Any]:
    ma, mb = _money(a), _money(b)
    ya, yb = sorted(set(_YEAR.findall(a or ""))), sorted(set(_YEAR.findall(b or "")))
    return {"money_a": ma, "money_b": mb, "money_mismatch": sorted(set(ma)) != sorted(set(mb)),
            "years_mismatch": ya != yb}


# ---------------------------------------------------------------------------
# LLM auditors
# ---------------------------------------------------------------------------

def _user(a: str, b: str) -> str:
    return f"TEXT A (original):\n{a}\n\nTEXT B (rewrite):\n{b}"


def call_claude(client, a: str, b: str) -> Dict[str, Any]:
    r = client.messages.create(
        model="claude-opus-5-5", max_tokens=8000, system=AUDIT_SYSTEM,
        messages=[{"role": "user", "content": _user(a, b)}],
        extra_body={"output_config": {"effort": "low", "format": {"type": "json_schema", "schema": AUDIT_SCHEMA}}})
    if r.stop_reason != "end_turn":
        raise RuntimeError(f"stop_reason={r.stop_reason}")
    txt = next(bl.text for bl in r.content if bl.type == "text")
    return {"parsed": json.loads(txt), "in": r.usage.input_tokens, "out": r.usage.output_tokens}


def call_gpt(client, a: str, b: str) -> Dict[str, Any]:
    r = client.chat.completions.create(
        model="gpt-5.2", reasoning_effort="low",
        messages=[{"role": "system", "content": AUDIT_SYSTEM}, {"role": "user", "content": _user(a, b)}],
        response_format={"type": "json_schema", "json_schema": {"name": "fact_audit", "strict": True, "schema": AUDIT_SCHEMA}})
    return {"parsed": json.loads(r.choices[0].message.content), "in": r.usage.prompt_tokens, "out": r.usage.completion_tokens}


def cmd_audit(smoke: bool, workers: int = 8):
    load_dotenv(".env")
    import anthropic
    import openai
    clients = {"claude-opus-5-5": anthropic.Anthropic(max_retries=4), "gpt-5.2": openai.OpenAI(max_retries=4)}
    callers = {"claude-opus-5-5": call_claude, "gpt-5.2": call_gpt}
    o = out_dir(smoke)
    pairs = pd.read_csv(o / "audit_pairs.csv")
    store = o / "audit_responses.jsonl"
    done, cost = set(), 0.0
    if store.exists():
        for line in store.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            cost += rec.get("cost", 0.0)
            if "parsed" in rec:
                done.add((rec["pair_id"], rec["auditor"]))
    todo = [(p.pair_id, aud, p.text_a, p.text_b) for p in pairs.itertuples() for aud in AUDITORS if (p.pair_id, aud) not in done]
    logger.info("%d audits done ($%.2f so far), %d to do", len(done), cost, len(todo))
    lock, stop = threading.Lock(), threading.Event()

    def run(item):
        pid, aud, a, b = item
        if stop.is_set():
            return None
        try:
            res = callers[aud](clients[aud], a, b)
            pin, pout = PRICES[aud]
            return {"pair_id": pid, "auditor": aud, "parsed": res["parsed"], "in": res["in"], "out": res["out"],
                    "cost": res["in"] * pin / 1e6 + res["out"] * pout / 1e6}
        except Exception as ex:  # recorded and retried on the next run
            return {"pair_id": pid, "auditor": aud, "error": repr(ex)[:300], "cost": 0.0}

    n_ok = n_err = 0
    with store.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(run, t) for t in todo]), 1):
            rec = f.result()
            if rec is None:
                continue
            with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                cost += rec["cost"]
                n_ok += "parsed" in rec
                n_err += "error" in rec
                if cost >= COST_CAP_AUDIT and not stop.is_set():
                    logger.error("cost cap $%.2f reached; stopping", COST_CAP_AUDIT)
                    stop.set()
            if i % 100 == 0:
                logger.info("  %d/%d (ok %d, errors %d), cost so far $%.2f", i, len(todo), n_ok, n_err, cost)
    logger.info("audit pass finished: ok %d, errors %d, total cost $%.2f", n_ok, n_err, cost)
    (o / "audit_cost.json").write_text(json.dumps({"total_cost_usd": cost}, indent=2))


# ---------------------------------------------------------------------------
# Same-day baseline: fresh decisions on M0 and M1 texts, interleaved
# ---------------------------------------------------------------------------

def baseline_frame(smoke: bool) -> pd.DataFrame:
    ids = holdout_ids()
    if smoke:
        cases = ids[500:502]
    else:
        cases = sorted(np.random.default_rng(SEED + 1).choice(ids[:N_MAIN], N_BASELINE, replace=False))
    rw = pd.read_csv("holdout_out/rewrites_n1000.csv")
    m0 = rw[(rw.rewrite_model == BANK) & rw.case_id.isin(cases) & rw.text.notna()][["case_id", "persona_label", "text"]].assign(arm="M0")
    if smoke:
        m1 = m0.assign(arm="M1")       # smoke set has no standardised texts; exercise the code path only
    else:
        st = pd.read_csv("holdout_out/style_mitigation/rewrites.csv")
        m1 = st[(st.kind == "guise") & (st["style"] == "A_plain") & st.case_id.isin(cases) & st.text.notna()][
            ["case_id", "persona_label", "text"]].assign(arm="M1")
    df = pd.concat([m0, m1], ignore_index=True)
    return df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)   # interleave the arms


def cmd_baseline(smoke: bool):
    o = out_dir(smoke)
    o.mkdir(parents=True, exist_ok=True)
    df = baseline_frame(smoke)
    df.to_csv(o / "baseline_texts.csv", index=False)
    # GPT-4o-mini: same prompt, client and settings as the frozen run, but a fresh cache, so every call is new
    load_dotenv(".env")
    from src.schemas import LLMConfig
    cfg = LLMConfig(api_key=os.getenv("OPENAI_API_KEY", ""), models=[BANK], temperature=0.0, max_tokens=16384,
                    timeout=120, max_retries=3, retry_delay=1.0)
    llm = LLMClient(cfg, ResponseCache(str(Path("cache") / ("fresh_baseline_smoke" if smoke else "fresh_baseline_20261022"))), max_workers=30)
    llm.CIRCUIT_BREAKER_WINDOW = 10 ** 9
    frame = pd.DataFrame(dict(case_id=df.case_id, rewrite_model=BANK, persona_label=df.persona_label + "||" + df.arm, text=df.text))
    d = hp.decide(llm, frame, BANK)
    d[["persona_label", "arm"]] = d.variant.str.split(r"\|\|", expand=True)
    d.to_csv(o / "baseline_decisions.csv", index=False)
    (o / "usage_baseline_decide.json").write_text(json.dumps(llm.usage_summary(), indent=2, default=str))
    # Jev: tier questions under all six orders, fresh output folder
    j = pd.DataFrame(dict(case_id=df.case_id, source=df.arm, rewrite_model=BANK, persona_label=df.persona_label, text=df.text))
    j["text_id"] = [hp_tid(t) for t in j.text]
    (o / "baseline_jev").mkdir(parents=True, exist_ok=True)
    q = jarms.order_questions(list(jarms.ORDERS), with_persona=False)
    jp.build_questions = lambda: q
    resp = jp.score_texts(j, o / "baseline_jev", 16)
    s = jarms.flatten(j, resp)
    s.to_csv(o / "baseline_jev" / "scores.csv", index=False)
    logger.info("baseline: %d GPT decisions (%d failed), %d Jev texts; Jev cost $%.2f", len(d), int(d.tier.isna().sum()),
                len(s), s.input_tokens.sum() * jp.PRICE_PER_M_INPUT / 1e6)


def hp_tid(t: str) -> str:
    import hashlib
    return hashlib.blake2b(t.encode(), digest_size=12).hexdigest()


# ---------------------------------------------------------------------------
# Frozen analysis
# ---------------------------------------------------------------------------

def load_audits(o: Path) -> pd.DataFrame:
    recs = [json.loads(line) for line in (o / "audit_responses.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = []
    for r in recs:
        if "parsed" not in r:
            continue
        p = r["parsed"]
        mats = [x for x in p["discrepancies"] if x["material"]]
        rows.append({"pair_id": r["pair_id"], "auditor": r["auditor"], "overall": p["overall"],
                     "material": p["overall"] == "material" or len(mats) > 0, "n_disc": len(p["discrepancies"]),
                     "material_categories": ";".join(sorted({x["category"] for x in mats}))})
    a = pd.DataFrame(rows).drop_duplicates(["pair_id", "auditor"], keep="last")
    return a.pivot(index="pair_id", columns="auditor")


def wmean_ci(v: np.ndarray, w: np.ndarray, cluster: np.ndarray, rng) -> Dict[str, Any]:
    """Weighted mean with a cluster (complaint) bootstrap."""
    ok = ~np.isnan(v)
    v, w, cluster = v[ok], w[ok], cluster[ok]
    if len(v) == 0:
        return {"estimate": None, "ci95": [None, None], "n": 0}
    est = float(np.sum(v * w) / np.sum(w)) if len(v) else float("nan")
    cl = np.unique(cluster)
    idx = {c: np.where(cluster == c)[0] for c in cl}
    b = []
    for _ in range(N_BOOT):
        ix = np.concatenate([idx[c] for c in rng.choice(cl, len(cl))])
        b.append(np.sum(v[ix] * w[ix]) / np.sum(w[ix]))
    return {"estimate": est, "ci95": [float(np.nanquantile(b, .025)), float(np.nanquantile(b, .975))], "n": int(len(v))}


def pairwise_disagree(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    iu = np.triu_indices(len(x), 1)
    return float(np.mean(x[iu[0]] != x[iu[1]]))


def mpad(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    iu = np.triu_indices(len(x), 1)
    return float(np.mean(np.abs(x[iu[0]] - x[iu[1]])))


def cmd_analyse(smoke: bool):
    rng = np.random.default_rng(SEED)
    o = out_dir(smoke)
    pairs = pd.read_csv(o / "audit_pairs.csv").set_index("pair_id")
    A = load_audits(o)
    res: Dict[str, Any] = {"auditors": AUDITORS, "n_pairs": {k: int(v) for k, v in pairs.groupby("set").size().items()}}
    for aud in AUDITORS:
        pairs[f"mat_{aud}"] = A[("material", aud)].reindex(pairs.index).astype(float) if ("material", aud) in A else np.nan
    cc = pd.DataFrame([code_check(a, b) for a, b in zip(pairs.text_a, pairs.text_b)], index=pairs.index)
    pairs["code_money"] = cc.money_mismatch.astype(float)
    pairs["code_years"] = cc.years_mismatch.astype(float)
    res["completeness"] = {aud: int(pairs[f"mat_{aud}"].notna().sum()) for aud in AUDITORS}

    # FA1: auditor validity on fact edits (each auditor must flag >= 80% of strengthen/weaken edits as material)
    e = pairs[pairs.set == "edit"]
    res["FA1_edit_flag_rate"] = {aud: {s: float(e.loc[e.stratum == s, f"mat_{aud}"].mean()) for s in N_EDIT} | {
        "all": float(e[f"mat_{aud}"].mean())} for aud in AUDITORS}
    valid = [aud for aud in AUDITORS if res["FA1_edit_flag_rate"][aud]["all"] >= 0.80]
    res["FA1_valid_auditors"] = valid
    use = valid if valid else AUDITORS
    pairs["mat_any"] = pairs[[f"mat_{a}" for a in use]].max(axis=1)
    pairs["mat_all"] = pairs[[f"mat_{a}" for a in use]].min(axis=1)
    pairs["verified"] = ((pairs.mat_any == 0) & (pairs.code_money == 0)).astype(float)
    pairs.loc[pairs.mat_any.isna(), "verified"] = np.nan

    # FA2 / FA5: weighted material-change rates (persona guises; standardised texts)
    for key, s in (("FA2_guise", "guise"), ("FA5_standardised", "std")):
        g = pairs[pairs.set == s]
        r = {}
        for col in [f"mat_{a}" for a in AUDITORS] + ["mat_any", "mat_all", "code_money", "code_years", "verified"]:
            r[col] = wmean_ci(g[col].to_numpy(float), g.weight.to_numpy(float), g.case_id.to_numpy(), rng)
        r["by_stratum"] = {st: {c: float(gg[c].mean()) for c in ("mat_any", "mat_all", "code_money")} for st, gg in g.groupby("stratum")}
        cats: Dict[str, int] = {}
        for aud in use:
            for v in A[("material_categories", aud)].reindex(g.index).dropna():
                for c in filter(None, str(v).split(";")):
                    cats[c] = cats.get(c, 0) + 1
        r["material_categories_count"] = cats
        res[key] = r

    # FA3: individual instability among verified fact-preserving guises vs all guises (same complaints)
    g = pairs[pairs.set == "guise"].reset_index()
    d0 = pd.read_csv("holdout_out/decisions_n1000.csv")
    d0 = d0[d0.model == BANK].rename(columns={"variant": "label"})[["case_id", "label", "tier"]]
    j0 = pd.read_csv("holdout_out/jev/scores.csv")
    j0 = j0[(j0.source == "guise") & (j0.rewrite_model == BANK)].rename(columns={"persona_label": "label"})
    j0 = j0.assign(jev_mon=(j0.argmax_o012 == 2).astype(float), jev_e=j0.p1_o012 + 2 * j0.p2_o012)[["case_id", "label", "jev_mon", "jev_e"]]
    g = g.merge(d0, on=["case_id", "label"], how="left").merge(j0.drop_duplicates(["case_id", "label"]), on=["case_id", "label"], how="left")
    g["gpt_mon"] = (g.tier == 2).astype(float)
    per = []
    for cid, c in g.groupby("case_id"):
        v = c[c.verified == 1]
        row = {"case_id": cid, "weight": c.weight.iloc[0], "n_guises": len(c), "n_verified": len(v)}
        for out, fn in (("gpt_mon", pairwise_disagree), ("jev_mon", pairwise_disagree), ("tier", mpad), ("jev_e", mpad)):
            row[f"{out}_all"] = fn(c[out].to_numpy(float))
            row[f"{out}_ver"] = fn(v[out].to_numpy(float))
        per.append(row)
    per = pd.DataFrame(per)
    per.to_csv(o / "fa3_per_complaint.csv", index=False)
    fa3 = {"n_complaints": len(per), "mean_verified_per_complaint": float(per.n_verified.mean())}
    for out in ("gpt_mon", "jev_mon", "tier", "jev_e"):
        sub = per.dropna(subset=[f"{out}_all", f"{out}_ver"])
        w = sub.weight.to_numpy()
        A_, V_ = sub[f"{out}_all"].to_numpy(), sub[f"{out}_ver"].to_numpy()
        est_all, est_ver = np.sum(w * A_) / np.sum(w), np.sum(w * V_) / np.sum(w)
        b = []
        for _ in range(N_BOOT):
            ix = rng.integers(0, len(sub), len(sub))
            da = np.sum(w[ix] * A_[ix])
            b.append(np.sum(w[ix] * V_[ix]) / da if da else np.nan)
        fa3[out] = {"all": float(est_all), "verified": float(est_ver), "ratio_verified_to_all": float(est_ver / est_all) if est_all else None,
                    "ratio_ci95": [float(np.nanquantile(b, .025)), float(np.nanquantile(b, .975))], "n_complaints": int(len(sub))}
    fa3["supported"] = bool(all((fa3[k]["ratio_verified_to_all"] or 0) >= 0.5 for k in ("gpt_mon", "jev_mon")))
    res["FA3_instability_among_verified"] = fa3

    # FA4: are guises whose GPT-4o-mini monetary decision departs from the complaint majority more often flagged?
    maj = g.groupby("case_id").gpt_mon.transform(lambda s: float(s.mean() >= 0.5))
    g["departs"] = (g.gpt_mon != maj).astype(float)
    res["FA4_flag_rate_by_departure"] = {"departs": float(g.loc[g.departs == 1, "mat_any"].mean()) if (g.departs == 1).any() else None,
                                         "agrees": float(g.loc[g.departs == 0, "mat_any"].mean()),
                                         "n_departs": int((g.departs == 1).sum())}

    # FA6: agreement between auditors and with the code check
    both = pairs.dropna(subset=[f"mat_{a}" for a in AUDITORS])
    x, y = both[f"mat_{AUDITORS[0]}"].to_numpy(), both[f"mat_{AUDITORS[1]}"].to_numpy()
    po = float(np.mean(x == y))
    pe = float(x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean()))
    res["FA6_agreement"] = {"raw": po, "cohen_kappa": (po - pe) / (1 - pe) if pe < 1 else None, "n": int(len(both)),
                            "code_money_flagged_and_auditor_material": {a: float(pairs.loc[pairs.code_money == 1, f"mat_{a}"].mean())
                                                                        for a in AUDITORS if (pairs.code_money == 1).any()}}
    if (o / "baseline_decisions.csv").exists():
        res["baseline"] = analyse_baseline(o, rng)
    (o / "pairs_scored.csv").write_text(pairs.drop(columns=["text_a", "text_b"]).to_csv())
    dest = o / "results.json" if smoke else Path("prereg/fidelity_audit_results.json")
    dest.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k.startswith(("FA1", "FA3"))}, indent=2, default=float))


CONTR = {"edu": ("college", "high_school"), "ses": ("high", "low"), "age": ("old", "young")}


def _lvl(lbl: str, dim: str) -> str:
    return {"age": lbl.split("_")[0], "edu": "high_school" if "_high_school_" in lbl else "college", "ses": lbl.split("_")[-1]}[dim]


def analyse_baseline(o: Path, rng) -> Dict[str, Any]:
    d = pd.read_csv(o / "baseline_decisions.csv")
    cases = sorted(d.case_id.unique())
    d0 = pd.read_csv("holdout_out/decisions_n1000.csv")
    d0 = d0[(d0.model == BANK) & d0.case_id.isin(cases)].rename(columns={"variant": "persona_label"})
    dm = pd.read_csv("holdout_out/style_mitigation/decisions.csv")
    dm = dm[(dm["style"] == "A_plain") & dm.case_id.isin(cases)]
    fresh0 = d[d.arm == "M0"][["case_id", "persona_label", "tier"]]
    fresh1 = d[d.arm == "M1"][["case_id", "persona_label", "tier"]]
    m = fresh0.merge(d0[["case_id", "persona_label", "tier"]], on=["case_id", "persona_label"], suffixes=("_fresh", "_old"))
    m1 = fresh1.merge(dm[["case_id", "persona_label", "tier"]], on=["case_id", "persona_label"], suffixes=("_fresh", "_old"))
    out: Dict[str, Any] = {"n_complaints": len(cases)}
    for nm, mm in (("M0_19oct_vs_fresh", m), ("M1_21oct_vs_fresh", m1)):
        diff = (mm.tier_fresh - mm.tier_old).to_numpy(float)
        cl = mm.case_id.to_numpy()
        out[f"DB1_gpt_{nm}"] = {"exact_agreement": float(np.mean(mm.tier_fresh == mm.tier_old)),
                                "mean_diff": wmean_ci(diff, np.ones(len(diff)), cl, rng)}
    # Jev drift
    js = pd.read_csv(o / "baseline_jev" / "scores.csv").assign(e=lambda x: x.p1_o012 + 2 * x.p2_o012)
    j0 = pd.read_csv("holdout_out/jev/scores.csv")
    j0 = j0[(j0.source == "guise") & (j0.rewrite_model == BANK) & j0.case_id.isin(cases)].assign(e=lambda x: x.p1_o012 + 2 * x.p2_o012)
    jm = js[js.source == "M0"].merge(j0[["case_id", "persona_label", "e", "p0_o012", "p1_o012", "p2_o012"]].drop_duplicates(["case_id", "persona_label"]),
                                     on=["case_id", "persona_label"], suffixes=("_fresh", "_old"))
    tv = 0.5 * sum(np.abs(jm[f"p{k}_o012_fresh"] - jm[f"p{k}_o012_old"]) for k in range(3))
    out["DB1_jev_M0_old_vs_fresh"] = {"mean_tv": float(tv.mean()),
                                      "mean_diff_e": wmean_ci((jm.e_fresh - jm.e_old).to_numpy(float), np.ones(len(jm)), jm.case_id.to_numpy(), rng)}

    # DB2: MH1/MH2 re-estimated on these complaints with all-fresh decisions, and with the original decisions
    def gaps(t0: pd.DataFrame, t1: pd.DataFrame, val: str) -> Dict[str, Any]:
        W0 = t0.pivot_table(index="case_id", columns="persona_label", values=val)
        W1 = t1.pivot_table(index="case_id", columns="persona_label", values=val)
        cols = sorted(set(W0.columns) & set(W1.columns))
        W0, W1 = W0[cols].dropna(), W1[cols].dropna()
        idx = W0.index.intersection(W1.index)
        W0, W1 = W0.loc[idx], W1.loc[idx]
        r: Dict[str, Any] = {"n_complaints": len(idx)}
        for dim, (hi, lo) in CONTR.items():
            lv = np.array([_lvl(c, dim) for c in cols])
            g0 = W0.to_numpy()[:, lv == hi].mean(1) - W0.to_numpy()[:, lv == lo].mean(1)
            g1 = W1.to_numpy()[:, lv == hi].mean(1) - W1.to_numpy()[:, lv == lo].mean(1)
            C = len(g0)
            bb = [(g0[ix] - g1[ix]).mean() for ix in (rng.integers(0, C, C) for _ in range(N_BOOT))]
            r[dim] = {"gap_M0": float(g0.mean()), "gap_M1": float(g1.mean()), "reduction": float((g0 - g1).mean()),
                      "reduction_ci95": [float(np.quantile(bb, .025)), float(np.quantile(bb, .975))]}
        a0 = np.array([mpad(row) for row in W0.to_numpy(float)])
        a1 = np.array([mpad(row) for row in W1.to_numpy(float)])
        C = len(a0)
        bb = [(a0[ix] - a1[ix]).mean() for ix in (rng.integers(0, C, C) for _ in range(N_BOOT))]
        r["mpad"] = {"M0": float(a0.mean()), "M1": float(a1.mean()), "M0_minus_M1": float((a0 - a1).mean()),
                     "ci95": [float(np.quantile(bb, .025)), float(np.quantile(bb, .975))]}
        return r

    out["DB2_gpt_fresh"] = gaps(fresh0, fresh1, "tier")
    out["DB2_gpt_original"] = gaps(d0[["case_id", "persona_label", "tier"]], dm[["case_id", "persona_label", "tier"]], "tier")
    jf0, jf1 = js[js.source == "M0"], js[js.source == "M1"]
    out["DB2_jev_fresh"] = gaps(jf0, jf1, "e")
    out["DB2_supported_gpt_edu_ses"] = bool(all(out["DB2_gpt_fresh"][k]["reduction_ci95"][0] > 0 for k in ("edu", "ses")))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["sample", "audit", "baseline", "analyse"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    {"sample": lambda: cmd_sample(a.smoke), "audit": lambda: cmd_audit(a.smoke), "baseline": lambda: cmd_baseline(a.smoke),
     "analyse": lambda: cmd_analyse(a.smoke)}[a.cmd]()


if __name__ == "__main__":
    main()
