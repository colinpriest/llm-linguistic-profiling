"""Holdout matched-guise panel for the Jev paper.

Draws fresh complaints that were never part of the frozen run's matched-guise
sample, rewrites each into the 36 persona styles with the same prompts, models
and client settings as Steps 4-5 of the original experiment, and (optionally)
collects GPT remedy decisions on the rewrites.  Run from the repository root:

    python holdout_panel.py --dry-run                      # counts + cost estimate, no API calls
    python holdout_panel.py --limit-complaints 5           # smoke test (first 5 of the frozen sample)
    python holdout_panel.py --n-complaints 1000            # full holdout
    python holdout_panel.py --n-complaints 1000 --no-decisions

The sample is fixed by --seed and --n-complaints; a smoke test uses the first
complaints of that same sample, so its responses are reused (via the LLM cache)
by the full run.  Outputs go to holdout_out/.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv

from src.cache import ResponseCache
from src.llm_client import LLMClient
from src.prompts import (
    PERSONA_REWRITE_SYSTEM,
    REMEDY_DECISION_SYSTEM,
    REMEDY_DECISION_USER,
    format_persona_rewrite_user,
    safe_narrative_format,
)
from src.schemas import (
    PERSONA_GRID,
    LLMConfig,
    LLMRequest,
    PersonaRewriteResponse,
    RemedyDecisionResponse,
    persona_label,
)

logger = logging.getLogger("holdout_panel")

POOL_PATH = "data/.pipeline/step10_seed42_n20000.parquet"
RUN_TS = "20260220_101746"
MODELS = ["gpt-4o-mini", "gpt-5-mini"]
# Mean cost per call observed in the February runs (all tasks), used only for --dry-run estimates
COST_PER_CALL = {"gpt-4o-mini": 0.00029, "gpt-5-mini": 0.0021}


def draw_sample(n: int, seed: int) -> pd.DataFrame:
    pool = pd.read_parquet(POOL_PATH, columns=["case_id", "narrative"])
    used = set(pd.read_csv(f"output/results_persona_decisions_{RUN_TS}.csv", usecols=["case_id"]).case_id)
    fresh = sorted(set(pool.case_id) - used)
    # Shuffle the whole fresh pool once and take a prefix, so a larger holdout extends a smaller one
    random.Random(seed).shuffle(fresh)
    ids = fresh[:n]
    logger.info("Pool %d, used in matched-guise %d, fresh %d, sampled %d", len(pool), len(used), len(fresh), n)
    return pool.set_index("case_id").loc[ids].reset_index()


def make_client(models: List[str], threads: int) -> LLMClient:
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise SystemExit("OPENAI_API_KEY not set")
    # Identical to Experiment.__init__ so cache keys and call settings match the frozen run
    config = LLMConfig(api_key=key, models=models, temperature=0.0, max_tokens=16384,
                       timeout=120, max_retries=3, retry_delay=1.0)
    client = LLMClient(config, ResponseCache("./cache"), max_workers=threads)
    # LLMClient's breaker aborts a whole batch once more than 6 requests have failed in total,
    # so transient DNS failures on this memory-starved machine killed 36k-request batches.
    # Disable it here; failures are recorded per request and retried on the next run.
    client.CIRCUIT_BREAKER_WINDOW = 10 ** 9
    # Build the response schemas once in the main thread.  Otherwise dozens of worker threads
    # trigger pydantic/instructor's lazy imports and schema generation simultaneously, which
    # deadlocked on the import lock and crashed under memory pressure (19 Oct).
    try:
        from instructor.processing.schema import generate_openai_schema
        for schema in (PersonaRewriteResponse, RemedyDecisionResponse):
            schema.model_json_schema()
            generate_openai_schema(schema)
    except Exception as e:  # warm-up is best-effort
        logger.warning("Schema warm-up skipped: %s", e)
    return client


def rewrite(llm: LLMClient, sample: pd.DataFrame, model: str) -> pd.DataFrame:
    requests, keys = [], []
    for cid, text in zip(sample.case_id, sample.narrative):
        for p in PERSONA_GRID:
            user = format_persona_rewrite_user(text, p["age"], p["geo"], p["edu"], p["ses"])
            requests.append(LLMRequest(request_id=LLMRequest.new_id(), case_id=cid, model=model,
                                       task="persona_rewrite", prompt_system=PERSONA_REWRITE_SYSTEM,
                                       prompt_user=user, expected_schema=PersonaRewriteResponse,
                                       variant=persona_label(**p)))
            keys.append(user)
    responses = llm.execute_batch(requests, keys, desc=f"Holdout [{model}] persona rewriting")
    rows = []
    for req, resp in zip(requests, responses):
        ok = resp.success and resp.response_parsed is not None
        rows.append(dict(case_id=req.case_id, rewrite_model=model, persona_label=req.variant,
                         text=resp.response_parsed.rewritten_complaint if ok else None,
                         error=None if ok else resp.error_message))
    return pd.DataFrame(rows)


def decide(llm: LLMClient, rw: pd.DataFrame, model: str) -> pd.DataFrame:
    rw = rw[(rw.rewrite_model == model) & rw.text.notna()]
    requests = [LLMRequest(request_id=LLMRequest.new_id(), case_id=cid, model=model, task="remedy_decision",
                           prompt_system=REMEDY_DECISION_SYSTEM,
                           prompt_user=safe_narrative_format(REMEDY_DECISION_USER, t),
                           expected_schema=RemedyDecisionResponse, variant=pl)
                for cid, pl, t in zip(rw.case_id, rw.persona_label, rw.text)]
    responses = llm.execute_batch(requests, list(rw.text), desc=f"Holdout [{model}] persona remedy")
    rows = []
    for req, resp in zip(requests, responses):
        p = resp.response_parsed if resp.success else None
        rows.append(dict(case_id=req.case_id, model=model, variant=req.variant,
                         tier=getattr(p, "tier", None), confidence=getattr(p, "confidence", None),
                         information_needed=getattr(p, "information_needed", None),
                         error=None if p is not None else resp.error_message))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-complaints", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--limit-complaints", type=int, default=None, help="use only the first N of the sample")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--no-decisions", action="store_true", help="skip GPT remedy decisions on the rewrites")
    ap.add_argument("--threads", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--merge", action="store_true", help="merge per-model output files and exit")
    ap.add_argument("--out", default="holdout_out")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.merge:
        n = args.limit_complaints or args.n_complaints
        for kind in ("rewrites", "decisions", "usage"):
            parts = [out / f"{kind}_n{n}_{m}.{'json' if kind == 'usage' else 'csv'}" for m in MODELS]
            if kind == "usage":
                merged = {m: json.loads(p.read_text(encoding="utf-8")) for m, p in zip(MODELS, parts) if p.exists()}
                (out / f"usage_n{n}.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
            elif all(p.exists() for p in parts):
                pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).to_csv(out / f"{kind}_n{n}.csv", index=False)
                logger.info("Merged %s -> %s", [p.name for p in parts], f"{kind}_n{n}.csv")
        return
    sample = draw_sample(args.n_complaints, args.seed)
    sample.to_csv(out / "holdout_sample_full.csv", index=False)
    if args.limit_complaints:
        sample = sample.head(args.limit_complaints)
    k = len(sample) * len(PERSONA_GRID)
    est = {m: k * COST_PER_CALL[m] * (1 if args.no_decisions else 2) for m in args.models}
    logger.info("%d complaints -> %d rewrites per model%s; estimated cost %s (total $%.2f)",
                len(sample), k, "" if args.no_decisions else f" + {k} decisions per model",
                {m: round(v, 2) for m, v in est.items()}, sum(est.values()))
    if args.dry_run:
        return

    # Per-model runs (e.g. one process per model in parallel) get their own files; merge with --merge
    tag = f"n{len(sample)}" + ("" if list(args.models) == MODELS else "_" + "_".join(args.models))
    llm = make_client(args.models, args.threads)
    t0 = time.time()
    rw = pd.concat([rewrite(llm, sample, m) for m in args.models], ignore_index=True)
    rw.to_csv(out / f"rewrites_{tag}.csv", index=False)
    logger.info("Rewrites: %d ok, %d failed", rw.text.notna().sum(), rw.text.isna().sum())
    if not args.no_decisions:
        dec = pd.concat([decide(llm, rw, m) for m in args.models], ignore_index=True)
        dec.to_csv(out / f"decisions_{tag}.csv", index=False)
        logger.info("Decisions: %d ok, %d failed; tier dist %s", dec.tier.notna().sum(), dec.tier.isna().sum(),
                    dec.groupby("model").tier.value_counts(normalize=True).round(3).to_dict())
    usage = llm.usage_summary()
    usage["wall_s"] = round(time.time() - t0, 1)
    (out / f"usage_{tag}.json").write_text(json.dumps(usage, indent=2, default=str), encoding="utf-8")
    logger.info("Usage: %s", json.dumps(usage, default=str))
    n_failed = int(rw.text.isna().sum()) + (0 if args.no_decisions else int(dec.tier.isna().sum()))
    if n_failed:
        logger.warning("%d requests failed; exiting non-zero so a rerun retries them (successes are cached)", n_failed)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
