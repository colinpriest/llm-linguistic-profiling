"""Additional Jev scoring arms for the FAccT paper.  Run from the repository root.

    python jev_arms.py build-texts                       # cache exploratory texts to parquet (no API calls)
    python jev_arms.py repeat   [--n 1000]               # determinism: re-score texts with the pilot question set
    python jev_arms.py orders   --panel exploratory      # the 4 option orders not yet scored
    python jev_arms.py orders   --panel holdout --tag n1000   # all 6 orders + persona questions
    add --limit N to any scoring command for a smoke test, --dry-run to count texts and estimate cost

Responses are appended to a JSONL store per arm (resumable; nothing is paid for twice).
Outputs: jev_pilot_out/{repeat,orders}/ and holdout_out/jev/.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

import jev_pilot as jp

logger = logging.getLogger("jev_arms")

TEXTS_PARQUET = Path("jev_pilot_out/texts_exploratory.parquet")
TIERS = list(jp.TIER_CRITERIA)                       # no_action, non_monetary_action, monetary_action
ORDERS = {"o" + "".join(str(TIERS.index(t)) for t in perm): perm for perm in itertools.permutations(TIERS)}
SCORED_IN_PILOT = {"o012", "o210"}                   # tier_fwd and tier_rev
TOKENS_PER_TIER_Q = 600                              # approximate, for cost estimates only


def tier_question(order: tuple) -> Dict[str, Any]:
    return {"type": "choice", "instructions": jp.TIER_INSTRUCTIONS,
            "criteria": {k: jp.TIER_CRITERIA[k] for k in order}}


def order_questions(which: List[str], with_persona: bool) -> Dict[str, Any]:
    q = {f"tier_{o}": tier_question(ORDERS[o]) for o in which}
    if with_persona:
        q.update({k: v for k, v in jp.build_questions().items() if k.startswith("persona_")})
    return q


def text_id(t: str) -> str:
    return hashlib.blake2b(t.encode(), digest_size=12).hexdigest()


def exploratory_texts() -> pd.DataFrame:
    if not TEXTS_PARQUET.exists():
        raise SystemExit(f"{TEXTS_PARQUET} missing; run `python jev_arms.py build-texts` first")
    return pd.read_parquet(TEXTS_PARQUET)


def holdout_texts(tag: str) -> pd.DataFrame:
    sample = pd.read_csv("holdout_out/holdout_sample_full.csv")
    rw = pd.read_csv(f"holdout_out/rewrites_{tag}.csv")
    rw = rw[rw.text.notna()]
    sample = sample[sample.case_id.isin(rw.case_id)]
    orig = pd.DataFrame(dict(case_id=sample.case_id, source="original", rewrite_model="",
                             persona_label="", text=sample.narrative))
    guise = rw.assign(source="guise")[["case_id", "source", "rewrite_model", "persona_label", "text"]]
    df = pd.concat([orig, guise], ignore_index=True)
    df["text_id"] = [text_id(t) for t in df.text]
    return df


def flatten(df: pd.DataFrame, responses: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    recs = []
    for row in df.itertuples(index=False):
        r = responses.get(row.text_id)
        if not r:
            continue
        rec = dict(text_id=row.text_id, case_id=row.case_id, source=row.source,
                   rewrite_model=row.rewrite_model, persona_label=row.persona_label,
                   jev_model=r.get("model"), input_tokens=r["usage"]["input_tokens"])
        for qid, ans in r["answers"].items():
            if qid.startswith("tier_"):
                o = qid[5:]
                for i, k in enumerate(TIERS):
                    rec[f"p{i}_{o}"] = ans["probabilities"].get(k, 0.0)
                rec[f"argmax_{o}"] = TIERS.index(ans["choice"])
            elif qid.startswith("persona_"):
                d = qid[8:]
                rec[f"{d}_choice"] = ans["choice"]
                for k, v in ans["probabilities"].items():
                    rec[f"{d}_p_{k}"] = v
        recs.append(rec)
    return pd.DataFrame(recs)


def run(df: pd.DataFrame, questions: Dict[str, Any], out: Path, workers: int, dry_run: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    n_unique = df.text_id.nunique()
    n_tier = sum(k.startswith("tier_") for k in questions)
    est_tokens = 400 + n_tier * TOKENS_PER_TIER_Q + (900 if any(k.startswith("persona_") for k in questions) else 0)
    logger.info("%d texts (%d unique); questions %s; est. cost $%.2f",
                len(df), n_unique, list(questions), n_unique * est_tokens * jp.PRICE_PER_M_INPUT / 1e6)
    if dry_run:
        return
    jp.build_questions = lambda: questions            # score_texts builds questions via this hook
    responses = jp.score_texts(df, out, workers)
    s = flatten(df, responses)
    s.to_csv(out / "scores.csv", index=False)
    tok = int(s.input_tokens.sum()) if len(s) else 0
    summary = {"texts": len(s), "unique_texts_scored": len(responses), "input_tokens_in_panel": tok,
               "cost_usd_panel": tok * jp.PRICE_PER_M_INPUT / 1e6,
               "jev_models": sorted(s.jev_model.dropna().unique().tolist()) if len(s) else []}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Done: %s", summary)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm", choices=["build-texts", "repeat", "orders"])
    ap.add_argument("--panel", choices=["exploratory", "holdout"], default="exploratory")
    ap.add_argument("--tag", default="n1000", help="holdout rewrites tag (rewrites_<tag>.csv)")
    ap.add_argument("--n", type=int, default=1000, help="texts for the repeat arm")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.arm == "build-texts":
        df = jp.load_texts(2000, 42)
        df.to_parquet(TEXTS_PARQUET, index=False)
        logger.info("Wrote %s (%d rows)", TEXTS_PARQUET, len(df))
        return

    if args.arm == "repeat":
        df = exploratory_texts().drop_duplicates("text_id").sample(args.n, random_state=args.seed)
        questions = jp.build_questions()              # identical to the pilot question set
        out = Path("jev_pilot_out/repeat")
    elif args.panel == "exploratory":
        df = exploratory_texts()
        questions = order_questions([o for o in ORDERS if o not in SCORED_IN_PILOT], with_persona=False)
        out = Path("jev_pilot_out/orders")
    else:
        df = holdout_texts(args.tag)
        questions = order_questions(list(ORDERS), with_persona=True)
        out = Path("holdout_out/jev")
    if args.limit:
        df = df.head(args.limit)
    run(df, questions, out, args.workers, args.dry_run)


if __name__ == "__main__":
    main()
