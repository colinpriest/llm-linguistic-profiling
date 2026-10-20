"""Score complaint texts on interpretable attributes with Jev (delivery / structure / content).

Pre-registered in prereg/attributes_prereg.md.  Run from the repository root:

    python jev_attributes.py pilot   [--dry-run] [--limit N]   # 200-complaint pilot (exploratory panel)
    python jev_attributes.py repeat  [--n 500]                 # test-retest subset of the pilot texts
    python jev_attributes.py holdout --tag n1000               # confirmatory phase (only if the pilot passes)

Responses go to a resumable JSONL store per arm; scores.csv has, per text and attribute,
P(low), P(medium), P(high), the choice and an expected score E = 0.5*P(medium) + P(high).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

import jev_arms as ja
import jev_pilot as jp

logger = logging.getLogger("jev_attributes")

INSTRUCTIONS = (
    "You are analysing how a consumer complaint to a bank is written. Judge ONLY the property "
    "described below, as it appears in this text. Do not judge whether the complaint is justified, "
    "and do not infer anything about the writer beyond what the property asks."
)

# family: delivery (how it is said) | structure (how it is organised) | content (what is said)
# Pre-specified normative relevance to the remedy decision: content = relevant,
# structure = procedurally relevant only (aids assessment, should not change merit), delivery = irrelevant.
ATTRIBUTES: Dict[str, Tuple[str, str, Dict[str, str]]] = {
    # ---- delivery ----
    "vocabulary": ("delivery", "Vocabulary sophistication", {
        "low": "Plain, everyday words only; limited vocabulary.",
        "medium": "Ordinary adult vocabulary with some precise or less common words.",
        "high": "Rich, sophisticated or technical vocabulary used fluently."}),
    "grammar": ("delivery", "Grammar, spelling and punctuation", {
        "low": "Frequent errors in grammar, spelling or punctuation.",
        "medium": "Occasional errors that do not impede reading.",
        "high": "Essentially error-free, polished writing."}),
    "formality": ("delivery", "Formality of register", {
        "low": "Casual, conversational or colloquial.",
        "medium": "Neutral, plain register.",
        "high": "Formal, business-like or legalistic register."}),
    "confidence": ("delivery", "Confidence and assertiveness of the writer", {
        "low": "Hesitant, uncertain, hedging or apologetic.",
        "medium": "Moderately assured.",
        "high": "Highly confident and assertive; states positions firmly."}),
    "politeness": ("delivery", "Politeness and courtesy toward the bank", {
        "low": "Discourteous or curt.",
        "medium": "Neutral courtesy.",
        "high": "Notably polite, courteous or deferential."}),
    "rudeness": ("delivery", "Rudeness, insults or hostility toward the bank or its staff", {
        "low": "No rudeness, insults or hostility.",
        "medium": "Some irritation, sarcasm or mildly hostile phrasing.",
        "high": "Clearly rude, insulting, abusive or hostile language."}),
    "emotion": ("delivery", "Intensity of emotional content (anger, distress, fear, frustration)", {
        "low": "Factual, little emotional language.",
        "medium": "Some emotional language.",
        "high": "Strongly emotional throughout."}),
    "entitlement": ("delivery", "Entitlement or demandingness", {
        "low": "Requests are modest or tentative.",
        "medium": "Requests are stated plainly as expectations.",
        "high": "Insistent demands; presents the outcome as owed regardless."}),
    # ---- structure ----
    "coherence": ("structure", "Logical coherence", {
        "low": "Points are hard to connect or contradict one another.",
        "medium": "Mostly coherent with some gaps or jumps.",
        "high": "Each point follows clearly from the previous ones."}),
    "chronology": ("structure", "Organisation and chronology of the account", {
        "low": "Disorganised; events are hard to place in order.",
        "medium": "Partly organised; the sequence can be reconstructed with effort.",
        "high": "Clearly organised with an easy-to-follow sequence of events."}),
    "argument": ("structure", "Argument quality: are claims linked to reasons?", {
        "low": "Assertions without reasons.",
        "medium": "Some claims are supported by stated reasons.",
        "high": "Claims are consistently backed by explicit reasons."}),
    "request_clarity": ("structure", "Clarity of what the writer wants the bank to do", {
        "low": "Unclear what outcome is wanted.",
        "medium": "The desired outcome can be inferred.",
        "high": "The desired outcome is stated explicitly and precisely."}),
    "concision": ("structure", "Concision", {
        "low": "Verbose, repetitive or padded.",
        "medium": "Some unnecessary material.",
        "high": "Concise; little or no unnecessary material."}),
    # ---- content (should be invariant under fact-preserving rewrites) ----
    "evidence": ("content", "Supporting evidence or documentation mentioned (statements, letters, dates, reference numbers)", {
        "low": "None mentioned.",
        "medium": "Some mentioned.",
        "high": "Extensive, specific documentation mentioned."}),
    "harm": ("content", "Severity of the financial harm described", {
        "low": "Little or no financial harm described.",
        "medium": "Moderate financial harm described.",
        "high": "Severe financial harm described."}),
    "bank_error": ("content", "How clearly a specific bank error or rule violation is alleged", {
        "low": "No specific bank error alleged.",
        "medium": "An error is implied or alleged vaguely.",
        "high": "A specific bank error or violation is clearly alleged."}),
    "money_demand": ("content", "Explicit request for money (refund, reversal, compensation)", {
        "low": "No request for money.",
        "medium": "Money is mentioned as a possible outcome.",
        "high": "An explicit request for a refund, reversal or compensation."}),
    "legal_basis": ("content", "Legal or regulatory basis cited (named laws, regulations, rights)", {
        "low": "None cited.",
        "medium": "General reference to rights or rules.",
        "high": "Specific laws or regulations named."}),
    "prior_attempts": ("content", "Prior attempts to resolve the issue with the bank", {
        "low": "None described.",
        "medium": "One attempt described.",
        "high": "Multiple attempts described."}),
}
FAMILY = {k: v[0] for k, v in ATTRIBUTES.items()}
LEVELS = ["low", "medium", "high"]


def build_attribute_questions() -> Dict[str, Any]:
    return {f"attr_{k}": {"type": "choice", "instructions": f"{INSTRUCTIONS}\n\nProperty: {label}.",
                          "criteria": dict(crit)}
            for k, (_, label, crit) in ATTRIBUTES.items()}


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
            a = qid[5:]
            p = ans["probabilities"]
            for lv in LEVELS:
                rec[f"{a}_p_{lv}"] = p.get(lv, 0.0)
            rec[f"{a}_choice"] = ans["choice"]
            rec[f"{a}_E"] = 0.5 * p.get("medium", 0.0) + p.get("high", 0.0)
        recs.append(rec)
    return pd.DataFrame(recs)


def pilot_texts(n_complaints: int = 200, seed: int = 42) -> pd.DataFrame:
    """Originals + guises for the same 200 complaints as the original Jev pilot
    (random.Random(42).sample over the sorted matched-guise case_ids)."""
    t = ja.exploratory_texts()
    ids = sorted(t.case_id.unique())
    keep = set(random.Random(seed).sample(ids, n_complaints))
    return t[t.case_id.isin(keep) & t.source.isin(["original", "guise"])].reset_index(drop=True)


def run(df: pd.DataFrame, out: Path, workers: int, dry_run: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    q = build_attribute_questions()
    n_unique = df.text_id.nunique()
    est_tokens = 450 + 175 * len(q)
    logger.info("%d texts (%d unique), %d attribute questions; est. cost $%.2f",
                len(df), n_unique, len(q), n_unique * est_tokens * jp.PRICE_PER_M_INPUT / 1e6)
    if dry_run:
        return
    jp.build_questions = lambda: q
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
    ap.add_argument("arm", choices=["pilot", "repeat", "holdout"])
    ap.add_argument("--tag", default="n1000")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.arm == "pilot":
        df, out = pilot_texts(), Path("jev_pilot_out/attributes_pilot")
    elif args.arm == "repeat":
        df = pilot_texts().drop_duplicates("text_id").sample(args.n, random_state=20261020)
        out = Path("jev_pilot_out/attributes_repeat")
    else:
        df, out = ja.holdout_texts(args.tag), Path("holdout_out/attributes")
    if args.limit:
        df = df.head(args.limit)
    run(df, out, args.workers, args.dry_run)


if __name__ == "__main__":
    main()
