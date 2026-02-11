"""Debiased triage extraction consistency test.

Picks N complaints, loads their 36 persona-style rewrites from cache,
runs fact extraction on originals + rewrites, and analyzes:

1. Extraction consistency across persona rewrites (within each model)
2. Tier consistency across persona rewrites (within each model)
3. Cross-model extraction differences (originals + rewrites)
4. Cache hit/miss statistics + token usage & estimated cost

Persona rewrites MUST already exist in cache (from a prior experiment run).
The script will exit with an error if any rewrite is missing.

Usage:
    python test_debiased.py                       # 10 complaints, both models
    python test_debiased.py --num 5               # fewer complaints
    python test_debiased.py --models gpt-4o-mini  # single model
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Debiased extraction consistency test"
    )
    parser.add_argument("--num", type=int, default=10, help="Number of complaints to test")
    parser.add_argument("--models", nargs="+", default=["gpt-4o-mini", "gpt-5-mini"])
    parser.add_argument("--pool-size", type=int, default=5000, help="Must match --num-complaints from experiment run")
    parser.add_argument("--causal-pct", type=float, default=0.10, help="Must match --causal-sample-pct from experiment run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print(
            "ERROR: OPENAI_API_KEY not set. Add it to .env or environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    from src.cache import ResponseCache
    from src.data_loader import load_and_clean
    from src.debiased_triage import (
        FactSet,
        FactSetResponse,
        ReasonCategory,
        Tier,
        TriageDecision,
        build_stage_1_prompt,
        compare_factsets,
        heuristic_triage,
        parse_factset,
    )
    from src.llm_client import LLMClient
    from src.prompts import format_persona_rewrite_user
    from src.schemas import (
        PERSONA_GRID,
        LLMConfig,
        LLMRequest,
        PersonaRewriteResponse,
        persona_label,
    )

    # ------------------------------------------------------------------
    # Setup — replicate the experiment's data load + causal sample
    # ------------------------------------------------------------------
    import random

    cache = ResponseCache("./cache")

    # Must match the experiment's load_and_clean call exactly
    all_complaints = load_and_clean(
        "./data/complaints.csv",
        num_complaints=args.pool_size,
        seed=args.seed,
    )
    print(f"Loaded {len(all_complaints)} complaints from dataset")

    # Replicate experiment._step_4 causal sample selection
    causal_rng = random.Random(args.seed)
    n_causal = max(1, int(len(all_complaints) * args.causal_pct))
    causal_pool = causal_rng.sample(all_complaints, n_causal)
    print(f"Causal sample: {len(causal_pool)} complaints (matching experiment)")

    # Pick N from the causal pool for this test
    test_rng = random.Random(args.seed + 1)
    complaints = test_rng.sample(
        causal_pool, min(args.num, len(causal_pool))
    )
    print(f"Selected {len(complaints)} for test")

    config = LLMConfig(
        api_key=api_key,
        models=args.models,
        temperature=0.0,
        max_tokens=16384,
        timeout=120,
        max_retries=3,  # 3 attempts total = up to 2 retries on failure
        retry_delay=1.0,
    )
    client = LLMClient(config, cache, max_workers=args.threads)

    n_personas = len(PERSONA_GRID)  # 36
    n_complaints = len(complaints)
    complaint_meta = {c.case_id: (c.product, c.issue) for c in complaints}

    # Storage: model -> case_id -> FactSet/Decision
    orig_fs: Dict[str, Dict[str, Optional[FactSet]]] = defaultdict(dict)
    orig_dec: Dict[str, Dict[str, TriageDecision]] = defaultdict(dict)
    # model -> case_id -> persona_label -> FactSet/Decision
    rw_fs: Dict[str, Dict[str, Dict[str, Optional[FactSet]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    rw_dec: Dict[str, Dict[str, Dict[str, TriageDecision]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    fail_decision = TriageDecision(
        tier=Tier.NO_ACTION,
        primary_reason=ReasonCategory.INSUFFICIENT_EVIDENCE,
        explanation="Extraction failed.",
        confidence="low",
    )

    # Record cache state before we start
    hits_before = cache.hits
    misses_before = cache.misses

    for model in args.models:
        print(f"\n{'=' * 60}")
        print(f"Model: {model}")
        print(f"{'=' * 60}")

        # ==============================================================
        # Step 1: fact extraction on originals
        # ==============================================================
        requests: List[LLMRequest] = []
        narratives: List[str] = []
        for c in complaints:
            sys_msg, usr_msg = build_stage_1_prompt(
                c.narrative, c.product, c.issue
            )
            requests.append(
                LLMRequest(
                    request_id=LLMRequest.new_id(),
                    case_id=c.case_id,
                    model=model,
                    task="fact_extraction",
                    prompt_system=sys_msg,
                    prompt_user=usr_msg,
                    expected_schema=FactSetResponse,
                )
            )
            narratives.append(c.narrative)

        responses = client.execute_batch(
            requests, narratives,
            desc=f"Step 1 [{model}] Extract facts from originals",
        )
        for c, resp in zip(complaints, responses):
            if resp.success and resp.response_parsed is not None:
                fs = parse_factset(resp.response_parsed)
                orig_fs[model][c.case_id] = fs
                orig_dec[model][c.case_id] = heuristic_triage(fs)
            else:
                orig_fs[model][c.case_id] = None
                orig_dec[model][c.case_id] = fail_decision

        # ==============================================================
        # Step 2: load persona rewrites from cache (must already exist)
        # ==============================================================
        rewrite_texts: Dict[str, Dict[str, Optional[str]]] = defaultdict(dict)
        missing: List[str] = []
        for c in complaints:
            for persona in PERSONA_GRID:
                plabel = persona_label(**persona)
                # Cache key must match what the main experiment uses:
                # the full formatted user prompt (see experiment.py line 1056)
                narrative_key = format_persona_rewrite_user(
                    narrative=c.narrative,
                    age=persona["age"],
                    geo=persona["geo"],
                    edu=persona["edu"],
                    ses=persona["ses"],
                )
                cached = cache.get(model, "persona_rewrite", narrative_key)
                if cached is None:
                    missing.append(
                        f"  {model} / {c.case_id} / {plabel}"
                    )
                else:
                    try:
                        parsed = PersonaRewriteResponse(**cached)
                        rewrite_texts[c.case_id][plabel] = (
                            parsed.rewritten_complaint
                        )
                    except Exception as exc:
                        missing.append(
                            f"  {model} / {c.case_id} / {plabel}"
                            f" (parse error: {exc})"
                        )

        if missing:
            print(
                f"\nFATAL: {len(missing)} persona rewrite(s) not found "
                f"in cache for {model}.\n"
                f"Run the main experiment first (or a prior test run) to "
                f"generate persona rewrites.\n"
                f"Missing entries:",
                file=sys.stderr,
            )
            for m in missing[:20]:
                print(m, file=sys.stderr)
            if len(missing) > 20:
                print(
                    f"  ... and {len(missing) - 20} more",
                    file=sys.stderr,
                )
            sys.exit(1)

        n_loaded = sum(len(v) for v in rewrite_texts.values())
        print(f"  Loaded {n_loaded} persona rewrites from cache")

        # ==============================================================
        # Step 3: fact extraction on rewrites
        # ==============================================================
        requests = []
        narratives = []
        fact_keys: List[Tuple[str, str]] = []
        for case_id, personas in rewrite_texts.items():
            product, issue = complaint_meta[case_id]
            for plabel, text in personas.items():
                if text is None:
                    continue
                sys_msg, usr_msg = build_stage_1_prompt(text, product, issue)
                requests.append(
                    LLMRequest(
                        request_id=LLMRequest.new_id(),
                        case_id=case_id,
                        model=model,
                        task="fact_extraction",
                        prompt_system=sys_msg,
                        prompt_user=usr_msg,
                        expected_schema=FactSetResponse,
                    )
                )
                narratives.append(text)
                fact_keys.append((case_id, plabel))

        responses = client.execute_batch(
            requests, narratives,
            desc=f"Step 3 [{model}] Extract facts from rewrites ({len(requests)})",
        )
        rw_ok = rw_fail = 0
        for (case_id, plabel), resp in zip(fact_keys, responses):
            if resp.success and resp.response_parsed is not None:
                fs = parse_factset(resp.response_parsed)
                rw_fs[model][case_id][plabel] = fs
                rw_dec[model][case_id][plabel] = heuristic_triage(fs)
                rw_ok += 1
            else:
                rw_fs[model][case_id][plabel] = None
                rw_dec[model][case_id][plabel] = fail_decision
                rw_fail += 1

        # Summary for this model
        orig_ok = sum(1 for v in orig_fs[model].values() if v is not None)
        orig_fail = n_complaints - orig_ok
        print(f"  Originals:  {orig_ok} ok, {orig_fail} failed  (of {n_complaints})")
        print(f"  Rewrites:   {rw_ok} ok, {rw_fail} failed  (of {len(fact_keys)})")

    # ==================================================================
    # ANALYSIS 1: Extraction consistency across persona rewrites
    # ==================================================================
    print(f"\n{'=' * 90}")
    print("EXTRACTION CONSISTENCY: original vs 36 persona rewrites")
    print(
        "Each % = fraction of rewrites matching the original on that field"
    )
    print(f"{'=' * 90}")

    for model in args.models:
        print(f"\n--- {model} ---")
        hdr = (
            f"{'CASE_ID':<14} {'ORIG':>4} "
            f"{'TIER':>6} {'ENTITY':>7} {'NEXUS':>6} "
            f"{'CONDUCT':>8} {'HARM':>6} {'EVID':>6} {'N':>4}"
        )
        print(hdr)
        print("-" * len(hdr))

        model_tier: List[float] = []
        model_entity: List[float] = []
        model_nexus: List[float] = []
        model_conduct: List[float] = []
        model_harm: List[float] = []
        model_evid: List[float] = []

        for c in complaints:
            ofs = orig_fs[model].get(c.case_id)
            odec = orig_dec[model].get(c.case_id)
            if ofs is None or odec is None:
                continue

            n = 0
            tier_m = entity_m = nexus_m = conduct_m = harm_m = evid_m = 0
            for plabel, rfs in rw_fs[model].get(c.case_id, {}).items():
                rdec = rw_dec[model][c.case_id].get(plabel)
                if rfs is None or rdec is None:
                    continue
                n += 1
                diffs = compare_factsets(ofs, rfs)
                if rdec.tier == odec.tier:
                    tier_m += 1
                if "entity_type" not in diffs:
                    entity_m += 1
                if "bank_nexus" not in diffs:
                    nexus_m += 1
                if "conduct_diff" not in diffs:
                    conduct_m += 1
                if "harm_diff" not in diffs:
                    harm_m += 1
                if "evidence_strength_diff" not in diffs:
                    evid_m += 1

            if n == 0:
                continue

            rates = [x / n for x in (tier_m, entity_m, nexus_m, conduct_m, harm_m, evid_m)]
            model_tier.append(rates[0])
            model_entity.append(rates[1])
            model_nexus.append(rates[2])
            model_conduct.append(rates[3])
            model_harm.append(rates[4])
            model_evid.append(rates[5])

            print(
                f"{c.case_id[:12]:<14} T{odec.tier.value:<3} "
                f"{rates[0]:>6.0%} {rates[1]:>7.0%} {rates[2]:>6.0%} "
                f"{rates[3]:>8.0%} {rates[4]:>6.0%} {rates[5]:>6.0%} "
                f"{n:>4}"
            )

        if model_tier:
            avg = lambda xs: sum(xs) / len(xs)  # noqa: E731
            print("-" * len(hdr))
            print(
                f"{'AVERAGE':<14} {'':>4} "
                f"{avg(model_tier):>6.0%} {avg(model_entity):>7.0%} "
                f"{avg(model_nexus):>6.0%} {avg(model_conduct):>8.0%} "
                f"{avg(model_harm):>6.0%} {avg(model_evid):>6.0%}"
            )

    # ==================================================================
    # ANALYSIS 2: Where do extractions diverge? (worst-case diffs)
    # ==================================================================
    print(f"\n{'=' * 90}")
    print("DIVERGENCE EXAMPLES: complaints with lowest tier consistency")
    print(f"{'=' * 90}")

    for model in args.models:
        # Rank complaints by tier consistency (ascending)
        scored: List[Tuple[str, float, int, Dict[int, int]]] = []
        for c in complaints:
            odec = orig_dec[model].get(c.case_id)
            if odec is None:
                continue
            tier_counts: Dict[int, int] = defaultdict(int)
            n = 0
            for plabel, rdec in rw_dec[model].get(c.case_id, {}).items():
                if rdec is not None:
                    tier_counts[rdec.tier.value] += 1
                    n += 1
            if n == 0:
                continue
            tier_match_rate = tier_counts.get(odec.tier.value, 0) / n
            scored.append((c.case_id, tier_match_rate, odec.tier.value, dict(tier_counts)))

        scored.sort(key=lambda x: x[1])

        print(f"\n--- {model} (showing 3 most inconsistent) ---")
        for case_id, rate, orig_tier, counts in scored[:3]:
            dist = ", ".join(f"T{t}={c}" for t, c in sorted(counts.items()))
            print(f"  {case_id[:12]}  orig=T{orig_tier}  tier_match={rate:.0%}  distribution: {dist}")

            # Show what fields actually changed for a mismatched rewrite
            ofs = orig_fs[model].get(case_id)
            if ofs is None:
                continue
            for plabel, rfs in rw_fs[model].get(case_id, {}).items():
                rdec = rw_dec[model][case_id].get(plabel)
                if rfs is None or rdec is None:
                    continue
                if rdec.tier.value != orig_tier:
                    diffs = compare_factsets(ofs, rfs)
                    diff_fields = list(diffs.keys()) or ["(identical FactSet, different rule path)"]
                    print(
                        f"    {plabel}: T{rdec.tier.value} "
                        f"({rdec.primary_reason.value}) "
                        f"diffs={diff_fields}"
                    )
                    break  # one example per complaint is enough

    # ==================================================================
    # ANALYSIS 3: Cross-model comparison
    # ==================================================================
    if len(args.models) >= 2:
        m1, m2 = args.models[0], args.models[1]
        print(f"\n{'=' * 90}")
        print(f"CROSS-MODEL COMPARISON: {m1} vs {m2}")
        print(f"{'=' * 90}")

        # -- Originals --
        print(f"\nOn original complaints:")
        hdr2 = (
            f"{'CASE_ID':<14} {'T_' + m1[:8]:>10} {'T_' + m2[:8]:>10} "
            f"{'TIER':>5} {'ENTITY':>7} {'NEXUS':>6} {'CONDUCT':>8} {'HARM':>6}"
        )
        print(hdr2)
        print("-" * len(hdr2))

        n_comp = 0
        agree = defaultdict(int)
        for c in complaints:
            fs1 = orig_fs[m1].get(c.case_id)
            fs2 = orig_fs[m2].get(c.case_id)
            d1 = orig_dec[m1].get(c.case_id)
            d2 = orig_dec[m2].get(c.case_id)
            if fs1 is None or fs2 is None:
                continue
            n_comp += 1
            diffs = compare_factsets(fs1, fs2)
            checks = {
                "tier": d1.tier == d2.tier,
                "entity": "entity_type" not in diffs,
                "nexus": "bank_nexus" not in diffs,
                "conduct": "conduct_diff" not in diffs,
                "harm": "harm_diff" not in diffs,
            }
            for k, v in checks.items():
                if v:
                    agree[k] += 1

            yn = lambda b: "Y" if b else "N"  # noqa: E731
            print(
                f"{c.case_id[:12]:<14} {'T' + str(d1.tier.value):>10} "
                f"{'T' + str(d2.tier.value):>10} "
                f"{yn(checks['tier']):>5} {yn(checks['entity']):>7} "
                f"{yn(checks['nexus']):>6} {yn(checks['conduct']):>8} "
                f"{yn(checks['harm']):>6}"
            )

        if n_comp:
            print("-" * len(hdr2))
            print(
                f"{'AGREEMENT':<14} {'':>10} {'':>10} "
                f"{agree['tier']/n_comp:>5.0%} "
                f"{agree['entity']/n_comp:>7.0%} "
                f"{agree['nexus']/n_comp:>6.0%} "
                f"{agree['conduct']/n_comp:>8.0%} "
                f"{agree['harm']/n_comp:>6.0%}"
            )

        # -- Rewrites (aggregated) --
        print(f"\nOn persona rewrites (averaged across personas):")
        rw_tier_agree: List[float] = []
        for c in complaints:
            n = 0
            tier_match = 0
            for plabel in rw_fs[m1].get(c.case_id, {}):
                fs1 = rw_fs[m1][c.case_id].get(plabel)
                fs2 = rw_fs[m2].get(c.case_id, {}).get(plabel)
                d1 = rw_dec[m1][c.case_id].get(plabel)
                d2 = rw_dec[m2].get(c.case_id, {}).get(plabel)
                if fs1 is None or fs2 is None or d1 is None or d2 is None:
                    continue
                n += 1
                if d1.tier == d2.tier:
                    tier_match += 1
            if n > 0:
                rw_tier_agree.append(tier_match / n)

        if rw_tier_agree:
            avg_rw = sum(rw_tier_agree) / len(rw_tier_agree)
            print(
                f"  Cross-model tier agreement on rewrites: "
                f"{avg_rw:.0%} (averaged across {len(rw_tier_agree)} complaints)"
            )

    # ==================================================================
    # ANALYSIS 4: Consistency by persona dimension
    # ==================================================================
    print(f"\n{'=' * 90}")
    print("CONSISTENCY BY PERSONA DIMENSION")
    print("Average match rate (original vs rewrite) grouped by dimension value")
    print(f"{'=' * 90}")

    dimensions = [
        ("age", ["young", "old"]),
        ("geo", ["urban", "suburban", "rural"]),
        ("edu", ["high_school", "college"]),
        ("ses", ["low", "middle", "high"]),
    ]

    for model in args.models:
        print(f"\n--- {model} ---")
        for dim_key, dim_values in dimensions:
            print(f"\n  {dim_key.upper()}:")
            hdr_d = (
                f"    {'VALUE':<14} "
                f"{'TIER':>6} {'ENTITY':>7} {'NEXUS':>6} "
                f"{'CONDUCT':>8} {'HARM':>6} {'EVID':>6} {'N':>6}"
            )
            print(hdr_d)
            print(f"    {'-' * (len(hdr_d) - 4)}")

            for dval in dim_values:
                tier_m = entity_m = nexus_m = conduct_m = harm_m = evid_m = 0
                n_total = 0

                for c in complaints:
                    ofs = orig_fs[model].get(c.case_id)
                    odec = orig_dec[model].get(c.case_id)
                    if ofs is None or odec is None:
                        continue

                    for persona in PERSONA_GRID:
                        if persona[dim_key] != dval:
                            continue
                        plabel = persona_label(**persona)
                        rfs = rw_fs[model].get(c.case_id, {}).get(plabel)
                        rdec = rw_dec[model].get(c.case_id, {}).get(plabel)
                        if rfs is None or rdec is None:
                            continue
                        n_total += 1
                        diffs = compare_factsets(ofs, rfs)
                        if rdec.tier == odec.tier:
                            tier_m += 1
                        if "entity_type" not in diffs:
                            entity_m += 1
                        if "bank_nexus" not in diffs:
                            nexus_m += 1
                        if "conduct_diff" not in diffs:
                            conduct_m += 1
                        if "harm_diff" not in diffs:
                            harm_m += 1
                        if "evidence_strength_diff" not in diffs:
                            evid_m += 1

                if n_total == 0:
                    continue
                rates = [
                    x / n_total
                    for x in (tier_m, entity_m, nexus_m, conduct_m, harm_m, evid_m)
                ]
                print(
                    f"    {dval:<14} "
                    f"{rates[0]:>6.0%} {rates[1]:>7.0%} {rates[2]:>6.0%} "
                    f"{rates[3]:>8.0%} {rates[4]:>6.0%} {rates[5]:>6.0%} "
                    f"{n_total:>6}"
                )

    # ==================================================================
    # ANALYSIS 5: Cross-model agreement by persona dimension
    # ==================================================================
    if len(args.models) >= 2:
        m1, m2 = args.models[0], args.models[1]
        print(f"\n{'=' * 90}")
        print(f"CROSS-MODEL AGREEMENT BY PERSONA DIMENSION: {m1} vs {m2}")
        print("Agreement rate on rewrites grouped by dimension value")
        print(f"{'=' * 90}")

        for dim_key, dim_values in dimensions:
            print(f"\n  {dim_key.upper()}:")
            hdr_d = (
                f"    {'VALUE':<14} "
                f"{'TIER':>6} {'ENTITY':>7} {'NEXUS':>6} "
                f"{'CONDUCT':>8} {'HARM':>6} {'N':>6}"
            )
            print(hdr_d)
            print(f"    {'-' * (len(hdr_d) - 4)}")

            for dval in dim_values:
                tier_a = entity_a = nexus_a = conduct_a = harm_a = 0
                n_total = 0

                for c in complaints:
                    for persona in PERSONA_GRID:
                        if persona[dim_key] != dval:
                            continue
                        plabel = persona_label(**persona)
                        fs1 = rw_fs[m1].get(c.case_id, {}).get(plabel)
                        fs2 = rw_fs[m2].get(c.case_id, {}).get(plabel)
                        d1 = rw_dec[m1].get(c.case_id, {}).get(plabel)
                        d2 = rw_dec[m2].get(c.case_id, {}).get(plabel)
                        if fs1 is None or fs2 is None or d1 is None or d2 is None:
                            continue
                        n_total += 1
                        diffs = compare_factsets(fs1, fs2)
                        if d1.tier == d2.tier:
                            tier_a += 1
                        if "entity_type" not in diffs:
                            entity_a += 1
                        if "bank_nexus" not in diffs:
                            nexus_a += 1
                        if "conduct_diff" not in diffs:
                            conduct_a += 1
                        if "harm_diff" not in diffs:
                            harm_a += 1

                if n_total == 0:
                    continue
                rates = [
                    x / n_total
                    for x in (tier_a, entity_a, nexus_a, conduct_a, harm_a)
                ]
                print(
                    f"    {dval:<14} "
                    f"{rates[0]:>6.0%} {rates[1]:>7.0%} {rates[2]:>6.0%} "
                    f"{rates[3]:>8.0%} {rates[4]:>6.0%} "
                    f"{n_total:>6}"
                )

    # ==================================================================
    # Cache statistics
    # ==================================================================
    hits_after = cache.hits
    misses_after = cache.misses
    session_hits = hits_after - hits_before
    session_misses = misses_after - misses_before
    session_total = session_hits + session_misses

    print(f"\n{'=' * 90}")
    print("CACHE STATISTICS")
    print(f"{'=' * 90}")
    print(f"  Fact extraction + rewrite calls this session: {session_total}")
    print(f"  Cache hits  (re-used):  {session_hits:>6}")
    print(f"  Cache misses (new LLM): {session_misses:>6}")
    if session_total > 0:
        print(f"  Hit rate: {session_hits / session_total:.1%}")
    print(f"\n  Cumulative: {cache.stats_summary()}")

    # ==================================================================
    # Token usage and estimated cost
    # ==================================================================
    usage = client.usage_summary()
    totals = usage.get("totals", {})
    per_model = usage.get("per_model", {})

    print(f"\n{'=' * 90}")
    print("TOKEN USAGE & ESTIMATED COST")
    print(f"{'=' * 90}")

    if per_model:
        hdr_tok = (
            f"  {'MODEL':<16} {'REQUESTS':>9} {'PROMPT':>12} "
            f"{'COMPLETION':>12} {'TOTAL':>12} {'EST. COST':>12}"
        )
        print(hdr_tok)
        print(f"  {'-' * (len(hdr_tok) - 2)}")
        for mname, mstats in per_model.items():
            p_tok = mstats.get("prompt_tokens", 0)
            c_tok = mstats.get("completion_tokens", 0)
            t_tok = p_tok + c_tok
            reqs = mstats.get("requests", 0)
            cost = mstats.get("estimated_cost_usd")
            cost_str = f"${cost:,.4f}" if cost is not None else "n/a"
            print(
                f"  {mname:<16} {reqs:>9,} {p_tok:>12,} "
                f"{c_tok:>12,} {t_tok:>12,} {cost_str:>12}"
            )
        print(f"  {'-' * (len(hdr_tok) - 2)}")
        grand_p = totals.get("prompt_tokens", 0)
        grand_c = totals.get("completion_tokens", 0)
        grand_t = totals.get("total_tokens", 0)
        grand_r = totals.get("requests", 0)
        grand_cost = totals.get("estimated_cost_usd")
        cost_str = f"${grand_cost:,.4f}" if grand_cost is not None else "n/a"
        print(
            f"  {'TOTAL':<16} {grand_r:>9,} {grand_p:>12,} "
            f"{grand_c:>12,} {grand_t:>12,} {cost_str:>12}"
        )
    else:
        print("  No API calls made (all responses served from cache).")


if __name__ == "__main__":
    main()
