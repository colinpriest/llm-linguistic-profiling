"""Load source-rewrite pairs from the project's LLM response cache.

This module imports from the main ``src/`` package to reconstruct
matched-guise pairs without making any new API calls.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _find_latest_file(directory: Path, pattern: str) -> Optional[Path]:
    """Return the most-recently-modified file matching *pattern* in *directory*."""
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _extract_narr_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """If *df* has case_id + a narrative column, return a two-column DataFrame."""
    for col in ("narrative", "consumer_complaint_narrative", "Consumer complaint narrative"):
        if col in df.columns and "case_id" in df.columns:
            return (
                df.rename(columns={col: "narrative"})[["case_id", "narrative"]]
                .drop_duplicates(subset="case_id")
                .reset_index(drop=True)
            )
        # Also check for 'Complaint ID' as an id column
        if col in df.columns:
            for id_col in ("Complaint ID", "complaint_id"):
                if id_col in df.columns:
                    return (
                        df.rename(columns={id_col: "case_id", col: "narrative"})[["case_id", "narrative"]]
                        .drop_duplicates(subset="case_id")
                        .reset_index(drop=True)
                    )
    return None


def _parquet_sort_key(p: Path) -> tuple:
    """Sort parquets so step10 > step9 > ... and larger n-values first."""
    name = p.stem
    m = re.search(r"step(\d+)", name)
    step = int(m.group(1)) if m else 0
    m2 = re.search(r"_n(\d+)", name)
    n_val = int(m2.group(1)) if m2 else 0
    return (-step, -n_val)


def _load_narratives(
    needed_ids: set,
    run_dir: Optional[Path] = None,
    data_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Return a DataFrame with at least ``case_id`` and ``narrative`` columns.

    *needed_ids* is the set of case_ids that must be present.  Sources
    are tried in priority order and the first one with adequate coverage
    is returned.

    Strategy:
    1. Try ``text_features_*.csv`` in *run_dir* (has ``case_id`` + ``narrative``).
    2. Try pipeline parquets under ``data/.pipeline/`` (later steps first).
    3. Fall back to *data_path* (raw complaints CSV).
    """
    # Strategy 1: text_features CSV from the experiment run
    if run_dir:
        tf = _find_latest_file(run_dir, "text_features_*.csv")
        if tf:
            logger.info("Loading narratives from %s", tf)
            df = pd.read_csv(tf, usecols=lambda c: c in ("case_id", "narrative"))
            ndf = _extract_narr_df(df)
            if ndf is not None:
                hits = needed_ids & set(ndf["case_id"])
                if hits:
                    logger.info("  text_features covers %d / %d case_ids", len(hits), len(needed_ids))
                    if len(hits) >= len(needed_ids) * 0.5:
                        return ndf

    # Strategy 2: pipeline parquets (try later steps first — step10/step9
    # have final case_ids; step1-step8 are intermediate and usually don't match)
    pipeline_dir = Path("data/.pipeline")
    if pipeline_dir.is_dir():
        parquets = sorted(pipeline_dir.glob("*.parquet"), key=_parquet_sort_key)
        for pq in parquets:
            logger.info("Trying pipeline parquet: %s", pq)
            try:
                df = pd.read_parquet(pq)
            except Exception:
                continue
            ndf = _extract_narr_df(df)
            if ndf is not None:
                hits = needed_ids & set(ndf["case_id"])
                logger.info("  covers %d / %d case_ids", len(hits), len(needed_ids))
                if len(hits) >= len(needed_ids) * 0.5:
                    return ndf

    # Strategy 3: raw complaints CSV
    if data_path and data_path.is_file():
        logger.info("Loading narratives from raw CSV: %s", data_path)
        df = pd.read_csv(data_path, low_memory=False)
        ndf = _extract_narr_df(df)
        if ndf is not None:
            return ndf

    raise FileNotFoundError(
        "Could not locate narrative text. Provide --data-path or ensure "
        "run_dir contains text_features_*.csv."
    )


def load_pairs_from_cache(
    run_dir: Path,
    cache_dir: Path = Path("./cache"),
    data_path: Optional[Path] = None,
    models: Optional[List[str]] = None,
    max_pairs: Optional[int] = None,
) -> pd.DataFrame:
    """Reconstruct source→rewrite pairs from the experiment cache.

    Parameters
    ----------
    run_dir : Path
        Experiment output directory (e.g. ``output/20260216_115325``).
    cache_dir : Path
        LLM response cache root.
    data_path : Path | None
        Fallback path to raw complaints CSV.
    models : list[str] | None
        Model names to include.  If *None*, auto-detect from decisions CSV.
    max_pairs : int | None
        If set, limit to approximately this many pairs by reducing the
        number of case_ids iterated (avoids loading the full 144k cache).

    Returns
    -------
    pd.DataFrame
        Columns: ``complaint_id, rewrite_id, model_name, persona_label,
        source_text, rewrite_text``.
    """
    # Lazy imports so the script also works standalone (--input mode)
    from src.cache import ResponseCache
    from src.prompts import format_persona_rewrite_user
    from src.schemas import PERSONA_GRID, PersonaRewriteResponse, persona_label

    # Load persona decisions to know which (case_id, model, persona) exist
    dec_csv = _find_latest_file(run_dir, "results_persona_decisions_*.csv")
    if dec_csv is None:
        raise FileNotFoundError(
            f"No results_persona_decisions_*.csv found in {run_dir}"
        )
    logger.info("Loading decisions from %s", dec_csv)
    decisions = pd.read_csv(dec_csv)

    if models is None:
        models = decisions["model"].unique().tolist()

    case_ids = set(decisions["case_id"].unique())

    # Load narrative texts
    narratives_df = _load_narratives(case_ids, run_dir, data_path)
    narratives_df = narratives_df[narratives_df["case_id"].isin(case_ids)]
    narr_map = dict(zip(narratives_df["case_id"], narratives_df["narrative"]))

    # Limit case_ids early if max_pairs requested (each case_id produces
    # len(models) * 36 pairs, so we only need a fraction of the complaints)
    if max_pairs is not None:
        n_personas = len(PERSONA_GRID)
        max_cases = max(1, max_pairs // (len(models) * n_personas) + 1)
        if max_cases < len(narr_map):
            limited_ids = list(narr_map.keys())[:max_cases]
            narr_map = {k: narr_map[k] for k in limited_ids}
            logger.info("Limiting to %d case_ids for --max-pairs %d", max_cases, max_pairs)

    logger.info(
        "Found %d case_ids in decisions, %d with narratives",
        len(case_ids), len(narr_map),
    )

    cache = ResponseCache(cache_dir=str(cache_dir))
    n_personas = len(PERSONA_GRID)
    total_lookups = len(models) * len(narr_map) * n_personas

    rows = []
    missing = 0
    with tqdm(total=total_lookups, desc="Loading pairs from cache") as pbar:
        for model in models:
            for case_id, narrative in narr_map.items():
                for persona in PERSONA_GRID:
                    plabel = persona_label(**persona)

                    narrative_key = format_persona_rewrite_user(
                        narrative,
                        persona["age"],
                        persona["geo"],
                        persona["edu"],
                        persona["ses"],
                    )

                    cached = cache.get(model, "persona_rewrite", narrative_key)
                    if cached is None:
                        missing += 1
                        pbar.update(1)
                        continue

                    try:
                        parsed = PersonaRewriteResponse(**cached)
                        rewrite_text = parsed.rewritten_complaint
                    except Exception:
                        missing += 1
                        pbar.update(1)
                        continue

                    rows.append({
                        "complaint_id": str(case_id),
                        "rewrite_id": f"{case_id}__{model}__{plabel}",
                        "model_name": model,
                        "persona_label": plabel,
                        "source_text": narrative,
                        "rewrite_text": rewrite_text,
                    })
                    pbar.update(1)

    if missing:
        logger.warning("Skipped %d cache misses", missing)

    logger.info("Loaded %d source->rewrite pairs from cache", len(rows))
    return pd.DataFrame(rows)
