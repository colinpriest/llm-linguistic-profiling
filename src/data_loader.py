"""CFPB data ingestion, cleaning, and sampling.

The cleaning pipeline (Steps 1-6) is split into individually cached stages.
Each step saves a parquet checkpoint in ``data/.pipeline/``.  On subsequent
runs, completed steps are skipped by loading the most recent valid checkpoint.
Steps 7-8 (ID generation and sampling) are seed-dependent and always re-run.
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.schemas import Complaint

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language detection — fasttext (fast, C-backed) with langid fallback
# ---------------------------------------------------------------------------

_ft_model = None


def _get_fasttext_model():
    """Lazy-load the fasttext language-identification model."""
    global _ft_model
    if _ft_model is None:
        import fasttext
        import urllib.request

        model_path = Path(__file__).parent.parent / "data" / "lid.176.ftz"
        if not model_path.exists():
            logger.info("Downloading fasttext language-id model (once)...")
            url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, str(model_path))
            logger.info("Model saved to %s", model_path)
        _ft_model = fasttext.load_model(str(model_path))
    return _ft_model


def _detect_lang_batch(texts: List[str]) -> List[str]:
    """Detect language for a batch of texts using fasttext."""
    model = _get_fasttext_model()
    cleaned = [t.replace("\n", " ") for t in texts]
    predictions = model.predict(cleaned)
    return [lbl[0].replace("__label__", "") for lbl in predictions[0]]


def _detect_lang_fallback(texts: List[str]) -> List[str]:
    """Fallback: langid (slow, pure Python)."""
    import langid
    return [langid.classify(t)[0] for t in texts]


# ---------------------------------------------------------------------------
# Column mapping: CFPB name -> internal name
# ---------------------------------------------------------------------------

COLUMN_MAP = {
    "Consumer complaint narrative": "narrative",
    "Product": "product",
    "Issue": "issue",
    "Date received": "date_received",
    "Complaint ID": "complaint_id",
}

# ---------------------------------------------------------------------------
# Product & issue category groupings
# ---------------------------------------------------------------------------

PRODUCT_CATEGORIES = {
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit Reporting",
    "Credit reporting": "Credit Reporting",
    "Debt collection": "Debt Collection",
    "Mortgage": "Lending",
    "Student loan": "Lending",
    "Vehicle loan or lease": "Lending",
    "Payday loan, title loan, or personal loan": "Lending",
    "Payday loan": "Lending",
    "Consumer Loan": "Lending",
    "Checking or savings account": "Banking",
    "Bank account or service": "Banking",
    "Prepaid card": "Banking",
    "Credit card or prepaid card": "Banking",
    "Credit card": "Banking",
    "Money transfer, virtual currency, or money service": "Banking",
    "Money transfers": "Banking",
}

ISSUE_CATEGORIES = {
    "Incorrect information on your report": "Credit Report Errors",
    "Improper use of your report": "Credit Report Errors",
    "Problem with a credit reporting company's investigation into an existing problem": "Credit Report Errors",
    "Attempts to collect debt not owed": "Debt Disputes",
    "Written notification about debt": "Debt Disputes",
    "Communication tactics": "Debt Disputes",
    "Took or threatened to take negative or legal action": "Debt Disputes",
    "Managing an account": "Account Management",
    "Problem with a purchase shown on your statement": "Account Management",
    "Struggling to pay mortgage": "Payment Issues",
    "Trouble during payment process": "Payment Issues",
    "Problem with a lender or other company charging your account": "Payment Issues",
}


# ---------------------------------------------------------------------------
# Pipeline cache helpers
# ---------------------------------------------------------------------------

def _pipeline_dir(data_path: str) -> Path:
    """Return the directory used for step-level caches."""
    d = Path(data_path).parent / ".pipeline"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _step_cache_path(data_path: str, step: int) -> Path:
    return _pipeline_dir(data_path) / f"step{step}.parquet"


def _csv_mtime(data_path: str) -> float:
    return Path(data_path).stat().st_mtime


def _step_is_cached(data_path: str, step: int) -> bool:
    """True if step cache exists and is newer than the source CSV."""
    path = _step_cache_path(data_path, step)
    if not path.exists():
        return False
    return path.stat().st_mtime >= _csv_mtime(data_path)


def _save_step(df: pd.DataFrame, data_path: str, step: int) -> None:
    """Persist a DataFrame as the checkpoint for *step*."""
    path = _step_cache_path(data_path, step)
    df.to_parquet(path, index=False)
    logger.info("  ✓ Clean %d cached (%d rows) → %s", step, len(df), path.name)


def _load_step(data_path: str, step: int) -> pd.DataFrame:
    """Load a previously cached step checkpoint."""
    path = _step_cache_path(data_path, step)
    df = pd.read_parquet(path)
    logger.info("  ✓ Clean %d loaded from cache (%d rows)", step, len(df))
    return df


# ---------------------------------------------------------------------------
# Individual pipeline steps (1-6: deterministic and cacheable)
# ---------------------------------------------------------------------------

def _step_0_load(data_path: str) -> pd.DataFrame:
    """Load raw CSV and drop rows without narrative."""
    logger.info("[Data Clean] Loading raw CSV...")
    df = pd.read_csv(
        data_path,
        usecols=list(COLUMN_MAP.keys()),
        dtype={"Complaint ID": int},
        low_memory=False,
    )
    df.rename(columns=COLUMN_MAP, inplace=True)
    logger.info("  Loaded %d raw records", len(df))
    df = df.dropna(subset=["narrative"])
    logger.info("  After dropping null narratives: %d", len(df))
    return df


def _step_1_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Step 1: Whitespace normalisation."""
    logger.info("[Data Clean 1/8] Whitespace normalisation (%d rows)...", len(df))
    df = df.copy()
    df["narrative"] = df["narrative"].apply(lambda t: re.sub(r"\s+", " ", str(t)).strip())
    return df


def _step_2_length(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: Length filter 300-2000 characters."""
    logger.info("[Data Clean 2/8] Length filter (%d rows)...", len(df))
    lengths = df["narrative"].str.len()
    df = df[(lengths >= 300) & (lengths <= 2000)].copy()
    logger.info("  After length filter: %d", len(df))
    return df


def _step_3_language(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: Language detection – keep English only."""
    n = len(df)
    logger.info("[Data Clean 3/8] Language detection (%d texts)...", n)
    try:
        df = df.copy()
        df["_lang"] = _detect_lang_batch(df["narrative"].tolist())
    except Exception as e:
        logger.warning("  fasttext unavailable (%s), falling back to langid...", e)
        df = df.copy()
        df["_lang"] = _detect_lang_fallback(df["narrative"].tolist())
    df = df[df["_lang"] == "en"].drop(columns=["_lang"]).copy()
    logger.info("  After English filter: %d", len(df))
    return df


def _step_4_ascii(df: pd.DataFrame) -> pd.DataFrame:
    """Step 4: ASCII ratio filter >= 0.85."""
    logger.info("[Data Clean 4/8] ASCII ratio filter (%d rows)...", len(df))

    def ascii_ratio(text: str) -> float:
        if not text:
            return 0.0
        return sum(1 for c in text if ord(c) < 128) / len(text)

    ratios = df["narrative"].apply(ascii_ratio)
    df = df[ratios >= 0.85].copy()
    logger.info("  After ASCII filter: %d", len(df))
    return df


def _step_5_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Step 5: Fill missing product / issue with 'Unknown'."""
    logger.info("[Data Clean 5/8] Null handling (%d rows)...", len(df))
    df = df.copy()
    df["product"] = df["product"].fillna("Unknown")
    df["issue"] = df["issue"].fillna("Unknown")
    return df


def _step_6_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Step 6: Deduplication by Blake2b hash of narrative."""
    logger.info("[Data Clean 6/8] Deduplication (%d rows)...", len(df))
    hashes = df["narrative"].apply(
        lambda t: hashlib.blake2b(t.encode(), digest_size=16).hexdigest()
    )
    df = df[~hashes.duplicated(keep="first")].copy()
    logger.info("  After deduplication: %d", len(df))
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# Steps 1-6 in order, each taking and returning a DataFrame
_CLEANING_STEPS = [
    (1, _step_1_whitespace),
    (2, _step_2_length),
    (3, _step_3_language),
    (4, _step_4_ascii),
    (5, _step_5_nulls),
    (6, _step_6_dedup),
]


def load_and_clean(
    data_path: str,
    num_complaints: int = 5000,
    seed: int = 42,
) -> List[Complaint]:
    """Load CFPB CSV, clean, and return sampled Complaint list.

    Steps 1-6 are individually cached as parquet files in
    ``data/.pipeline/``.  On re-runs, the pipeline resumes from the
    latest valid checkpoint, skipping expensive steps like language
    detection (Step 3).

    Steps 7-8 (ID generation and stratified sampling) depend on the
    random seed and are always re-run.
    """

    # --- Find the latest valid cached step ---
    resume_from = 0  # 0 = must start from raw CSV
    for step_num, _ in _CLEANING_STEPS:
        if _step_is_cached(data_path, step_num):
            resume_from = step_num
        else:
            break

    # --- Load starting point ---
    if resume_from > 0:
        logger.info("Data pipeline: resuming from cached clean step %d/8", resume_from)
        df = _load_step(data_path, resume_from)
    else:
        df = _step_0_load(data_path)

    # --- Run remaining steps, caching each ---
    for step_num, step_fn in _CLEANING_STEPS:
        if step_num <= resume_from:
            continue
        df = step_fn(df)
        _save_step(df, data_path, step_num)

    # --- Step 7: ID generation (seed-dependent, always re-run) ---
    logger.info("[Data Clean 7/8] ID generation & date parsing (%d rows)...", len(df))
    rng = np.random.RandomState(seed)
    df = df.copy()
    df["case_id"] = [
        uuid.UUID(int=int(rng.randint(0, 2**31)) | (int(rng.randint(0, 2**31)) << 31)
                  | (int(rng.randint(0, 2**31)) << 62) | (int(rng.randint(0, 2**31)) << 93)).hex
        for _ in range(len(df))
    ]
    df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["year"] = pd.to_datetime(df["date_received"], errors="coerce").dt.strftime("%Y")

    # Category groupings
    df["product_category"] = df["product"].map(PRODUCT_CATEGORIES).fillna("Other")
    df["issue_category"] = df["issue"].map(ISSUE_CATEGORIES).fillna("Other")

    # --- Step 8: Stratified sampling ---
    df["_bucket"] = df["product_category"] + "|" + df["issue_category"]
    n = min(num_complaints, len(df))

    if n < len(df):
        logger.info("[Data Clean 8/8] Stratified sampling %d from %d complaints", n, len(df))
        sampled = _stratified_sample(df, n, seed)
    else:
        sampled = df.copy()
        logger.info("[Data Clean 8/8] Using all %d complaints (fewer than requested %d)", len(df), num_complaints)

    # Drop temp columns
    sampled = sampled.drop(columns=[c for c in sampled.columns if c.startswith("_")], errors="ignore")

    # Convert to Complaint objects
    complaints = [
        Complaint(
            case_id=row["case_id"],
            complaint_id=int(row["complaint_id"]),
            narrative=row["narrative"],
            product=row["product"],
            issue=row["issue"],
            date_received=str(row["date_received"]),
            year=str(row["year"]),
            product_category=row["product_category"],
            issue_category=row["issue_category"],
        )
        for _, row in sampled.iterrows()
    ]

    logger.info(
        "Data pipeline complete: %d cleaned → %d sampled complaints",
        len(df), len(complaints),
    )
    return complaints


def _stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Proportional stratified sample by _bucket column."""
    rng = np.random.RandomState(seed)
    bucket_counts = df["_bucket"].value_counts()
    bucket_fracs = bucket_counts / len(df)

    frames = []
    remaining = n
    buckets = list(bucket_fracs.index)

    for i, bucket in enumerate(buckets):
        bucket_df = df[df["_bucket"] == bucket]
        if i == len(buckets) - 1:
            k = remaining
        else:
            k = max(1, int(round(bucket_fracs[bucket] * n)))
        k = min(k, len(bucket_df), remaining)
        if k > 0:
            frames.append(bucket_df.sample(n=k, random_state=rng))
            remaining -= k
        if remaining <= 0:
            break

    return pd.concat(frames, ignore_index=True)
