"""CFPB data ingestion, cleaning, and sampling.

The cleaning pipeline (Steps 1-6) is split into individually cached stages.
Each step saves a parquet checkpoint in ``data/.pipeline/``.  On subsequent
runs, completed steps are skipped by loading the most recent valid checkpoint.
Steps 7-8 (ID generation and sampling) are seed-dependent and always re-run.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

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
            _download_fasttext_model(model_path)
        try:
            _ft_model = fasttext.load_model(str(model_path))
        except Exception as exc:
            logger.warning("Fasttext model load failed (%s); re-downloading...", exc)
            if model_path.exists():
                model_path.unlink()
            _download_fasttext_model(model_path)
            _ft_model = fasttext.load_model(str(model_path))
    return _ft_model


def _download_fasttext_model(model_path: Path) -> None:
    logger.info("Downloading fasttext language-id model (once)...")
    url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(f"{model_path.suffix}.tmp")
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            with open(tmp_path, "wb") as fh:
                fh.write(response.read())
        os.replace(tmp_path, model_path)
        logger.info("Model saved to %s", model_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


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

# Product×issue combinations to exclude from analysis.
# These are dominated by non-bank actors (bureaus, third-party collectors)
# and are less informative about bank discretion in remedy decisions.
EXCLUDED_CATEGORY_PAIRS = {
    ("Credit Reporting", "Credit Report Errors"),
    ("Debt Collection", "Debt Disputes"),
}


# ---------------------------------------------------------------------------
# Issue severity scoring  (product base + issue modifier → low/medium/high)
# ---------------------------------------------------------------------------

# Tier 1: Product-based severity (base score)
_PRODUCT_SEVERITY: Dict[str, int] = {
    # High (3) – life-altering consequences
    "Mortgage": 3,
    "Student loan": 3,
    "Debt collection": 3,
    # Medium (2) – significant but typically recoverable
    "Credit card or prepaid card": 2,
    "Credit card": 2,
    "Vehicle loan or lease": 2,
    "Payday loan, title loan, or personal loan": 2,
    "Payday loan": 2,
    "Consumer Loan": 2,
    # Low (1) – usually smaller amounts, procedural
    "Checking or savings account": 1,
    "Bank account or service": 1,
    "Prepaid card": 1,
    "Money transfer, virtual currency, or money service": 1,
    "Money transfers": 1,
    "Credit reporting, credit repair services, or other personal consumer reports": 1,
    "Credit reporting": 1,
}

# Tier 2: Issue-based severity modifier
# +2 (Critical): threatened/actual loss of asset or service
# +1 (Elevated): active collection or credit damage
#  0 (Standard): billing/service disputes
# -1 (Lower):   informational/procedural
_ISSUE_MODIFIER: Dict[str, int] = {
    # +2  Critical – asset/service loss
    "Struggling to pay mortgage": 2,
    "Loan is not being serviced by the servicer you expected": 2,
    "Took or threatened to take negative or legal action": 2,
    "Problem when making payments": 2,
    "Dealing with your lender or servicer": 2,
    "Threatened with jail or sued": 2,
    "Closing your account": 2,

    # +1  Elevated – collection or credit damage
    "Attempts to collect debt not owed": 1,
    "Communication tactics": 1,
    "Incorrect information on your report": 1,
    "Problem with a credit reporting company's investigation into an existing problem": 1,
    "False statements or representation": 1,
    "Written notification about debt": 1,
    "Problem with a purchase shown on your statement": 1,
    "Problem with fraud alerts or security freezes": 1,
    "Getting a credit card": 1,

    # 0  Standard – billing / service disputes
    "Managing an account": 0,
    "Trouble during payment process": 0,
    "Problem with a lender or other company charging your account": 0,
    "Deposits and withdrawals": 0,
    "Problem caused by your funds being low": 0,
    "Opening an account": 0,
    "Closing an account": 0,
    "Problem with a company's investigation into an existing complaint": 0,
    "Fees or interest": 0,

    # -1  Lower – informational / procedural
    "Getting information about your loan": -1,
    "Improper use of your report": -1,
    "Problem with customer service": -1,
    "Other features, terms, or problems": -1,
    "Advertising and marketing, including promotional offers": -1,
    "Privacy and online account management": -1,
    "Confusing or missing disclosures": -1,
}

# Top ~50 product×issue overrides for combinations where the default
# composite score would be misleading.
_PRODUCT_ISSUE_OVERRIDE: Dict[Tuple[str, str], int] = {
    # Mortgage foreclosure situations → always high
    ("Mortgage", "Struggling to pay mortgage"): 5,
    ("Mortgage", "Trouble during payment process"): 4,
    ("Mortgage", "Closing on a mortgage"): 4,
    # Student loan wage garnishment
    ("Student loan", "Dealing with your lender or servicer"): 5,
    ("Student loan", "Struggling to repay your loan"): 4,
    # Vehicle repossession
    ("Vehicle loan or lease", "Managing the loan or lease"): 3,
    ("Vehicle loan or lease", "Struggling to pay your loan"): 4,
    # Debt collection threats → critical regardless of product
    ("Debt collection", "Took or threatened to take negative or legal action"): 5,
    ("Debt collection", "Attempts to collect debt not owed"): 4,
    ("Debt collection", "Threatened with jail or sued"): 5,
    # Credit card fraud
    ("Credit card or prepaid card", "Problem with a purchase shown on your statement"): 3,
    ("Credit card", "Problem with a purchase shown on your statement"): 3,
    # Low-stakes procedural
    ("Checking or savings account", "Managing an account"): 1,
    ("Checking or savings account", "Opening an account"): 1,
    ("Prepaid card", "Problem with a purchase shown on your statement"): 1,
}


def compute_issue_severity(product: str, issue: str) -> str:
    """Compute composite issue severity from raw product and issue fields.

    Returns ``"low"``, ``"medium"``, or ``"high"``.
    """
    # Check product×issue override first
    override = _PRODUCT_ISSUE_OVERRIDE.get((product, issue))
    if override is not None:
        score = override
    else:
        base = _PRODUCT_SEVERITY.get(product, 2)          # default medium
        modifier = _ISSUE_MODIFIER.get(issue, 0)           # default standard
        score = base + modifier

    # Clamp to [1, 5] then bucket
    score = max(1, min(5, score))
    if score <= 2:
        return "low"
    elif score <= 3:
        return "medium"
    else:
        return "high"


# ---------------------------------------------------------------------------
# Pipeline cache helpers
# ---------------------------------------------------------------------------

def _pipeline_dir(data_path: str) -> Path:
    """Return the directory used for step-level caches."""
    d = Path(data_path).parent / ".pipeline"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _step_cache_path(data_path: str, step: int, suffix: str = "") -> Path:
    name = f"step{step}{suffix}.parquet"
    return _pipeline_dir(data_path) / name


def _csv_mtime(data_path: str) -> float:
    return Path(data_path).stat().st_mtime


def _step_is_cached(data_path: str, step: int, suffix: str = "") -> bool:
    """True if step cache exists and is newer than the source CSV."""
    path = _step_cache_path(data_path, step, suffix)
    if not path.exists():
        return False
    return path.stat().st_mtime >= _csv_mtime(data_path)


def _save_step(df: pd.DataFrame, data_path: str, step: int, suffix: str = "") -> None:
    """Persist a DataFrame as the checkpoint for *step*."""
    path = _step_cache_path(data_path, step, suffix)
    df.to_parquet(path, index=False)
    logger.info("  ✓ Clean %d cached (%d rows) → %s", step, len(df), path.name)


def _load_step(data_path: str, step: int, suffix: str = "") -> pd.DataFrame:
    """Load a previously cached step checkpoint."""
    path = _step_cache_path(data_path, step, suffix)
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
        dtype={"Complaint ID": "Int64"},
        low_memory=False,
    )
    df.rename(columns=COLUMN_MAP, inplace=True)
    logger.info("  Loaded %d raw records", len(df))
    df = df.dropna(subset=["narrative"])
    df = df.dropna(subset=["complaint_id"])
    df["complaint_id"] = df["complaint_id"].astype(int)
    logger.info("  After dropping null narratives: %d", len(df))
    return df


def _step_1_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Step 1: Whitespace normalisation."""
    logger.info("[Data Clean 1/10] Whitespace normalisation (%d rows)...", len(df))
    df = df.copy()
    df["narrative"] = df["narrative"].apply(lambda t: re.sub(r"\s+", " ", str(t)).strip())
    return df


def _step_2_length(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: Length filter 300-1500 characters."""
    logger.info("[Data Clean 2/10] Length filter (%d rows)...", len(df))
    lengths = df["narrative"].str.len()
    df = df[(lengths >= 300) & (lengths <= 1500)].copy()
    logger.info("  After length filter: %d", len(df))
    return df


def _step_3_language(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: Language detection – keep English only."""
    n = len(df)
    logger.info("[Data Clean 3/10] Language detection (%d texts)...", n)
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
    logger.info("[Data Clean 4/10] ASCII ratio filter (%d rows)...", len(df))

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
    logger.info("[Data Clean 5/10] Null handling (%d rows)...", len(df))
    df = df.copy()
    df["product"] = df["product"].fillna("Unknown")
    df["issue"] = df["issue"].fillna("Unknown")
    return df


def _step_6_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Step 6: Deduplication by Blake2b hash of narrative."""
    logger.info("[Data Clean 6/10] Deduplication (%d rows)...", len(df))
    hashes = df["narrative"].apply(
        lambda t: hashlib.blake2b(t.encode(), digest_size=16).hexdigest()
    )
    df = df[~hashes.duplicated(keep="first")].copy()
    logger.info("  After deduplication: %d", len(df))
    return df


def _unique_word_ratio(text: str) -> float:
    """Fraction of unique words in text; degenerate repetitions → low ratio."""
    words = text.lower().split()
    if len(words) < 5:
        return 1.0
    return len(set(words)) / len(words)


def _step_6b_repetition(df: pd.DataFrame) -> pd.DataFrame:
    """Step 6b: Filter out degenerate repetitive narratives.

    Complaints like 'Account Name: XXXX' repeated hundreds of times have
    very low unique-word ratios and cause token-limit failures downstream.
    """
    logger.info("[Data Clean 7/10] Repetition filter (%d rows)...", len(df))
    ratios = df["narrative"].apply(_unique_word_ratio)
    df = df[ratios >= 0.20].copy()
    logger.info("  After repetition filter: %d", len(df))
    return df


def _step_6c_redaction(df: pd.DataFrame) -> pd.DataFrame:
    """Step 6c: Filter narratives where redaction markers dominate.

    The CFPB replaces personal information with 'XXXX' sequences. Some
    complaints are mostly redacted text, causing the LLM to reproduce
    thousands of X characters in its output and hit token limits.
    """
    MAX_X_RATIO = 0.20
    logger.info("[Data Clean 8/10] Redaction ratio filter (%d rows)...", len(df))
    x_ratios = df["narrative"].apply(
        lambda t: t.count("X") / len(t) if len(t) > 0 else 0.0
    )
    df = df[x_ratios < MAX_X_RATIO].copy()
    logger.info("  After redaction filter (X < %d%%): %d", int(MAX_X_RATIO * 100), len(df))
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# Steps 1-6c in order, each taking and returning a DataFrame
_CLEANING_STEPS = [
    (1, _step_1_whitespace),
    (2, _step_2_length),
    (3, _step_3_language),
    (4, _step_4_ascii),
    (5, _step_5_nulls),
    (6, _step_6_dedup),
    (7, _step_6b_repetition),
    (8, _step_6c_redaction),
]


def load_and_clean(
    data_path: str,
    num_complaints: int = 5000,
    seed: int = 42,
) -> List[Complaint]:
    """Load CFPB CSV, clean, and return sampled Complaint list.

    All cleaning steps are individually cached as parquet files in
    ``data/.pipeline/``.  On re-runs, the pipeline resumes from the
    latest valid checkpoint, skipping expensive steps like language
    detection (Step 3).

    Steps 9-10 (ID generation and stratified sampling) are cached with
    seed/sample-size suffixes so different configurations get their own
    checkpoints.
    """

    if num_complaints <= 0:
        raise ValueError("num_complaints must be a positive integer")
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"CSV data file not found: {data_path}")

    # --- Find the latest valid cached step ---
    resume_from = 0  # 0 = must start from raw CSV
    for step_num, _ in _CLEANING_STEPS:
        if _step_is_cached(data_path, step_num):
            resume_from = step_num
        else:
            break

    # --- Load starting point ---
    if resume_from > 0:
        logger.info("Data pipeline: resuming from cached clean step %d/10", resume_from)
        df = _load_step(data_path, resume_from)
    else:
        df = _step_0_load(data_path)

    # --- Run remaining steps, caching each ---
    for step_num, step_fn in _CLEANING_STEPS:
        if step_num <= resume_from:
            continue
        df = step_fn(df)
        _save_step(df, data_path, step_num)

    # --- Step 9: ID generation (seed-dependent, cached with seed suffix) ---
    s9_suffix = f"_seed{seed}"
    if _step_is_cached(data_path, 9, s9_suffix):
        df = _load_step(data_path, 9, s9_suffix)
    else:
        logger.info("[Data Clean 9/10] ID generation & date parsing (%d rows)...", len(df))
        rng = np.random.RandomState(seed)
        df = df.copy()
        df["case_id"] = [
            uuid.UUID(int=int(rng.randint(0, 2**31)) | (int(rng.randint(0, 2**31)) << 31)
                      | (int(rng.randint(0, 2**31)) << 62) | (int(rng.randint(0, 2**31)) << 93)).hex
            for _ in range(len(df))
        ]
        df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")
        df["year"] = df["date_received"].dt.strftime("%Y")
        df["date_received"] = df["date_received"].dt.strftime("%Y-%m-%d")
        df["date_received"] = df["date_received"].fillna("unknown")
        df["year"] = df["year"].fillna("unknown")

        # Category groupings
        df["product_category"] = df["product"].map(PRODUCT_CATEGORIES).fillna("Other")
        df["issue_category"] = df["issue"].map(ISSUE_CATEGORIES).fillna("Other")

        # Exclude product×issue pairs dominated by non-bank actors
        before = len(df)
        pair_key = df["product_category"] + "|" + df["issue_category"]
        excluded_keys = {f"{p}|{i}" for p, i in EXCLUDED_CATEGORY_PAIRS}
        df = df[~pair_key.isin(excluded_keys)].copy()
        excluded = before - len(df)
        if excluded > 0:
            logger.info(
                "  Excluded %d complaints from non-bank category pairs: %s",
                excluded, sorted(EXCLUDED_CATEGORY_PAIRS),
            )

        _save_step(df, data_path, 9, s9_suffix)

    # --- Step 10: Stratified sampling (seed + sample-size dependent) ---
    s10_suffix = f"_seed{seed}_n{num_complaints}"
    if _step_is_cached(data_path, 10, s10_suffix):
        sampled = _load_step(data_path, 10, s10_suffix)
    else:
        df["_bucket"] = df["product_category"] + "|" + df["issue_category"]
        n = min(num_complaints, len(df))

        if n < len(df):
            logger.info("[Data Clean 10/10] Stratified sampling %d from %d complaints", n, len(df))
            sampled = _stratified_sample(df, n, seed)
        else:
            sampled = df.copy()
            logger.info("[Data Clean 10/10] Using all %d complaints (fewer than requested %d)", len(df), num_complaints)

        # Drop temp columns
        sampled = sampled.drop(columns=[c for c in sampled.columns if c.startswith("_")], errors="ignore")
        _save_step(sampled, data_path, 10, s10_suffix)

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
            issue_severity=compute_issue_severity(row["product"], row["issue"]),
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
    if n <= 0:
        return df.head(0).copy()
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
