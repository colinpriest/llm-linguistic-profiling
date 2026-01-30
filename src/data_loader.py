"""CFPB data ingestion, cleaning, and sampling."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import uuid
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional

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
        from pathlib import Path

        model_path = Path(__file__).parent.parent / "data" / "lid.176.ftz"
        if not model_path.exists():
            logger.info("Downloading fasttext language-id model (once)...")
            url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, str(model_path))
            logger.info("Model saved to %s", model_path)
        # Suppress fasttext's stderr warning about "\n" in input
        _ft_model = fasttext.load_model(str(model_path))
    return _ft_model


def _detect_lang_batch(texts: List[str]) -> List[str]:
    """Detect language for a batch of texts using fasttext."""
    model = _get_fasttext_model()
    # fasttext expects no newlines inside a single "document"
    cleaned = [t.replace("\n", " ") for t in texts]
    predictions = model.predict(cleaned)
    # predictions is (labels_list, probs_list); label looks like "__label__en"
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


def load_and_clean(
    data_path: str,
    num_complaints: int = 5000,
    seed: int = 42,
) -> List[Complaint]:
    """Load CFPB CSV, clean, and return sampled Complaint list."""

    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(
        data_path,
        usecols=list(COLUMN_MAP.keys()),
        dtype={"Complaint ID": int},
        low_memory=False,
    )
    df.rename(columns=COLUMN_MAP, inplace=True)
    initial = len(df)
    logger.info("Loaded %d raw records", initial)

    # Drop rows without narrative
    df = df.dropna(subset=["narrative"])
    logger.info("After dropping null narratives: %d", len(df))

    # Step 1: Whitespace normalisation
    df["narrative"] = df["narrative"].apply(lambda t: re.sub(r"\s+", " ", str(t)).strip())

    # Step 2: Length filter (300-2000 chars)
    df["_len"] = df["narrative"].str.len()
    df = df[(df["_len"] >= 300) & (df["_len"] <= 2000)].copy()
    logger.info("After length filter (300-2000): %d", len(df))

    # Step 3: Language detection – keep English only
    n_texts = len(df)
    try:
        logger.info("Running language detection via fasttext on %d texts...", n_texts)
        df["_lang"] = _detect_lang_batch(df["narrative"].tolist())
    except Exception as e:
        logger.warning("fasttext unavailable (%s), falling back to langid (slower)...", e)
        df["_lang"] = _detect_lang_fallback(df["narrative"].tolist())
    df = df[df["_lang"] == "en"].copy()
    logger.info("After English filter: %d", len(df))

    # Step 4: ASCII ratio filter >= 0.85
    def ascii_ratio(text: str) -> float:
        if not text:
            return 0.0
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        return ascii_chars / len(text)

    df["_ascii"] = df["narrative"].apply(ascii_ratio)
    df = df[df["_ascii"] >= 0.85].copy()
    logger.info("After ASCII ratio filter: %d", len(df))

    # Step 5: Null handling for product / issue
    df["product"] = df["product"].fillna("Unknown")
    df["issue"] = df["issue"].fillna("Unknown")

    # Step 6: Deduplication by Blake2b hash of narrative
    df["_hash"] = df["narrative"].apply(
        lambda t: hashlib.blake2b(t.encode(), digest_size=16).hexdigest()
    )
    df = df.drop_duplicates(subset=["_hash"], keep="first")
    logger.info("After deduplication: %d", len(df))

    # Step 7: ID generation
    df["case_id"] = [uuid.uuid4().hex for _ in range(len(df))]
    df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["year"] = pd.to_datetime(df["date_received"], errors="coerce").dt.strftime("%Y")

    # Category groupings
    df["product_category"] = df["product"].map(PRODUCT_CATEGORIES).fillna("Other")
    df["issue_category"] = df["issue"].map(ISSUE_CATEGORIES).fillna("Other")

    # Step 8: Stratified sampling
    df["_bucket"] = df["product_category"] + "|" + df["issue_category"]
    n = min(num_complaints, len(df))

    if n < len(df):
        logger.info("Stratified sampling %d from %d complaints", n, len(df))
        sampled = _stratified_sample(df, n, seed)
    else:
        sampled = df.copy()
        logger.info("Using all %d complaints (fewer than requested %d)", len(df), num_complaints)

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
        "Data pipeline complete: %d → %d complaints (%.1f%% retained)",
        initial, len(complaints), 100 * len(complaints) / initial,
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
