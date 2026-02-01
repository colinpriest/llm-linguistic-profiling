"""Text feature extraction pipeline."""

from __future__ import annotations

import logging
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.schemas import TextFeatures

logger = logging.getLogger(__name__)

# Lazy-loaded heavy modules
_nltk_ready = False
_vader_analyzer = None
_lda_model = None
_vectorizer = None


def _ensure_nltk() -> None:
    global _nltk_ready
    if _nltk_ready:
        return
    import nltk
    for resource in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)
    _nltk_ready = True


def _get_vader():
    global _vader_analyzer
    if _vader_analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer


# ---------------------------------------------------------------------------
# Individual feature extractors
# ---------------------------------------------------------------------------

def _length_features(text: str) -> Dict[str, Any]:
    _ensure_nltk()
    from nltk.tokenize import sent_tokenize

    words = text.split()
    sentences = sent_tokenize(text)
    word_count = len(words)
    sentence_count = max(len(sentences), 1)

    return {
        "word_count": word_count,
        "char_count": len(text),
        "sentence_count": sentence_count,
        "avg_word_length": (sum(len(w) for w in words) / max(word_count, 1)),
        "avg_sentence_length": word_count / sentence_count,
    }


def _complexity_features(text: str) -> Dict[str, float]:
    import textstat
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
            "gunning_fog": textstat.gunning_fog(text),
            "smog_index": textstat.smog_index(text),
            "ari": textstat.automated_readability_index(text),
        }


def _vocabulary_features(text: str) -> Dict[str, float]:
    words = [w.lower() for w in text.split()]
    word_count = max(len(words), 1)
    unique = set(words)
    freq = Counter(words)
    hapax = sum(1 for w, c in freq.items() if c == 1)

    ttr = len(unique) / word_count

    # MTLD and HD-D via lexicalrichness (graceful fallback)
    mtld_val = 0.0
    hdd_val = 0.0
    try:
        from lexicalrichness import LexicalRichness
        lr = LexicalRichness(text)
        try:
            mtld_val = lr.mtld()
        except Exception:
            pass
        try:
            hdd_val = lr.hdd()
        except Exception:
            pass
    except ImportError:
        pass

    return {
        "type_token_ratio": ttr,
        "hapax_ratio": hapax / word_count,
        "mtld": mtld_val,
        "hdd": hdd_val,
    }


def _sentiment_features(text: str) -> Dict[str, float]:
    from textblob import TextBlob

    blob = TextBlob(text)
    vader = _get_vader()
    vs = vader.polarity_scores(text)

    return {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity,
        "vader_compound": vs["compound"],
        "vader_pos": vs["pos"],
        "vader_neg": vs["neg"],
        "vader_neu": vs["neu"],
        "emotional_intensity": abs(vs["compound"]),
    }


# ---------------------------------------------------------------------------
# Topic model
# ---------------------------------------------------------------------------

def fit_topic_model(
    texts: List[str],
    n_topics: int = 10,
    seed: int = 42,
) -> Tuple[Any, Any]:
    """Fit LDA topic model on corpus. Returns (lda_model, vectorizer)."""
    global _lda_model, _vectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    logger.info("Fitting LDA topic model with %d topics on %d documents", n_topics, len(texts))
    _vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        max_features=5000,
        stop_words="english",
    )
    dtm = _vectorizer.fit_transform(texts)
    _lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=seed,
        max_iter=20,
        learning_method="online",
    )
    _lda_model.fit(dtm)
    logger.info("LDA topic model fitted")
    return _lda_model, _vectorizer


def _topic_features(text: str) -> Dict[str, Any]:
    """Get topic distribution for a single document. Requires prior fit."""
    if _lda_model is None or _vectorizer is None:
        return {
            "topic_distribution": [0.0] * 10,
            "dominant_topic": 0,
            "topic_coherence": 0.0,
        }
    vec = _vectorizer.transform([text])
    dist = _lda_model.transform(vec)[0].tolist()
    dominant = int(np.argmax(dist))
    return {
        "topic_distribution": dist,
        "dominant_topic": dominant,
        "topic_coherence": float(max(dist)),
    }


def get_topic_top_words(n_words: int = 10) -> List[List[str]]:
    """Return top words for each topic. Requires prior fit."""
    if _lda_model is None or _vectorizer is None:
        return []
    feature_names = _vectorizer.get_feature_names_out()
    topics = []
    for topic_weights in _lda_model.components_:
        top_idx = topic_weights.argsort()[-n_words:][::-1]
        topics.append([feature_names[i] for i in top_idx])
    return topics


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_features(text: str, case_id: str, variant: str) -> TextFeatures:
    """Extract all text features for a single document."""
    length = _length_features(text)
    complexity = _complexity_features(text)
    vocab = _vocabulary_features(text)
    sentiment = _sentiment_features(text)
    topics = _topic_features(text)

    return TextFeatures(
        case_id=case_id,
        variant=variant,
        **length,
        **complexity,
        **vocab,
        **sentiment,
        **topics,
    )


def extract_features_batch(
    texts: List[str],
    case_ids: List[str],
    variant: str,
) -> List[TextFeatures]:
    """Extract features for a list of texts."""
    results = []
    for text, cid in tqdm(zip(texts, case_ids), total=len(texts),
                          desc=f"Text features ({variant})", unit="doc"):
        try:
            results.append(extract_features(text, cid, variant))
        except Exception as e:
            logger.warning("Feature extraction failed for %s: %s", cid, e)
            results.append(TextFeatures(case_id=cid, variant=variant))
    return results


def features_to_dataframe(features: List[TextFeatures]) -> pd.DataFrame:
    """Convert a list of TextFeatures to a DataFrame (excluding topic_distribution list)."""
    records = []
    for f in features:
        d = {
            "case_id": f.case_id,
            "variant": f.variant,
            "word_count": f.word_count,
            "char_count": f.char_count,
            "sentence_count": f.sentence_count,
            "avg_word_length": f.avg_word_length,
            "avg_sentence_length": f.avg_sentence_length,
            "flesch_reading_ease": f.flesch_reading_ease,
            "flesch_kincaid_grade": f.flesch_kincaid_grade,
            "gunning_fog": f.gunning_fog,
            "smog_index": f.smog_index,
            "ari": f.ari,
            "type_token_ratio": f.type_token_ratio,
            "hapax_ratio": f.hapax_ratio,
            "mtld": f.mtld,
            "hdd": f.hdd,
            "polarity": f.polarity,
            "subjectivity": f.subjectivity,
            "vader_compound": f.vader_compound,
            "vader_pos": f.vader_pos,
            "vader_neg": f.vader_neg,
            "vader_neu": f.vader_neu,
            "emotional_intensity": f.emotional_intensity,
            "dominant_topic": f.dominant_topic,
            "topic_coherence": f.topic_coherence,
        }
        # Add individual topic columns
        for i, val in enumerate(f.topic_distribution):
            d[f"topic_{i}"] = val
        records.append(d)
    return pd.DataFrame(records)
