"""Shared pytest fixtures for slot_audit tests."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

_HERE = Path(__file__).parent
_CONFIGS = _HERE.parent / "configs"


@pytest.fixture(scope="session")
def anchor_config():
    with open(_CONFIGS / "issue_anchors.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def event_config():
    with open(_CONFIGS / "event_lexicon.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def fixture_df():
    return pd.read_csv(_HERE / "pairs_fixture.csv")
