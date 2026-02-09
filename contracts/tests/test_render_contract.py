"""Tests for render contract validation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from contracts.validators.render_contract import (
    validate_ipw_data_structure,
    _assert_finite_numbers,
    _assert_ci_consistency,
    FORBIDDEN_TOKENS,
)
from contracts.validators.reason_notes import (
    REASON_MAP,
    footnote_markers,
    panel_footnotes,
    get_reason_text,
    format_ne_cell,
    format_footnotes_html,
)


class TestReasonNotes:
    """Tests for reason code handling."""

    def test_reason_map_loaded(self):
        """Verify reason map is loaded from JSON."""
        assert len(REASON_MAP) > 0
        assert "ALL_ZERO_DENOMINATOR" in REASON_MAP
        assert "INSUFFICIENT_CELL_N" in REASON_MAP

    def test_get_reason_text_known(self):
        """Known codes return proper text."""
        text = get_reason_text("ALL_ZERO_DENOMINATOR")
        assert "zero" in text.lower()
        assert "undefined" in text.lower()

    def test_get_reason_text_unknown(self):
        """Unknown codes return fallback."""
        text = get_reason_text("UNKNOWN_CODE_XYZ")
        assert "UNKNOWN_CODE_XYZ" in text

    def test_footnote_markers_unique(self):
        """Markers are unique letters."""
        codes = ["ALL_ZERO_DENOMINATOR", "INSUFFICIENT_CELL_N", "ESTIMATION_FAILED"]
        markers = footnote_markers(codes)
        assert markers["ALL_ZERO_DENOMINATOR"] == "a"
        assert markers["INSUFFICIENT_CELL_N"] == "b"
        assert markers["ESTIMATION_FAILED"] == "c"

    def test_footnote_markers_dedup(self):
        """Duplicate codes get same marker."""
        codes = ["ALL_ZERO_DENOMINATOR", "INSUFFICIENT_CELL_N", "ALL_ZERO_DENOMINATOR"]
        markers = footnote_markers(codes)
        assert len(markers) == 2

    def test_panel_footnotes(self):
        """Panel footnotes generate proper structure."""
        codes = ["ALL_ZERO_DENOMINATOR", "INSUFFICIENT_CELL_N"]
        footnotes = panel_footnotes(codes)
        assert len(footnotes) == 2
        assert footnotes[0]["code"] == "ALL_ZERO_DENOMINATOR"
        assert footnotes[0]["marker"] == "a"
        assert "text" in footnotes[0]

    def test_format_ne_cell(self):
        """NE cell formatting includes marker."""
        markers = {"ALL_ZERO_DENOMINATOR": "a"}
        cell = format_ne_cell("ALL_ZERO_DENOMINATOR", markers)
        assert "NE" in cell
        assert "<sup>a</sup>" in cell

    def test_format_footnotes_html(self):
        """HTML footnotes render as list."""
        footnotes = [
            {"code": "A", "marker": "a", "text": "Test reason A"},
            {"code": "B", "marker": "b", "text": "Test reason B"},
        ]
        html = format_footnotes_html(footnotes)
        assert "<ul" in html
        assert "<sup>a</sup>" in html
        assert "Test reason A" in html


class TestFiniteNumbers:
    """Tests for finite number validation."""

    def test_finite_passes(self):
        """Finite numbers pass validation."""
        _assert_finite_numbers({"a": 1.0, "b": [2.0, 3.0]})

    def test_nan_fails(self):
        """NaN values fail validation."""
        with pytest.raises(AssertionError):
            _assert_finite_numbers({"a": float("nan")})

    def test_inf_fails(self):
        """Infinity values fail validation."""
        with pytest.raises(AssertionError):
            _assert_finite_numbers({"a": float("inf")})

    def test_nested_nan_fails(self):
        """Nested NaN values are detected."""
        with pytest.raises(AssertionError):
            _assert_finite_numbers({"a": {"b": [1.0, float("nan")]}})


class TestCIConsistency:
    """Tests for confidence interval validation."""

    def test_valid_ci_passes(self):
        """Valid CIs pass validation."""
        rows = [
            {"status": "ok", "estimate": 0.5, "ci_low": 0.3, "ci_high": 0.7},
        ]
        _assert_ci_consistency(rows, "test_panel")

    def test_inverted_ci_fails(self):
        """Inverted CIs fail validation."""
        rows = [
            {"status": "ok", "estimate": 0.5, "ci_low": 0.7, "ci_high": 0.3},
        ]
        with pytest.raises(AssertionError):
            _assert_ci_consistency(rows, "test_panel")

    def test_estimate_outside_ci_fails(self):
        """Estimate outside CI fails validation."""
        rows = [
            {"status": "ok", "estimate": 0.1, "ci_low": 0.3, "ci_high": 0.7},
        ]
        with pytest.raises(AssertionError):
            _assert_ci_consistency(rows, "test_panel")

    def test_ne_rows_skip_ci_check(self):
        """NE rows skip CI validation."""
        rows = [
            {"status": "ne", "estimate": None, "ci_low": None, "ci_high": None},
        ]
        _assert_ci_consistency(rows, "test_panel")


class TestForbiddenTokens:
    """Tests for forbidden token detection."""

    def test_nan_detected(self):
        """NaN token is detected."""
        assert FORBIDDEN_TOKENS.search("value is nan")
        assert FORBIDDEN_TOKENS.search("NaN found")

    def test_nanx_detected(self):
        """nanx token is detected."""
        assert FORBIDDEN_TOKENS.search("result: nanx")

    def test_inf_detected(self):
        """inf tokens are detected."""
        assert FORBIDDEN_TOKENS.search("value is inf")
        assert FORBIDDEN_TOKENS.search("-inf result")

    def test_valid_text_passes(self):
        """Valid text does not trigger detection."""
        assert not FORBIDDEN_TOKENS.search("value is 0.123")
        assert not FORBIDDEN_TOKENS.search("banana contains nan substring but is not NaN")


class TestIPWDataStructure:
    """Tests for IPW data structure validation."""

    def test_empty_data_warns(self):
        """Empty data generates warning."""
        warnings = validate_ipw_data_structure({}, ["gpt-4o-mini"])
        assert any("empty" in w.lower() for w in warnings)

    def test_error_in_data_warns(self):
        """Error in data generates warning."""
        warnings = validate_ipw_data_structure({"error": "test error"}, ["gpt-4o-mini"])
        assert any("error" in w.lower() for w in warnings)

    def test_missing_model_warns(self):
        """Missing model data generates warning."""
        data = {"propensity_diagnostics": {}}
        warnings = validate_ipw_data_structure(data, ["gpt-4o-mini"])
        assert any("gpt-4o-mini" in w for w in warnings)

    def test_valid_structure_no_warnings(self):
        """Valid structure generates no warnings."""
        data = {
            "propensity_diagnostics": {"n_M1": 100},
            "gpt-4o-mini": {
                "dimensions": {
                    "age": {
                        "unweighted": {
                            "amplification": {"point": 1.5},
                            "interaction": {"p_value": 0.05},
                        },
                        "weighted": {
                            "amplification": {"point": 1.4},
                            "interaction": {"p_value": 0.06},
                        },
                    },
                    "geographic": {
                        "unweighted": {
                            "amplification": {"point": 1.2},
                            "interaction": {"p_value": 0.10},
                        },
                        "weighted": {
                            "amplification": {"point": 1.1},
                            "interaction": {"p_value": 0.12},
                        },
                    },
                    "education": {
                        "unweighted": {
                            "amplification": {"point": 1.3},
                            "interaction": {"p_value": 0.08},
                        },
                        "weighted": {
                            "amplification": {"point": 1.2},
                            "interaction": {"p_value": 0.09},
                        },
                    },
                    "socioeconomic": {
                        "unweighted": {
                            "amplification": {"point": 1.1},
                            "interaction": {"p_value": 0.15},
                        },
                        "weighted": {
                            "amplification": {"point": 1.0},
                            "interaction": {"p_value": 0.18},
                        },
                    },
                }
            }
        }
        warnings = validate_ipw_data_structure(data, ["gpt-4o-mini"])
        assert len(warnings) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
