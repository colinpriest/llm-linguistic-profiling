"""Tests for comparison logic and end-to-end fixture validation."""
from __future__ import annotations

import pandas as pd
import pytest

from slot_audit import SlotStatus
from slot_audit.aggregate_report import aggregate_high_fidelity, compute_fixture_accuracy
from slot_audit.compare_slots import compare_polarity, compare_set_slots
from slot_audit.slot_audit import audit_pair, run_audit


# ---------------------------------------------------------------------------
# Unit tests for compare_set_slots
# ---------------------------------------------------------------------------

class TestCompareSetSlots:
    def test_preserved_absence(self):
        r = compare_set_slots(set(), set(), "numeric")
        assert r.status == SlotStatus.PRESERVED_ABSENCE

    def test_preserved_present(self):
        r = compare_set_slots({"money:120.00"}, {"money:120.00"}, "numeric")
        assert r.status == SlotStatus.PRESERVED_PRESENT

    def test_omission(self):
        r = compare_set_slots({"money:120.00"}, set(), "numeric")
        assert r.status == SlotStatus.OMISSION

    def test_addition(self):
        r = compare_set_slots(set(), {"money:500.00"}, "numeric")
        assert r.status == SlotStatus.ADDITION

    def test_mutation(self):
        r = compare_set_slots({"money:120.00"}, {"money:102.00"}, "numeric")
        assert r.status == SlotStatus.MUTATION


class TestComparePolarity:
    def test_preserved_absence(self):
        r = compare_polarity({}, {})
        assert r.status == SlotStatus.PRESERVED_ABSENCE

    def test_preserved_present(self):
        r = compare_polarity({"refund": "NEG"}, {"refund": "NEG"})
        assert r.status == SlotStatus.PRESERVED_PRESENT

    def test_omission(self):
        r = compare_polarity({"refund": "NEG"}, {})
        assert r.status == SlotStatus.OMISSION

    def test_addition(self):
        r = compare_polarity({}, {"deny": "POS"})
        assert r.status == SlotStatus.ADDITION

    def test_contradiction(self):
        r = compare_polarity({"refund": "NEG"}, {"refund": "POS"})
        assert r.status == SlotStatus.MUTATION_POLARITY_CONTRADICTION
        assert r.has_contradiction is True

    def test_mutation_no_contradiction(self):
        """Different event sets (no shared keys) = MUTATION, not contradiction."""
        r = compare_polarity({"refund": "NEG"}, {"deny": "POS"})
        assert r.status == SlotStatus.MUTATION
        assert r.has_contradiction is False


# ---------------------------------------------------------------------------
# Fixture-driven end-to-end tests
# ---------------------------------------------------------------------------

class TestFixtureRows:
    """Validate each fixture row against expected statuses."""

    def test_all_fixture_rows(self, fixture_df, anchor_config, event_config):
        errors = []
        for _, row in fixture_df.iterrows():
            label = f"{row['complaint_id']}-{row['rewrite_id']}"
            result = audit_pair(
                source_text=str(row["source_text"]),
                rewrite_text=str(row["rewrite_text"]),
                anchor_config=anchor_config,
                event_config=event_config,
            )

            for fam in ("numeric", "temporal", "polarity", "anchor"):
                expected = row[f"expected_{fam}_status"]
                actual = result[f"{fam}_status"]
                if actual != expected:
                    errors.append(
                        f"  {label} {fam}: expected={expected}, got={actual}"
                    )

            expected_pass = bool(int(row["expected_strict_pass"]))
            if result["strict_pair_pass"] != expected_pass:
                errors.append(
                    f"  {label} strict_pass: expected={expected_pass}, "
                    f"got={result['strict_pair_pass']}"
                )

            expected_contra = bool(int(row["expected_contradiction_flag"]))
            if result["has_contradiction"] != expected_contra:
                errors.append(
                    f"  {label} contradiction: expected={expected_contra}, "
                    f"got={result['has_contradiction']}"
                )

        if errors:
            pytest.fail("Fixture mismatches:\n" + "\n".join(errors))


class TestValidationChecklist:
    """Quick validation checklist from the spec."""

    def test_c001_r1_all_pass(self, fixture_df, anchor_config, event_config):
        row = fixture_df[(fixture_df["complaint_id"] == "C001") & (fixture_df["rewrite_id"] == "R1")].iloc[0]
        result = audit_pair(str(row["source_text"]), str(row["rewrite_text"]), anchor_config, event_config)
        assert result["strict_pair_pass"] is True

    def test_c003_r1_polarity_contradiction(self, fixture_df, anchor_config, event_config):
        row = fixture_df[(fixture_df["complaint_id"] == "C003") & (fixture_df["rewrite_id"] == "R1")].iloc[0]
        result = audit_pair(str(row["source_text"]), str(row["rewrite_text"]), anchor_config, event_config)
        assert result["polarity_status"] == "MUTATION_POLARITY_CONTRADICTION"
        assert result["has_contradiction"] is True

    def test_c006_r1_absence_preservation(self, fixture_df, anchor_config, event_config):
        row = fixture_df[(fixture_df["complaint_id"] == "C006") & (fixture_df["rewrite_id"] == "R1")].iloc[0]
        result = audit_pair(str(row["source_text"]), str(row["rewrite_text"]), anchor_config, event_config)
        assert result["strict_pair_pass"] is True
        # All families should be PRESERVED_ABSENCE
        for fam in ("numeric", "temporal", "polarity", "anchor"):
            assert result[f"{fam}_status"] == "PRESERVED_ABSENCE"

    def test_c006_r2_synonym_anchor(self, fixture_df, anchor_config, event_config):
        row = fixture_df[(fixture_df["complaint_id"] == "C006") & (fixture_df["rewrite_id"] == "R2")].iloc[0]
        result = audit_pair(str(row["source_text"]), str(row["rewrite_text"]), anchor_config, event_config)
        assert result["anchor_status"] == "PRESERVED_PRESENT"

    def test_c009_r1_temporal_omission(self, fixture_df, anchor_config, event_config):
        row = fixture_df[(fixture_df["complaint_id"] == "C009") & (fixture_df["rewrite_id"] == "R1")].iloc[0]
        result = audit_pair(str(row["source_text"]), str(row["rewrite_text"]), anchor_config, event_config)
        assert result["temporal_status"] == "OMISSION"


class TestHighFidelityPass:
    """Validate high_fidelity_pass field from audit_pair."""

    def test_all_preserved_is_high_fidelity(self, fixture_df, anchor_config, event_config):
        """C001-R1: all families PRESERVED_PRESENT → high-fidelity."""
        row = fixture_df[(fixture_df["complaint_id"] == "C001") & (fixture_df["rewrite_id"] == "R1")].iloc[0]
        result = audit_pair(str(row["source_text"]), str(row["rewrite_text"]), anchor_config, event_config)
        assert result["high_fidelity_pass"] is True

    def test_numeric_mutation_not_high_fidelity(self, fixture_df, anchor_config, event_config):
        """C001-R2: numeric MUTATION → not high-fidelity."""
        row = fixture_df[(fixture_df["complaint_id"] == "C001") & (fixture_df["rewrite_id"] == "R2")].iloc[0]
        result = audit_pair(str(row["source_text"]), str(row["rewrite_text"]), anchor_config, event_config)
        assert result["high_fidelity_pass"] is False

    def test_contradiction_not_high_fidelity(self, fixture_df, anchor_config, event_config):
        """C003-R1: polarity contradiction → not high-fidelity."""
        row = fixture_df[(fixture_df["complaint_id"] == "C003") & (fixture_df["rewrite_id"] == "R1")].iloc[0]
        result = audit_pair(str(row["source_text"]), str(row["rewrite_text"]), anchor_config, event_config)
        assert result["high_fidelity_pass"] is False

    def test_polarity_omission_is_high_fidelity(self, fixture_df, anchor_config, event_config):
        """C003-R2: polarity OMISSION (no contradiction) + numeric preserved → high-fidelity."""
        row = fixture_df[(fixture_df["complaint_id"] == "C003") & (fixture_df["rewrite_id"] == "R2")].iloc[0]
        result = audit_pair(str(row["source_text"]), str(row["rewrite_text"]), anchor_config, event_config)
        assert result["high_fidelity_pass"] is True

    def test_anchor_omission_is_high_fidelity(self, fixture_df, anchor_config, event_config):
        """C004-R1: anchor OMISSION (not MUTATION) → high-fidelity."""
        row = fixture_df[(fixture_df["complaint_id"] == "C004") & (fixture_df["rewrite_id"] == "R1")].iloc[0]
        result = audit_pair(str(row["source_text"]), str(row["rewrite_text"]), anchor_config, event_config)
        assert result["high_fidelity_pass"] is True

    def test_temporal_mutation_not_high_fidelity(self, fixture_df, anchor_config, event_config):
        """C002-R2: temporal MUTATION → not high-fidelity."""
        row = fixture_df[(fixture_df["complaint_id"] == "C002") & (fixture_df["rewrite_id"] == "R2")].iloc[0]
        result = audit_pair(str(row["source_text"]), str(row["rewrite_text"]), anchor_config, event_config)
        assert result["high_fidelity_pass"] is False

    def test_numeric_addition_not_high_fidelity(self, fixture_df, anchor_config, event_config):
        """C005-R1: numeric ADDITION → not high-fidelity (num not in _preserved)."""
        row = fixture_df[(fixture_df["complaint_id"] == "C005") & (fixture_df["rewrite_id"] == "R1")].iloc[0]
        result = audit_pair(str(row["source_text"]), str(row["rewrite_text"]), anchor_config, event_config)
        assert result["high_fidelity_pass"] is False

    def test_temporal_addition_is_high_fidelity(self, fixture_df, anchor_config, event_config):
        """C005-R2: temporal ADDITION (not MUTATION) + numeric preserved → high-fidelity."""
        row = fixture_df[(fixture_df["complaint_id"] == "C005") & (fixture_df["rewrite_id"] == "R2")].iloc[0]
        result = audit_pair(str(row["source_text"]), str(row["rewrite_text"]), anchor_config, event_config)
        assert result["high_fidelity_pass"] is True


class TestComplaintAggregation:
    """Test complaint-level aggregation on the fixture data."""

    def test_strict_complaint_criteria(self, fixture_df, anchor_config, event_config):
        pair_df, complaint_df, preserved_df, metrics = run_audit(
            fixture_df.rename(columns={
                "complaint_id": "complaint_id",
                "rewrite_id": "rewrite_id",
                "source_text": "source_text",
                "rewrite_text": "rewrite_text",
            }),
            anchor_config=anchor_config,
            event_config=event_config,
            strict_threshold=1.0,
        )

        # C006 has two rewrites both passing → strict pass rate = 1.0
        c006 = complaint_df[complaint_df["complaint_id"] == "C006"]
        assert len(c006) == 1
        assert c006.iloc[0]["strict_pass_rate"] == 1.0
        assert bool(c006.iloc[0]["meets_criteria"]) is True

        # C001 has one pass and one fail → strict pass rate = 0.5
        c001 = complaint_df[complaint_df["complaint_id"] == "C001"]
        assert len(c001) == 1
        assert c001.iloc[0]["strict_pass_rate"] == 0.5
        assert bool(c001.iloc[0]["meets_criteria"]) is False

    def test_high_fidelity_complaint_aggregation(self, fixture_df, anchor_config, event_config):
        pair_df, complaint_df, preserved_df, metrics = run_audit(
            fixture_df, anchor_config=anchor_config, event_config=event_config,
        )

        # C006 both rewrites are high-fidelity → meets_high_fidelity
        c006 = complaint_df[complaint_df["complaint_id"] == "C006"]
        assert bool(c006.iloc[0]["meets_high_fidelity"]) is True
        assert c006.iloc[0]["high_fidelity_rate"] == 1.0

        # C001 has R1=HF but R2=not HF → not meets_high_fidelity
        c001 = complaint_df[complaint_df["complaint_id"] == "C001"]
        assert bool(c001.iloc[0]["meets_high_fidelity"]) is False
        assert c001.iloc[0]["high_fidelity_rate"] == 0.5

        # C004: both rewrites are HF (anchor omission/addition tolerated)
        c004 = complaint_df[complaint_df["complaint_id"] == "C004"]
        assert bool(c004.iloc[0]["meets_high_fidelity"]) is True

    def test_high_fidelity_metrics(self, fixture_df, anchor_config, event_config):
        pair_df, complaint_df, preserved_df, metrics = run_audit(
            fixture_df, anchor_config=anchor_config, event_config=event_config,
        )

        hf = metrics["global"]["high_fidelity"]
        # 10 out of 16 pairs should be high-fidelity
        assert hf["n_pairs"] == 10
        assert hf["pair_rate"] == round(10 / 16, 4)

        # 4 out of 9 complaints should be high-fidelity (C004, C006, C008, C009)
        assert hf["n_complaints"] == 4


class TestFixtureAccuracy:
    """Test fixture correctness auto-detection and accuracy computation."""

    def test_fixture_mode_detected(self, fixture_df, anchor_config, event_config):
        """When expected_ columns present, fixture_accuracy should appear in metrics."""
        _, _, _, metrics = run_audit(
            fixture_df, anchor_config=anchor_config, event_config=event_config,
        )
        assert "fixture_accuracy" in metrics

    def test_fixture_accuracy_values(self, fixture_df, anchor_config, event_config):
        """All fixture labels should match → 100% accuracy."""
        pair_df, _, _, metrics = run_audit(
            fixture_df, anchor_config=anchor_config, event_config=event_config,
        )
        fa = metrics["fixture_accuracy"]
        assert fa["overall_accuracy"] == 1.0
        assert fa["strict_pass_accuracy"] == 1.0
        assert fa["contradiction_accuracy"] == 1.0
        for fam in ("numeric", "temporal", "polarity", "anchor"):
            assert fa["per_family"][fam]["accuracy"] == 1.0
            assert fa["per_family"][fam]["mismatches"] == []

    def test_no_fixture_mode_without_expected_cols(self, fixture_df, anchor_config, event_config):
        """Without expected_ columns, fixture_accuracy should not appear."""
        stripped = fixture_df.drop(columns=[
            c for c in fixture_df.columns if c.startswith("expected_")
        ])
        _, _, _, metrics = run_audit(
            stripped, anchor_config=anchor_config, event_config=event_config,
        )
        assert "fixture_accuracy" not in metrics
