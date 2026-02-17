"""Tests for issue-anchor extraction."""
from __future__ import annotations

from slot_audit.extract_anchors import extract_anchors


class TestAnchorMatching:
    def test_exact_match(self, anchor_config):
        found, _ = extract_anchors("This was a chargeback dispute.", anchor_config)
        assert "chargeback" in found

    def test_synonym_match(self, anchor_config):
        """'merchant dispute' should map to canonical 'chargeback'."""
        found, _ = extract_anchors("I filed a merchant dispute.", anchor_config)
        assert "chargeback" in found

    def test_credit_report(self, anchor_config):
        found, _ = extract_anchors("There is an error on my credit report.", anchor_config)
        assert "credit_report" in found

    def test_hard_inquiry(self, anchor_config):
        found, _ = extract_anchors("I found a hard inquiry I do not recognise.", anchor_config)
        assert "hard_inquiry" in found

    def test_debt_collection(self, anchor_config):
        found, _ = extract_anchors("Regarding debt collection calls.", anchor_config)
        assert "debt_collection" in found

    def test_foreclosure(self, anchor_config):
        found, _ = extract_anchors("They mentioned foreclosure.", anchor_config)
        assert "foreclosure" in found

    def test_escrow(self, anchor_config):
        found, _ = extract_anchors("My mortgage escrow shortage.", anchor_config)
        assert "escrow" in found

    def test_servicer(self, anchor_config):
        found, _ = extract_anchors("The servicer did not respond.", anchor_config)
        assert "mortgage_servicer" in found

    def test_auto_loan(self, anchor_config):
        found, _ = extract_anchors("The auto loan error continued.", anchor_config)
        assert "auto_loan" in found


class TestSynonymCanonicalisation:
    def test_chargeback_synonyms(self, anchor_config):
        """All chargeback synonyms should map to 'chargeback'."""
        for text in [
            "chargeback", "merchant dispute", "card dispute",
            "disputed charge", "unauthorised charge", "unauthorized charge",
        ]:
            found, _ = extract_anchors(f"About {text}.", anchor_config)
            assert "chargeback" in found, f"Failed for: {text}"


class TestPluralS:
    def test_chargebacks_plural(self, anchor_config):
        found, _ = extract_anchors("Multiple chargebacks were filed.", anchor_config)
        assert "chargeback" in found


class TestExcludePhrases:
    def test_customer_service_excluded(self, anchor_config):
        """Anchors within exclude_phrases should not match."""
        found, _ = extract_anchors("I contacted customer service.", anchor_config)
        # "service" alone isn't an anchor synonym anyway, but the phrase is excluded
        assert "mortgage_servicer" not in found


class TestNoAnchors:
    def test_plain_text(self, anchor_config):
        found, _ = extract_anchors("The representative handled my request.", anchor_config)
        assert len(found) == 0

    def test_rude_service(self, anchor_config):
        found, _ = extract_anchors("Customer service was rude.", anchor_config)
        assert len(found) == 0


class TestSpanReporting:
    def test_spans_returned(self, anchor_config):
        _, spans = extract_anchors("This was a chargeback dispute.", anchor_config)
        assert len(spans) > 0
        # Spans should be (start, end) tuples
        for s, e in spans:
            assert s < e
