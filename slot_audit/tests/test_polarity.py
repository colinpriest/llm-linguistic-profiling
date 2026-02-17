"""Tests for polarity slot extraction."""
from __future__ import annotations

from slot_audit.extract_polarity import extract_polarity


class TestBasicPolarity:
    def test_positive_event(self, event_config):
        events = extract_polarity("The bank refunded me.", event_config)
        assert events.get("refund") == "POS"

    def test_negated_event(self, event_config):
        events = extract_polarity("The bank did not refund me.", event_config)
        assert events.get("refund") == "NEG"

    def test_contraction_negation(self, event_config):
        events = extract_polarity("They didn't refund the charge.", event_config)
        assert events.get("refund") == "NEG"

    def test_denied_positive(self, event_config):
        """'denied' is a positive polarity for the deny event (the denial happened)."""
        events = extract_polarity("The bank denied my claim.", event_config)
        assert events.get("deny") == "POS"


class TestNegationWindow:
    def test_negation_within_window(self, event_config):
        events = extract_polarity(
            "The bank did not refund me.", event_config, negation_window=4,
        )
        assert events.get("refund") == "NEG"

    def test_negation_outside_window(self, event_config):
        """Negation too far away should not flip polarity."""
        # "not" is > 4 tokens before "refunded"
        events = extract_polarity(
            "I was not sure about the entire situation but they refunded me.",
            event_config,
            negation_window=4,
        )
        assert events.get("refund") == "POS"


class TestAnchorMasking:
    def test_credit_in_credit_report(self, event_config, anchor_config):
        """'credit' in 'credit report' should NOT trigger refund event."""
        from slot_audit.extract_anchors import extract_anchors

        text = "There is an error on my credit report."
        _, spans = extract_anchors(text, anchor_config)
        events = extract_polarity(text, event_config, anchor_spans=spans)
        assert "refund" not in events

    def test_report_in_credit_report(self, event_config, anchor_config):
        """'report' in 'credit report' should NOT trigger report event."""
        from slot_audit.extract_anchors import extract_anchors

        text = "Check my credit report for errors."
        _, spans = extract_anchors(text, anchor_config)
        events = extract_polarity(text, event_config, anchor_spans=spans)
        assert "report" not in events

    def test_standalone_reported(self, event_config, anchor_config):
        """Standalone 'reported' should still be detected."""
        from slot_audit.extract_anchors import extract_anchors

        text = "I reported the issue."
        _, spans = extract_anchors(text, anchor_config)
        events = extract_polarity(text, event_config, anchor_spans=spans)
        assert events.get("report") == "POS"


class TestIgnorePatterns:
    def test_not_only(self, event_config):
        """'not only' should NOT count as negation."""
        events = extract_polarity(
            "They not only charged me but also denied my claim.", event_config,
        )
        assert events.get("charge") == "POS"


class TestNoEvents:
    def test_plain_text(self, event_config):
        events = extract_polarity("Customer service was rude.", event_config)
        assert len(events) == 0

    def test_no_event_verbs(self, event_config):
        events = extract_polarity("The representative handled my request.", event_config)
        assert len(events) == 0
