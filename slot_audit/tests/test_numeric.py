"""Tests for numeric slot extraction."""
from __future__ import annotations

import pytest

from slot_audit.extract_numeric import extract_numeric


class TestMoneyExtraction:
    def test_dollar_sign(self):
        slots = extract_numeric("The bank charged me $120 for a fee.")
        canonicals = {s.canonical() for s in slots}
        assert "money:120.00" in canonicals

    def test_dollar_with_cents(self):
        slots = extract_numeric("I was charged $1,234.56 total.")
        canonicals = {s.canonical() for s in slots}
        assert "money:1234.56" in canonicals

    def test_word_dollars(self):
        slots = extract_numeric("500 dollars was charged.")
        canonicals = {s.canonical() for s in slots}
        assert "money:500.00" in canonicals

    def test_usd(self):
        slots = extract_numeric("They charged usd 300.")
        canonicals = {s.canonical() for s in slots}
        assert "money:300.00" in canonicals

    def test_formatting_variants_match(self):
        """$1,200 and 1200 dollars should produce the same canonical."""
        s1 = extract_numeric("$1,200")
        s2 = extract_numeric("1200 dollars")
        assert s1[0].canonical() == s2[0].canonical()


class TestPercentAPR:
    def test_percent_sign(self):
        slots = extract_numeric("The rate was 12%.")
        canonicals = {s.canonical() for s in slots}
        assert "percent:12.00" in canonicals

    def test_percent_word(self):
        slots = extract_numeric("19.99 percent interest.")
        canonicals = {s.canonical() for s in slots}
        assert "percent:19.99" in canonicals

    def test_apr(self):
        slots = extract_numeric("APR 19.99%")
        canonicals = {s.canonical() for s in slots}
        assert "apr:19.99" in canonicals


class TestCounts:
    def test_calls(self):
        slots = extract_numeric("after 3 calls")
        canonicals = {s.canonical() for s in slots}
        assert "count:3:call" in canonicals

    def test_payments(self):
        slots = extract_numeric("2 payments were missed")
        canonicals = {s.canonical() for s in slots}
        assert "count:2:payment" in canonicals

    def test_singular_unit(self):
        slots = extract_numeric("1 letter was sent")
        canonicals = {s.canonical() for s in slots}
        assert "count:1:letter" in canonicals


class TestBounds:
    def test_over(self):
        slots = extract_numeric("over 500 dollars")
        assert any(s.subtype == "bound" and s.operator == "over" for s in slots)

    def test_less_than(self):
        slots = extract_numeric("less than 30 dollars")
        assert any(s.subtype == "bound" and s.operator == "less_than" for s in slots)

    def test_between(self):
        slots = extract_numeric("between 3 and 5")
        assert any(s.operator == "between_low" for s in slots)
        assert any(s.operator == "between_high" for s in slots)


class TestNoNumeric:
    def test_plain_text(self):
        slots = extract_numeric("The representative handled my request.")
        assert len(slots) == 0

    def test_no_dollar_amounts(self):
        slots = extract_numeric("Customer service was rude.")
        assert len(slots) == 0
