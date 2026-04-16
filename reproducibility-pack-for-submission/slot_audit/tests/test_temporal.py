"""Tests for temporal slot extraction."""
from __future__ import annotations

from slot_audit.extract_temporal import extract_temporal


class TestDurations:
    def test_within_days(self):
        slots = extract_temporal("within 30 days")
        canonicals = {s.canonical() for s in slots}
        assert "duration:30:day:within" in canonicals

    def test_for_months(self):
        slots = extract_temporal("for 6 months")
        canonicals = {s.canonical() for s in slots}
        assert "duration:6:month:for" in canonicals

    def test_after_weeks(self):
        slots = extract_temporal("after 2 weeks")
        canonicals = {s.canonical() for s in slots}
        assert "duration:2:week:after" in canonicals

    def test_bare_duration(self):
        slots = extract_temporal("3 years of service")
        canonicals = {s.canonical() for s in slots}
        assert "duration:3:year" in canonicals

    def test_for_weeks(self):
        slots = extract_temporal("for 4 weeks")
        canonicals = {s.canonical() for s in slots}
        assert "duration:4:week:for" in canonicals


class TestFrequency:
    def test_twice(self):
        slots = extract_temporal("I called twice")
        canonicals = {s.canonical() for s in slots}
        assert "frequency:2" in canonicals

    def test_multiple_times(self):
        slots = extract_temporal("I called multiple times")
        canonicals = {s.canonical() for s in slots}
        assert "frequency:multiple" in canonicals

    def test_three_times(self):
        slots = extract_temporal("three times")
        canonicals = {s.canonical() for s in slots}
        assert "frequency:3" in canonicals


class TestYears:
    def test_year_2023(self):
        slots = extract_temporal("in 2023")
        canonicals = {s.canonical() for s in slots}
        assert "year:2023" in canonicals

    def test_year_not_number(self):
        """Standalone years should be temporal, not numeric."""
        slots = extract_temporal("This happened in 2023.")
        assert all(s.subtype == "year" for s in slots)


class TestNoTemporal:
    def test_plain_text(self):
        slots = extract_temporal("The representative handled my request.")
        assert len(slots) == 0

    def test_no_durations(self):
        slots = extract_temporal("Customer service was rude.")
        assert len(slots) == 0
