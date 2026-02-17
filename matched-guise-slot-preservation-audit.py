#!/usr/bin/env python
"""Thin CLI entry point for the matched-guise slot-preservation audit.

Usage examples:

  # Directly from the experiment cache (no new LLM calls) — most common
  python matched-guise-slot-preservation-audit.py --from-cache output/20260216_115325 --output-dir ./audit_out

  # From a pre-built pairs CSV (if you've already exported one)
  python matched-guise-slot-preservation-audit.py --input my_pairs.csv --output-dir ./audit_out

  # Quick test with the bundled fixture
  python matched-guise-slot-preservation-audit.py --input slot_audit/tests/pairs_fixture.csv --output-dir ./audit_out

Run with --help for full option list.
"""
from slot_audit.slot_audit import main

if __name__ == "__main__":
    main()
