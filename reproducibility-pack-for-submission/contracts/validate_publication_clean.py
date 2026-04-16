#!/usr/bin/env python3
"""
validate_publication_clean.py

Single-command publication cleanliness gate for fairness reporting artifacts.

Usage example:
    python contracts/validate_publication_clean.py \
      --payload-glob "artifacts/panels/*.json" \
      --html-glob "output/*.html" \
      --tex-glob "artifacts/**/tables/**/*.tex" \
      --reason-map "contracts/reason_code_map.json" \
      --schema-dir "contracts/schemas" \
      --manifest "contracts/publication_manifest.json" \
      --strict

Exit codes:
    0 = PASS
    1 = FAIL (one or more blocking errors)

Notes:
- JSON is parsed with parse_constant rejector (NaN/Infinity are hard failures).
- If jsonschema package is unavailable, schema validation is skipped with warning.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# -----------------------------
# Config
# -----------------------------
# Pattern for forbidden tokens that indicate unformatted NaN/inf/None values
# Excludes common legitimate uses:
#   - CSS: "none;" or ": none" or "=none" or "'none'"
#   - English: "None of", "none of", "have none", "is none"
FORBIDDEN_TOKEN_PATTERN = re.compile(
    r"(?i)\b(nanx|nan(?!o)|inf|-inf)\b"  # nan/inf/nanx (nan excludes "nano")
)
# Separate pattern for 'none' that excludes CSS and common English contexts
FORBIDDEN_NONE_PATTERN = re.compile(
    r"(?i)(?<![:\-='\"])(?<!\bof )(?<!\bhave )(?<!\bis )(?<!\bare )\bnone\b(?![;\s]*[}\)])"
)

ALLOWED_STATUS = {"ok", "ne", "suppressed"}
NE_CODES_REQUIRED = True

# Panel IDs
PANEL_IPW = "panel12_ipw_robustness_monetary"
PANEL_PROCESS = "panel_process_fairness_severity"

# Key fields for duplicate detection (in priority order)
PREFERRED_KEY_FIELDS = [
    ("model", "dimension", "metric", "severity"),
    ("model", "dimension", "metric"),
    ("model", "dimension"),
    ("model", "metric"),
    ("model",),
]


@dataclass
class Issue:
    severity: str  # "ERROR" or "WARN"
    scope: str
    message: str


class Gate:
    """Accumulates validation issues for final pass/fail determination."""

    def __init__(self) -> None:
        self.issues: List[Issue] = []

    def error(self, scope: str, msg: str) -> None:
        self.issues.append(Issue("ERROR", scope, msg))

    def warn(self, scope: str, msg: str) -> None:
        self.issues.append(Issue("WARN", scope, msg))

    @property
    def errors(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == "ERROR"]

    @property
    def warnings(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == "WARN"]

    def summary(self) -> str:
        lines = []
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
            for i in self.errors:
                lines.append(f"  - [{i.scope}] {i.message}")
        if self.warnings:
            lines.append(f"Warnings: {len(self.warnings)}")
            for i in self.warnings:
                lines.append(f"  - [{i.scope}] {i.message}")
        if not lines:
            lines.append("No issues found.")
        return "\n".join(lines)


# -----------------------------
# Utilities
# -----------------------------
def load_json_strict(path: Path) -> Any:
    """Load JSON with strict parsing (rejects NaN/Infinity constants)."""
    def _reject_constant(x: str) -> None:
        raise ValueError(f"Non-JSON numeric constant encountered: {x}")

    txt = path.read_text(encoding="utf-8")
    return json.loads(txt, parse_constant=_reject_constant)


def scan_forbidden_tokens(text: str) -> List[str]:
    """Find forbidden tokens in text.

    For HTML files, we need to distinguish between:
    - Legitimate uses: CSS 'none', English phrases like 'None of the'
    - Data leakage: Python None/nan/inf rendered as strings
    """
    tokens = set()

    # Find nan/inf/nanx (these are almost always data issues)
    for m in FORBIDDEN_TOKEN_PATTERN.finditer(text):
        tokens.add(m.group(0))

    # For 'none', only flag if it looks like a data value (not CSS or English)
    # Look for patterns like: >None<, >none<, "None", 'None' (in table cells or JSON)
    none_data_pattern = re.compile(r'(?i)[>\"\']none[<\"\']')
    if none_data_pattern.search(text):
        # Check if there are suspicious 'None' in contexts suggesting Python values
        # Specifically: inside td/th tags, or as JSON values
        td_none = re.compile(r'<t[dh][^>]*>\s*None\s*</t[dh]>', re.IGNORECASE)
        if td_none.search(text):
            tokens.add('None')

    return sorted(tokens)


def assert_finite_numbers(obj: Any, path: str, gate: Gate) -> None:
    """Recursively check all numbers are finite."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            gate.error(path, "Non-finite float encountered.")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            assert_finite_numbers(v, f"{path}.{k}", gate)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            assert_finite_numbers(v, f"{path}[{i}]", gate)


def discover_files(patterns: List[str]) -> List[Path]:
    """Expand glob patterns into sorted unique file paths."""
    out: List[Path] = []
    for p in patterns:
        out.extend(Path(x) for x in glob.glob(p, recursive=True))
    return sorted(set(out))


def is_number(x: Any) -> bool:
    """Check if value is a number (excluding booleans)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


# -----------------------------
# Optional JSON Schema support
# -----------------------------
def maybe_load_jsonschema(gate: Gate):
    """Try to import jsonschema, warn if unavailable."""
    try:
        import jsonschema
        return jsonschema
    except ImportError:
        gate.warn("schema", "jsonschema not installed; schema validation skipped.")
        return None


def load_schemas(schema_dir: Optional[Path], gate: Gate) -> Dict[str, Dict[str, Any]]:
    """Load all schemas from directory, keyed by panel_id."""
    schemas: Dict[str, Dict[str, Any]] = {}
    if not schema_dir:
        return schemas
    if not schema_dir.exists():
        gate.warn("schema", f"Schema dir not found: {schema_dir}")
        return schemas

    for f in schema_dir.glob("*.json"):
        if f.name.endswith(".schema.json") or f.suffix == ".json":
            try:
                sch = load_json_strict(f)
                # Extract panel_id from schema const or filename
                panel_id = (
                    sch.get("properties", {})
                    .get("panel_id", {})
                    .get("const")
                )
                if not panel_id:
                    panel_id = f.stem.replace(".schema", "")
                schemas[panel_id] = sch
            except Exception as e:
                gate.error("schema", f"Failed loading schema {f}: {e}")
    return schemas


# -----------------------------
# Payload validation
# -----------------------------
def validate_payload_common(
    payload: Dict[str, Any],
    payload_path: Path,
    gate: Gate,
    reason_map: Dict[str, str],
) -> None:
    """Validate common payload structure and row semantics."""
    scope = f"payload:{payload_path.name}"

    # panel_id and rows presence
    panel_id = payload.get("panel_id")
    if not panel_id:
        gate.error(scope, "Missing panel_id.")
    rows = payload.get("rows", [])
    if rows is None:
        gate.error(scope, "rows is null; expected list.")
        rows = []

    # finite check
    assert_finite_numbers(payload, scope, gate)

    # row-level checks
    if isinstance(rows, list):
        for i, row in enumerate(rows):
            rs = f"{scope}.rows[{i}]"
            if not isinstance(row, dict):
                gate.error(rs, "Row must be object.")
                continue

            status = row.get("status")
            if status not in ALLOWED_STATUS:
                gate.error(rs, f"Invalid status={status}; expected one of {sorted(ALLOWED_STATUS)}.")
                continue

            # Status-specific constraints
            if status == "ok":
                # At least one primary estimand field should exist and be numeric
                has_estimate = False
                for fld in ("estimate", "ratio", "effect", "or", "rr"):
                    if fld in row and row[fld] is not None:
                        if not is_number(row[fld]) or (isinstance(row[fld], float) and not math.isfinite(row[fld])):
                            gate.error(rs, f"{fld} must be finite numeric for status=ok.")
                        has_estimate = True
                if not has_estimate:
                    gate.error(rs, "status=ok but no estimand field present (estimate/ratio/effect/or/rr).")

                # CI consistency if present
                lo = row.get("ci_low")
                hi = row.get("ci_high")
                est = row.get("estimate", row.get("ratio", row.get("effect")))
                if lo is not None and hi is not None:
                    if not (is_number(lo) and is_number(hi)):
                        gate.error(rs, "ci_low/ci_high must be numeric when present.")
                    else:
                        if lo > hi:
                            gate.error(rs, "ci_low > ci_high.")
                        if est is not None and is_number(est) and not (lo <= est <= hi):
                            gate.warn(rs, "Estimate not inside CI bounds.")

                if "ne_reason_code" in row and row["ne_reason_code"] is not None:
                    gate.error(rs, "status=ok must not carry ne_reason_code.")

            else:  # ne/suppressed
                code = row.get("ne_reason_code")
                if NE_CODES_REQUIRED and not code:
                    gate.error(rs, "status in {ne,suppressed} requires ne_reason_code.")
                if code and code not in reason_map:
                    gate.error(rs, f"Unknown ne_reason_code={code} (missing in reason map).")

                # numeric estimate fields should be null/absent
                for fld in ("estimate", "ratio", "effect", "or", "rr", "ci_low", "ci_high", "p_value", "q_value"):
                    if fld in row and row[fld] is not None:
                        gate.error(rs, f"{fld} must be null for status={status}.")

        # duplicate key check
        check_duplicate_row_keys(rows, scope, gate)

    # reason-code coverage from ne_summary, if present
    ne_summary = payload.get("ne_summary", [])
    if isinstance(ne_summary, list):
        for j, r in enumerate(ne_summary):
            ns = f"{scope}.ne_summary[{j}]"
            code = r.get("reason_code")
            if code and code not in reason_map:
                gate.error(ns, f"Unknown reason_code={code} (missing in reason map).")


def check_duplicate_row_keys(rows: List[Dict[str, Any]], scope: str, gate: Gate) -> None:
    """Check for duplicate row keys using preferred key field combinations."""
    if not rows:
        return
    for key_fields in PREFERRED_KEY_FIELDS:
        if all(all(k in r for k in key_fields) for r in rows):
            seen: Set[Tuple[Any, ...]] = set()
            for i, r in enumerate(rows):
                key = tuple(r.get(k) for k in key_fields)
                if key in seen:
                    gate.error(f"{scope}.rows[{i}]", f"Duplicate row key {key} for fields {key_fields}.")
                seen.add(key)
            return
    gate.warn(scope, "Could not infer stable uniqueness key fields for duplicate check.")


def validate_panel_specific(
    payload: Dict[str, Any],
    payload_path: Path,
    gate: Gate
) -> None:
    """Panel-specific validation rules."""
    panel_id = payload.get("panel_id")
    scope = f"payload:{payload_path.name}"

    if panel_id == PANEL_IPW:
        models = payload.get("models", [])
        diagnostics = payload.get("diagnostics", [])
        rows = payload.get("rows", [])
        ne_summary = payload.get("ne_summary", [])

        # index diagnostics by model
        diag_by_model: Dict[str, Dict[str, Any]] = {}
        if isinstance(diagnostics, list):
            for d in diagnostics:
                m = d.get("model")
                if m:
                    diag_by_model[m] = d

        # each model with obs>0 must have rows or ne_summary
        for m in models:
            d = diag_by_model.get(m)
            if not d:
                gate.error(scope, f"Missing diagnostics for model={m}.")
                continue
            obs = d.get("model_obs", 0)
            if not isinstance(obs, int):
                gate.error(scope, f"diagnostics.model_obs must be int for model={m}.")
                continue

            model_rows = [r for r in rows if r.get("model") == m]
            model_ok_rows = [r for r in model_rows if r.get("status") == "ok"]
            model_ne = [r for r in ne_summary if r.get("model") == m]
            if obs > 0 and len(model_rows) == 0 and len(model_ne) == 0:
                gate.error(scope, f"Panel IPW empty for model={m} with model_obs={obs}.")

            # optional consistency hint: rows_emitted should match count of ok rows
            emitted = d.get("rows_emitted")
            if emitted is not None and isinstance(emitted, int):
                if emitted != len(model_ok_rows):
                    gate.warn(scope, f"rows_emitted mismatch for model={m}: diag={emitted}, actual={len(model_ok_rows)}.")

    elif panel_id == PANEL_PROCESS:
        rows = payload.get("rows", [])
        for i, r in enumerate(rows):
            rs = f"{scope}.rows[{i}]"
            status = r.get("status")
            low = r.get("low_rate")
            high = r.get("high_rate")
            ratio = r.get("ratio")

            if status == "ne":
                # Canonical check for all-zero denominator case
                if (low == 0 or low == 0.0) and (high == 0 or high == 0.0):
                    code = r.get("ne_reason_code")
                    if code != "ALL_ZERO_DENOMINATOR":
                        gate.warn(rs, "Expected ne_reason_code=ALL_ZERO_DENOMINATOR for low=high=0.")
                if ratio is not None:
                    gate.error(rs, "ratio must be null when status=ne.")

            if status == "ok":
                if ratio is None or not is_number(ratio):
                    gate.error(rs, "status=ok requires numeric ratio.")


# -----------------------------
# Rendered HTML / LaTeX checks
# -----------------------------
def validate_rendered_text_file(path: Path, gate: Gate) -> None:
    """Validate rendered HTML/TeX file for forbidden tokens and empty tables."""
    scope = f"render:{path.name}"
    txt = path.read_text(encoding="utf-8", errors="replace")

    bad_tokens = scan_forbidden_tokens(txt)
    if bad_tokens:
        gate.error(scope, f"Forbidden tokens found: {bad_tokens}")

    # basic anti-empty check for table-like content
    if path.suffix.lower() == ".html":
        validate_html_tables(txt, scope, gate)
    elif path.suffix.lower() == ".tex":
        validate_tex_tables(txt, scope, gate)


def validate_html_tables(html: str, scope: str, gate: Gate) -> None:
    """Check HTML tables for empty body content."""
    tables = re.findall(r"<table\b[^>]*>.*?</table>", html, flags=re.IGNORECASE | re.DOTALL)
    if not tables:
        gate.warn(scope, "No <table> elements found in HTML.")
        return

    for idx, tbl in enumerate(tables):
        tbl_scope = f"{scope}.table[{idx}]"

        # Special-case Panel 12 if annotated
        if re.search(r'data-panel-id\s*=\s*["\']panel12_ipw_robustness_monetary["\']', tbl, flags=re.IGNORECASE):
            tbody_match = re.search(r"<tbody\b[^>]*>(.*?)</tbody>", tbl, flags=re.IGNORECASE | re.DOTALL)
            if tbody_match:
                body = tbody_match.group(1)
                n_body_rows = len(re.findall(r"<tr\b", body, flags=re.IGNORECASE))
                if n_body_rows == 0:
                    gate.error(tbl_scope, "Panel 12 rendered with empty tbody.")
            else:
                # fallback count
                n_tr = len(re.findall(r"<tr\b", tbl, flags=re.IGNORECASE))
                if n_tr < 2:
                    gate.error(tbl_scope, "Panel 12 table appears empty/header-only.")

        # generic
        n_tr = len(re.findall(r"<tr\b", tbl, flags=re.IGNORECASE))
        if n_tr < 2:
            gate.warn(tbl_scope, "Table appears header-only or empty (tr<2).")


def validate_tex_tables(tex: str, scope: str, gate: Gate) -> None:
    """Check LaTeX tables for empty tabular content."""
    blocks = re.findall(
        r"\\begin\{tabular\}.*?\\end\{tabular\}",
        tex,
        flags=re.DOTALL
    )
    if not blocks:
        gate.warn(scope, "No \\begin{tabular} blocks found in TeX.")
        return

    for i, b in enumerate(blocks):
        bs = f"{scope}.tabular[{i}]"
        lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
        # count lines that look like rows: contain '&' and end with '\\'
        data_like = [
            ln for ln in lines
            if ("&" in ln and ln.endswith(r"\\") and
                not any(cmd in ln for cmd in [r"\toprule", r"\midrule", r"\bottomrule", r"\cmidrule"]))
        ]
        if len(data_like) == 0:
            gate.warn(bs, "Likely empty tabular (no data-like rows).")


# -----------------------------
# Manifest consistency checks (optional)
# -----------------------------
def validate_manifest_consistency(
    payloads: Dict[Path, Dict[str, Any]],
    manifest_path: Optional[Path],
    gate: Gate
) -> None:
    """Validate payload counts against expected manifest values."""
    if manifest_path is None:
        gate.warn("manifest", "No manifest provided; subset-count consistency checks skipped.")
        return
    if not manifest_path.exists():
        gate.error("manifest", f"Manifest file not found: {manifest_path}")
        return

    try:
        m = load_json_strict(manifest_path)
    except Exception as e:
        gate.error("manifest", f"Failed reading manifest: {e}")
        return

    expected = m.get("expected_counts", [])
    if not isinstance(expected, list):
        gate.error("manifest", "expected_counts must be a list.")
        return

    # index payloads by panel_id
    panel_payloads: Dict[str, List[Dict[str, Any]]] = {}
    for _, p in payloads.items():
        pid = p.get("panel_id")
        if pid:
            panel_payloads.setdefault(pid, []).append(p)

    for i, e in enumerate(expected):
        es = f"manifest.expected_counts[{i}]"
        pid = e.get("panel_id")
        model = e.get("model")
        exp_obs = e.get("model_obs")
        if not pid or not model or exp_obs is None:
            gate.error(es, "Required keys: panel_id, model, model_obs.")
            continue

        candidates = panel_payloads.get(pid, [])
        if not candidates:
            gate.error(es, f"No payload found for panel_id={pid}.")
            continue

        found = False
        for p in candidates:
            for d in p.get("diagnostics", []):
                if d.get("model") == model:
                    found = True
                    got = d.get("model_obs")
                    if got != exp_obs:
                        gate.error(es, f"Count mismatch for panel={pid}, model={model}: expected={exp_obs}, got={got}.")
                    break
        if not found:
            gate.error(es, f"No diagnostics row for model={model} in panel={pid}.")


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publication cleanliness gate for fairness reporting artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python contracts/validate_publication_clean.py \\
      --payload-glob "artifacts/panels/*.json" \\
      --html-glob "output/*.html" \\
      --tex-glob "artifacts/**/tables/**/*.tex" \\
      --reason-map "contracts/reason_code_map.json" \\
      --schema-dir "contracts/schemas" \\
      --strict
        """,
    )
    parser.add_argument("--payload-glob", action="append", default=[], help="Glob for panel payload JSON files.")
    parser.add_argument("--html-glob", action="append", default=[], help="Glob for rendered HTML files.")
    parser.add_argument("--tex-glob", action="append", default=[], help="Glob for rendered TeX files.")
    parser.add_argument("--reason-map", required=True, help="Path to reason_code_map.json")
    parser.add_argument("--schema-dir", default=None, help="Directory containing JSON schemas (optional).")
    parser.add_argument("--manifest", default=None, help="Path to publication manifest for count checks (optional).")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors.")
    parser.add_argument("--latest", action="store_true", help="Auto-select the most recent run in artifacts/runs/.")
    args = parser.parse_args()

    # Handle --latest flag: find most recent run and set globs automatically
    if args.latest:
        runs_dir = Path("artifacts/runs")
        if not runs_dir.exists():
            print(f"Error: {runs_dir} does not exist.", file=sys.stderr)
            return 1
        run_dirs = sorted(
            [d for d in runs_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True,
        )
        if not run_dirs:
            print(f"Error: No run directories found in {runs_dir}.", file=sys.stderr)
            return 1
        latest_run = run_dirs[0]
        print(f"Using latest run: {latest_run.name}")
        # Extract timestamp (YYYYMMDD_HHMMSS) from run directory name
        run_timestamp = latest_run.name[:15] if len(latest_run.name) >= 15 else ""
        run_date = latest_run.name[:8] if len(latest_run.name) >= 8 else ""
        # Auto-populate globs if not explicitly provided
        if not args.payload_glob:
            args.payload_glob = [str(latest_run / "analysis" / "panels" / "*.json")]
        if not args.tex_glob:
            args.tex_glob = [str(latest_run / "analysis" / "tables" / "**" / "*.tex")]
        if not args.html_glob and run_date:
            # Store run timestamp to filter HTML files later
            args._run_timestamp = run_timestamp
            args.html_glob = [f"output/fairness_report_{run_date}*.html"]

    gate = Gate()

    # Load reason map
    reason_map_path = Path(args.reason_map)
    if not reason_map_path.exists():
        print(f"Missing reason map: {reason_map_path}", file=sys.stderr)
        return 1

    try:
        reason_map_obj = load_json_strict(reason_map_path)
    except Exception as e:
        print(f"Failed loading reason map: {e}", file=sys.stderr)
        return 1

    if not isinstance(reason_map_obj, dict) or not reason_map_obj:
        print("reason map must be a non-empty JSON object.", file=sys.stderr)
        return 1

    reason_map: Dict[str, str] = {}
    for k, v in reason_map_obj.items():
        if not isinstance(k, str) or not isinstance(v, str) or not v.strip():
            print(f"Invalid reason map entry: {k}: {v}", file=sys.stderr)
            return 1
        reason_map[k] = v

    # Discover files
    payload_files = discover_files(args.payload_glob)
    html_files = discover_files(args.html_glob)
    tex_files = discover_files(args.tex_glob)

    # Filter HTML files by modification time when using --latest
    # Only include HTML files modified at or after the run's start time
    run_timestamp = getattr(args, "_run_timestamp", None)
    if run_timestamp and html_files:
        from datetime import datetime
        # Convert run_timestamp (YYYYMMDD_HHMMSS) to datetime for comparison
        try:
            run_dt = datetime.strptime(run_timestamp, "%Y%m%d_%H%M%S")
            filtered_html = []
            for hf in html_files:
                # Use file modification time, not filename timestamp
                mtime = datetime.fromtimestamp(hf.stat().st_mtime)
                if mtime >= run_dt:
                    filtered_html.append(hf)
            html_files = filtered_html
        except (ValueError, OSError):
            pass  # Keep all files if timestamp parsing fails

    if not payload_files:
        gate.warn("cli", "No payload JSON files found (check --payload-glob).")
    if not html_files:
        gate.warn("cli", "No HTML files found (check --html-glob).")
    if not tex_files:
        gate.warn("cli", "No TeX files found (check --tex-glob).")

    # Optional schema setup
    jsonschema = maybe_load_jsonschema(gate)
    schemas = load_schemas(Path(args.schema_dir), gate) if args.schema_dir else {}

    # Validate payloads
    payload_objects: Dict[Path, Dict[str, Any]] = {}
    for pf in payload_files:
        scope = f"payload:{pf.name}"
        try:
            txt = pf.read_text(encoding="utf-8", errors="replace")
            bad_tokens = scan_forbidden_tokens(txt)
            if bad_tokens:
                gate.error(scope, f"Forbidden tokens in JSON text: {bad_tokens}")

            payload = load_json_strict(pf)
            if not isinstance(payload, dict):
                gate.error(scope, "Top-level payload must be JSON object.")
                continue

            payload_objects[pf] = payload
            validate_payload_common(payload, pf, gate, reason_map)
            validate_panel_specific(payload, pf, gate)

            # Optional schema validation
            if jsonschema is not None:
                panel_id = payload.get("panel_id")
                schema = schemas.get(panel_id)
                if schema is not None:
                    validator = jsonschema.Draft202012Validator(schema)
                    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
                    for e in errors:
                        loc = "->".join([str(x) for x in e.path]) or "$"
                        gate.error(scope, f"Schema violation at {loc}: {e.message}")
                else:
                    gate.warn(scope, f"No schema found for panel_id={panel_id}; skipped schema validation.")

        except Exception as e:
            gate.error(scope, f"Failed validating payload: {e}")

    # Manifest checks
    validate_manifest_consistency(
        payload_objects,
        Path(args.manifest) if args.manifest else None,
        gate
    )

    # Validate rendered files
    for f in html_files + tex_files:
        try:
            validate_rendered_text_file(f, gate)
        except Exception as e:
            gate.error(f"render:{f.name}", f"Validation crashed: {e}")

    # Output + exit
    print(gate.summary())
    if args.strict and gate.warnings:
        print("\nStrict mode enabled: warnings treated as failures.")
        return 1
    return 1 if gate.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
