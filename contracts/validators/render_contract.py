"""Fail-fast validators for render contracts.

These validators run before HTML/LaTeX rendering to ensure:
1. No raw NaN/Inf values leak into output
2. No empty tables are rendered
3. NE (not estimable) rows have proper reason codes
4. CI consistency (low <= estimate <= high)
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# Forbidden tokens that should never appear in rendered output
FORBIDDEN_TOKENS = re.compile(r"(?i)\b(nanx|nan|inf|-inf)\b")


def _load_json_strict(path: Path) -> Dict[str, Any]:
    """Load JSON with strict parsing (rejects NaN/Infinity constants).

    Parameters
    ----------
    path : Path
        Path to JSON file.

    Returns
    -------
    Dict[str, Any]
        Parsed JSON object.

    Raises
    ------
    ValueError
        If the file contains non-standard JSON constants (NaN, Infinity).
    """
    def _reject_constant(x: str) -> None:
        raise ValueError(f"Non-JSON numeric constant found: {x}")

    txt = path.read_text(encoding="utf-8")
    return json.loads(txt, parse_constant=_reject_constant)


def _assert_finite_numbers(x: Any, path: str = "$") -> None:
    """Recursively check that all numbers are finite.

    Parameters
    ----------
    x : Any
        Value to check (can be nested dict/list).
    path : str
        JSON path for error reporting.

    Raises
    ------
    AssertionError
        If a non-finite number is found.
    """
    if isinstance(x, float):
        if not math.isfinite(x):
            raise AssertionError(f"Non-finite number at {path}: {x}")
    elif isinstance(x, dict):
        for k, v in x.items():
            _assert_finite_numbers(v, f"{path}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            _assert_finite_numbers(v, f"{path}[{i}]")


def _assert_ci_consistency(rows: List[Dict[str, Any]], panel_name: str) -> None:
    """Check that confidence intervals are consistent.

    Parameters
    ----------
    rows : List[Dict[str, Any]]
        Panel rows.
    panel_name : str
        Panel identifier for error messages.

    Raises
    ------
    AssertionError
        If ci_low > ci_high or estimate is outside CI.
    """
    for i, r in enumerate(rows):
        if r.get("status") == "ok":
            est = r.get("estimate", r.get("ratio"))
            lo = r.get("ci_low")
            hi = r.get("ci_high")
            if lo is not None and hi is not None:
                if lo > hi:
                    raise AssertionError(f"{panel_name}: row {i} has ci_low > ci_high")
                if est is not None and not (lo <= est <= hi):
                    raise AssertionError(f"{panel_name}: row {i} estimate {est} outside CI [{lo}, {hi}]")


def _assert_ipw_model_completeness(payload: Dict[str, Any]) -> None:
    """Check that all models have either data rows or NE summary.

    Parameters
    ----------
    payload : Dict[str, Any]
        IPW panel payload.

    Raises
    ------
    AssertionError
        If a model with observations has neither rows nor NE summary.
    """
    models = set(payload["models"])
    diag_by_model = {d["model"]: d for d in payload["diagnostics"]}
    rows = payload.get("rows", [])
    ne_summary = payload.get("ne_summary", [])

    # Every model must be in diagnostics
    missing_diag = models - set(diag_by_model.keys())
    if missing_diag:
        raise AssertionError(f"Missing diagnostics for models: {sorted(missing_diag)}")

    rows_by_model: Dict[str, int] = {m: 0 for m in models}
    ne_by_model: Dict[str, int] = {m: 0 for m in models}

    for r in rows:
        model = r["model"]
        rows_by_model[model] = rows_by_model.get(model, 0) + 1

    for n in ne_summary:
        model = n["model"]
        ne_by_model[model] = ne_by_model.get(model, 0) + 1

    # Fail-fast: if model_obs > 0, must have either data rows or explicit NE summary
    for m in models:
        obs = diag_by_model[m]["model_obs"]
        if obs > 0 and rows_by_model[m] == 0 and ne_by_model[m] == 0:
            raise AssertionError(
                f"IPW panel empty for model={m}: model_obs={obs}, no rows and no ne_summary"
            )


def _assert_no_duplicate_keys(
    rows: List[Dict[str, Any]],
    key_fields: List[str],
    panel_name: str,
) -> None:
    """Check for duplicate row keys.

    Parameters
    ----------
    rows : List[Dict[str, Any]]
        Panel rows.
    key_fields : List[str]
        Fields that form the unique key.
    panel_name : str
        Panel identifier for error messages.

    Raises
    ------
    AssertionError
        If duplicate keys are found.
    """
    seen = set()
    for r in rows:
        k = tuple(r.get(f) for f in key_fields)
        if k in seen:
            raise AssertionError(f"{panel_name}: duplicate row key {k}")
        seen.add(k)


def validate_payload(payload_path: str, schema_path: str) -> None:
    """Validate a panel payload against its JSON schema.

    Parameters
    ----------
    payload_path : str
        Path to the payload JSON file.
    schema_path : str
        Path to the JSON schema file.

    Raises
    ------
    AssertionError
        If validation fails.
    ImportError
        If jsonschema is not installed.
    """
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        raise ImportError("jsonschema package required: pip install jsonschema")

    payload = _load_json_strict(Path(payload_path))
    schema = _load_json_strict(Path(schema_path))

    # JSON Schema validation
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
    if errors:
        msg = "\n".join([f"{list(e.path)}: {e.message}" for e in errors[:20]])
        raise AssertionError(f"Schema validation failed:\n{msg}")

    # Additional fail-fast invariants
    _assert_finite_numbers(payload)

    txt = Path(payload_path).read_text(encoding="utf-8")
    if FORBIDDEN_TOKENS.search(txt):
        raise AssertionError("Forbidden raw token found in payload (nanx/nan/inf).")

    panel_id = payload.get("panel_id", "")
    rows = payload.get("rows", [])

    if panel_id == "panel12_ipw_robustness_monetary":
        _assert_ipw_model_completeness(payload)
        _assert_no_duplicate_keys(rows, ["model", "dimension"], panel_id)
        _assert_ci_consistency(rows, panel_id)

    if panel_id == "panel_process_fairness_severity":
        _assert_no_duplicate_keys(rows, ["model", "metric"], panel_id)
        # Forbid computed artifacts: if NE, ratio must be null
        for r in rows:
            if r["status"] == "ne" and r.get("ratio") is not None:
                raise AssertionError(f"{panel_id}: NE row has non-null ratio")


def validate_rendered_html(html_path: str) -> None:
    """Validate rendered HTML for forbidden tokens and empty tables.

    Parameters
    ----------
    html_path : str
        Path to the rendered HTML file.

    Raises
    ------
    AssertionError
        If forbidden tokens are found or Panel 12 table is empty.
    """
    html = Path(html_path).read_text(encoding="utf-8")

    if FORBIDDEN_TOKENS.search(html):
        raise AssertionError("Rendered HTML contains forbidden tokens (nanx/nan/inf).")

    # Check Panel 12 table if present (requires data-panel-id attribute)
    panel_match = re.search(
        r'<table[^>]*data-panel-id="panel12_ipw_robustness_monetary"[^>]*>(.*?)</table>',
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if panel_match:
        table_html = panel_match.group(1)
        # Count body rows (excluding header)
        tbody_match = re.search(r'<tbody[^>]*>(.*?)</tbody>', table_html, flags=re.DOTALL)
        if tbody_match:
            tbody_html = tbody_match.group(1)
            tr_count = len(re.findall(r"<tr\b", tbody_html, flags=re.IGNORECASE))
            if tr_count == 0:
                raise AssertionError("Panel 12 table rendered with zero rows.")


def validate_ipw_data_structure(ipw_data: Dict[str, Any], models: List[str]) -> List[str]:
    """Validate IPW data structure and return warnings.

    This is a lighter-weight check that can be run inline during report
    generation to catch issues before rendering.

    Parameters
    ----------
    ipw_data : Dict[str, Any]
        The ipw_robustness dict from analysis_results.
    models : List[str]
        List of model names to check.

    Returns
    -------
    List[str]
        List of warning messages (empty if all OK).
    """
    warnings = []

    if not ipw_data:
        warnings.append("IPW data is empty or None")
        return warnings

    if ipw_data.get("error"):
        warnings.append(f"IPW analysis had error: {ipw_data.get('error')}")
        return warnings

    prop_diag = ipw_data.get("propensity_diagnostics", {})
    if not prop_diag:
        warnings.append("Missing propensity_diagnostics")

    for model in models:
        mr = ipw_data.get(model, {})
        if not mr:
            warnings.append(f"No IPW data for model: {model}")
            continue

        if mr.get("skipped"):
            continue  # Expected, not a warning

        dims = mr.get("dimensions", {})
        if not dims:
            warnings.append(f"Model {model} has empty dimensions dict")
            continue

        for dim in ["age", "geographic", "education", "socioeconomic"]:
            dim_data = dims.get(dim, {})
            if not dim_data:
                warnings.append(f"Model {model} missing dimension: {dim}")
                continue

            uw = dim_data.get("unweighted", {})
            wt = dim_data.get("weighted", {})

            # Check for the expected structure
            if "amplification" not in uw:
                warnings.append(f"Model {model}, dim {dim}: unweighted missing 'amplification' key")
            if "amplification" not in wt:
                warnings.append(f"Model {model}, dim {dim}: weighted missing 'amplification' key")

            if "interaction" not in uw or "p_value" not in uw.get("interaction", {}):
                warnings.append(f"Model {model}, dim {dim}: unweighted missing 'interaction.p_value'")
            if "interaction" not in wt or "p_value" not in wt.get("interaction", {}):
                warnings.append(f"Model {model}, dim {dim}: weighted missing 'interaction.p_value'")

    return warnings


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python render_contract.py validate_payload <payload.json> <schema.json>")
        print("  python render_contract.py validate_rendered_html <report.html>")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "validate_payload":
        if len(sys.argv) != 4:
            print("Usage: python render_contract.py validate_payload <payload.json> <schema.json>")
            sys.exit(1)
        validate_payload(sys.argv[2], sys.argv[3])
        print("Payload validation passed.")

    elif cmd == "validate_rendered_html":
        if len(sys.argv) != 3:
            print("Usage: python render_contract.py validate_rendered_html <report.html>")
            sys.exit(1)
        validate_rendered_html(sys.argv[2])
        print("HTML validation passed.")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
