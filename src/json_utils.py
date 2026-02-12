"""JSON extraction, repair, and validation utilities."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Balanced-brace JSON extraction
# ---------------------------------------------------------------------------

def _extract_balanced_json(text: str) -> Optional[str]:
    """Extract the first balanced JSON object from text."""
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == '\\' and in_string:
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class JSONExtractionError(Exception):
    pass


class JSONParseError(Exception):
    pass


class SchemaValidationError(Exception):
    pass


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def extract_json(response_text: str) -> Optional[str]:
    """Extract JSON from LLM response, handling common formats."""

    # Pattern 1: JSON in markdown code block
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if match:
        return match.group(1)

    # Pattern 2: Balanced-brace extraction (handles arbitrary nesting depth)
    balanced = _extract_balanced_json(response_text)
    if balanced:
        return balanced

    # Pattern 2b: Regex fallback for one level of nesting
    match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', response_text, re.DOTALL)
    if match:
        return match.group(1)

    # Pattern 3: Entire response is JSON
    stripped = response_text.strip()
    if stripped.startswith('{') and stripped.endswith('}'):
        return stripped

    return None


# ---------------------------------------------------------------------------
# JSON repair helpers
# ---------------------------------------------------------------------------

def _close_brackets(json_str: str) -> str:
    """Add missing closing brackets to truncated JSON."""
    open_braces = json_str.count('{') - json_str.count('}')
    open_brackets = json_str.count('[') - json_str.count(']')

    # Remove trailing incomplete key-value
    json_str = re.sub(r',\s*"[^"]*"\s*:\s*$', '', json_str)
    json_str = re.sub(r',\s*$', '', json_str)

    return json_str + (']' * max(0, open_brackets)) + ('}' * max(0, open_braces))


def _escape_newlines_in_strings(json_str: str) -> str:
    """Escape newlines that appear inside JSON string values."""
    result: List[str] = []
    in_string = False
    escape_next = False

    for char in json_str:
        if escape_next:
            result.append(char)
            escape_next = False
        elif char == '\\':
            result.append(char)
            escape_next = True
        elif char == '"':
            result.append(char)
            in_string = not in_string
        elif char == '\n' and in_string:
            result.append('\\n')
        else:
            result.append(char)

    return ''.join(result)


# Ordered list of (name, repair_function) pairs
_REPAIR_STRATEGIES: List[Tuple[str, Any]] = [
    ("trailing_comma_removal", lambda s: re.sub(r',\s*([}\]])', r'\1', s)),
    ("single_quote_replacement", lambda s: s.replace("'", '"')),
    ("unquoted_key_fix", lambda s: re.sub(r'(?<=[{,])\s*(\w+)\s*:', r' "\1":', s)),
    ("bracket_closing", _close_brackets),
    ("newline_escape", _escape_newlines_in_strings),
    ("control_char_removal", lambda s: re.sub(r'[\x00-\x1f\x7f-\x9f]', '', s)),
]


def repair_json(json_str: str) -> Tuple[Optional[str], List[str]]:
    """Attempt to repair malformed JSON.

    Returns (repaired_string_or_None, list_of_strategies_applied).
    """
    strategies_applied: List[str] = []

    # Try each repair individually first
    for name, fn in _REPAIR_STRATEGIES:
        try:
            repaired = fn(json_str)
            json.loads(repaired)
            strategies_applied.append(name)
            return repaired, strategies_applied
        except Exception as e:
            logger.debug("JSON repair attempt failed: %s", e)

    # Apply repairs cumulatively
    current = json_str
    for name, fn in _REPAIR_STRATEGIES:
        try:
            current = fn(current)
            strategies_applied.append(name)
        except Exception as e:
            logger.debug("Repair strategy %s failed: %s", name, e)

    # Final attempt with all repairs applied
    try:
        json.loads(current)
        return current, strategies_applied
    except json.JSONDecodeError:
        return None, strategies_applied


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate_response(
    parsed_json: Dict[str, Any],
    expected_schema: Type[BaseModel],
) -> Tuple[bool, Optional[BaseModel], Optional[str]]:
    """Validate parsed JSON against a Pydantic schema."""
    try:
        validated = expected_schema(**parsed_json)
        return True, validated, None
    except ValidationError as e:
        return False, None, str(e)


# ---------------------------------------------------------------------------
# Combined parse-and-validate
# ---------------------------------------------------------------------------

def parse_llm_response(
    response_text: str,
    expected_schema: Type[BaseModel],
) -> Tuple[Any, str, bool, List[str]]:
    """Parse an LLM response string into a validated Pydantic object.

    Returns:
        (validated_object, raw_json_string, repair_applied, strategies_used)

    Raises:
        JSONExtractionError – no JSON found
        JSONParseError      – JSON could not be parsed even after repair
        SchemaValidationError – parsed JSON doesn't match schema
    """
    json_str = extract_json(response_text)
    if json_str is None:
        raise JSONExtractionError(
            f"No JSON found in response: {response_text[:200]}..."
        )

    repair_applied = False
    strategies_used: List[str] = []

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        repaired, strategies_used = repair_json(json_str)
        if repaired is None:
            raise JSONParseError(
                f"Could not parse or repair JSON: {json_str[:200]}..."
            )
        parsed = json.loads(repaired)
        json_str = repaired
        repair_applied = True

    is_valid, validated_obj, error_msg = validate_response(parsed, expected_schema)
    if not is_valid:
        raise SchemaValidationError(f"Schema validation failed: {error_msg}")

    return validated_obj, json_str, repair_applied, strategies_used
