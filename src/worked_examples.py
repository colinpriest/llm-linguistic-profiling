"""Worked-examples report: select stratified complaint examples and generate
an HTML + JSON companion to the main fairness dashboard.

The single public entry point is :func:`generate_worked_examples`.
"""

from __future__ import annotations

import html as html_mod
import json
import logging
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.schemas import (
    Complaint,
    NeutralizedComplaint,
    PersonaEstimate,
    PersonaRewrite,
    RemedyDecision,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Example selection
# ---------------------------------------------------------------------------

def _most_popular_persona(
    persona_estimates: List[PersonaEstimate],
) -> str:
    """Return the persona_label that appears most often across all models."""
    counts: Counter[str] = Counter()
    for pe in persona_estimates:
        counts[pe.persona_label] += 1
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def _build_case_index(
    complaints: List[Complaint],
    persona_estimates: List[PersonaEstimate],
    baseline_decisions: List[RemedyDecision],
    persona_rewrites: List[PersonaRewrite],
    persona_decisions: List[RemedyDecision],
    neutralized: List[NeutralizedComplaint],
    neutralized_decisions: List[RemedyDecision],
    rewrite_persona_estimates: List[Tuple[str, str, PersonaEstimate]],
    neutralized_persona_estimates: List[PersonaEstimate],
    models: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Build a per-case_id index aggregating all experiment artefacts.

    Returns ``{case_id: {complaint, estimates, baseline, rewrites,
    persona_decs, rewrite_estimates, neutral_text, neutral_decs,
    neutral_estimates}}`` with sub-dicts keyed by model.
    """
    idx: Dict[str, Dict[str, Any]] = {}

    for c in complaints:
        idx[c.case_id] = {
            "complaint": c,
            "estimates": {},            # model -> PersonaEstimate
            "baseline": {},             # model -> RemedyDecision
            "rewrites": {},             # model -> {persona_label -> PersonaRewrite}
            "persona_decs": {},         # model -> {persona_label -> RemedyDecision}
            "rewrite_estimates": {},    # model -> {persona_label -> PersonaEstimate}
            "neutral_text": {},         # model -> NeutralizedComplaint
            "neutral_decs": {},         # model -> RemedyDecision
            "neutral_estimates": {},    # model -> PersonaEstimate
        }

    for pe in persona_estimates:
        if pe.case_id in idx:
            idx[pe.case_id]["estimates"][pe.model] = pe

    for bd in baseline_decisions:
        if bd.case_id in idx and bd.variant == "baseline":
            idx[bd.case_id]["baseline"][bd.model] = bd

    for rw in persona_rewrites:
        if rw.case_id in idx:
            idx[rw.case_id]["rewrites"].setdefault(rw.model, {})[rw.persona_label] = rw

    for pd_ in persona_decisions:
        if pd_.case_id in idx and pd_.variant not in ("baseline", "neutralized"):
            idx[pd_.case_id]["persona_decs"].setdefault(pd_.model, {})[pd_.variant] = pd_

    # Re-estimated personas on rewritten text: (intended_label, model, estimate)
    for intended_label, model, est in rewrite_persona_estimates:
        if est.case_id in idx:
            idx[est.case_id]["rewrite_estimates"].setdefault(model, {})[intended_label] = est

    for nc in neutralized:
        if nc.case_id in idx:
            idx[nc.case_id]["neutral_text"][nc.model] = nc

    for nd in neutralized_decisions:
        if nd.case_id in idx and nd.variant == "neutralized":
            idx[nd.case_id]["neutral_decs"][nd.model] = nd

    # Re-estimated personas on neutralised text
    for nest in neutralized_persona_estimates:
        if nest.case_id in idx:
            idx[nest.case_id]["neutral_estimates"][nest.model] = nest

    return idx


def _tier_variation(case_data: Dict[str, Any], models: List[str]) -> float:
    """Score how much persona-tier variation exists for a case (higher = more bias)."""
    tiers: List[int] = []
    for m in models:
        for dec in case_data["persona_decs"].get(m, {}).values():
            tiers.append(dec.tier)
    if len(tiers) < 2:
        return 0.0
    return max(tiers) - min(tiers) + len(set(tiers)) / len(tiers)


def _mitigation_fixes_bias(case_data: Dict[str, Any], models: List[str]) -> bool:
    """True if neutralisation reduces tier variation across personas.

    We compare the spread of persona-rewrite tiers to the neutralised tier.
    If persona rewrites produce varied tiers but the neutralised tier is
    closer to the baseline, that demonstrates mitigation working.
    """
    for m in models:
        persona_tiers = [d.tier for d in case_data["persona_decs"].get(m, {}).values()]
        baseline = case_data["baseline"].get(m)
        neutral = case_data["neutral_decs"].get(m)
        if not persona_tiers or baseline is None or neutral is None:
            continue
        spread = max(persona_tiers) - min(persona_tiers)
        if spread == 0:
            continue
        # Mitigation "fixes" if neutral tier equals baseline (removes persona influence)
        # OR if neutral tier is closer to the median than extreme persona tiers
        if neutral.tier == baseline.tier:
            return True
        # At least neutral tier should be within the persona range
        if min(persona_tiers) <= neutral.tier <= max(persona_tiers):
            return True
    return False


def _has_severity_amplification(
    case_data: Dict[str, Any],
    models: List[str],
) -> bool:
    """True if this complaint shows severity-dependent bias patterns.

    High-severity complaints where persona variation is present.
    """
    c: Complaint = case_data["complaint"]
    if c.monetary_severity not in ("high",) and c.issue_severity not in ("high",):
        return False
    return _tier_variation(case_data, models) > 0


def _has_complete_data(case_data: Dict[str, Any], models: List[str]) -> bool:
    """True if the case has data for all models across all stages."""
    for m in models:
        if m not in case_data["estimates"]:
            return False
        if m not in case_data["baseline"]:
            return False
        if not case_data["persona_decs"].get(m):
            return False
        if m not in case_data["neutral_decs"]:
            return False
        if m not in case_data["neutral_text"]:
            return False
    return True


def _baseline_tier_for(case_data: Dict[str, Any], models: List[str]) -> Optional[int]:
    """Return the baseline tier from the first model (used for stratification)."""
    for m in models:
        bd = case_data["baseline"].get(m)
        if bd is not None:
            return bd.tier
    return None


def select_worked_examples(
    complaints: List[Complaint],
    persona_estimates: List[PersonaEstimate],
    baseline_decisions: List[RemedyDecision],
    persona_rewrites: List[PersonaRewrite],
    persona_decisions: List[RemedyDecision],
    neutralized: List[NeutralizedComplaint],
    neutralized_decisions: List[RemedyDecision],
    rewrite_persona_estimates: List[Tuple[str, str, PersonaEstimate]],
    neutralized_persona_estimates: List[PersonaEstimate],
    models: List[str],
) -> List[Dict[str, Any]]:
    """Select 6 stratified worked examples.

    Returns a list of dicts, each with keys:
        label, description, case_data (the full case index entry)

    The 6 examples are:
    1. Tier 0, non-popular persona, mitigation fixes bias
    2. Tier 1, non-popular persona, mitigation fixes bias
    3. Tier 2, non-popular persona, mitigation fixes bias
    4. Tier 1, non-popular persona, mitigation fixes bias (alternate)
    5. Tier 1, most popular persona, severity amplification
    6. Tier 2, most popular persona, severity amplification
    """
    popular = _most_popular_persona(persona_estimates)
    logger.info("Most popular persona: %s", popular)

    case_idx = _build_case_index(
        complaints, persona_estimates, baseline_decisions,
        persona_rewrites, persona_decisions, neutralized,
        neutralized_decisions, rewrite_persona_estimates,
        neutralized_persona_estimates, models,
    )

    # Partition cases by estimated persona popularity
    popular_cases: List[Tuple[str, Dict[str, Any]]] = []
    non_popular_cases: List[Tuple[str, Dict[str, Any]]] = []

    for cid, cdata in case_idx.items():
        if not _has_complete_data(cdata, models):
            continue
        # Check if any model estimated this case as the popular persona
        is_popular = any(
            cdata["estimates"].get(m) and cdata["estimates"][m].persona_label == popular
            for m in models
        )
        if is_popular:
            popular_cases.append((cid, cdata))
        else:
            non_popular_cases.append((cid, cdata))

    # Sort by tier variation (prefer cases with more variation = more interesting examples)
    non_popular_cases.sort(key=lambda x: _tier_variation(x[1], models), reverse=True)
    popular_cases.sort(key=lambda x: _tier_variation(x[1], models), reverse=True)

    # Selection targets
    targets = [
        # (tier, source_list, filter_fn, label, description)
        (0, non_popular_cases, _mitigation_fixes_bias,
         "Tier 0 — Mitigation Fix",
         "Non-popular persona, no-action recommendation where neutralisation corrects bias"),
        (1, non_popular_cases, _mitigation_fixes_bias,
         "Tier 1 — Mitigation Fix",
         "Non-popular persona, non-monetary action where neutralisation corrects bias"),
        (2, non_popular_cases, _mitigation_fixes_bias,
         "Tier 2 — Mitigation Fix",
         "Non-popular persona, monetary action where neutralisation corrects bias"),
        (1, non_popular_cases, _mitigation_fixes_bias,
         "Tier 1 — Mitigation Fix (Alt)",
         "Non-popular persona, non-monetary action showing alternative mitigation pattern"),
        (1, popular_cases, _has_severity_amplification,
         "Tier 1 — Severity Amplification",
         "Most popular persona, non-monetary action with severity-dependent bias"),
        (2, popular_cases, _has_severity_amplification,
         "Tier 2 — Severity Amplification",
         "Most popular persona, monetary action with severity-dependent bias"),
    ]

    selected: List[Dict[str, Any]] = []
    used_case_ids: set = set()

    for target_tier, source, filter_fn, label, desc in targets:
        best = None
        for cid, cdata in source:
            if cid in used_case_ids:
                continue
            bt = _baseline_tier_for(cdata, models)
            if bt != target_tier:
                continue
            if not filter_fn(cdata, models):
                continue
            best = (cid, cdata)
            break

        if best is None:
            # Relax: accept any tier from the source that passes the filter
            for cid, cdata in source:
                if cid in used_case_ids:
                    continue
                if not filter_fn(cdata, models):
                    continue
                best = (cid, cdata)
                break

        if best is None:
            # Last resort: any complete case from the source
            for cid, cdata in source:
                if cid in used_case_ids:
                    continue
                best = (cid, cdata)
                break

        if best is not None:
            cid, cdata = best
            used_case_ids.add(cid)
            selected.append({
                "label": label,
                "description": desc,
                "case_data": cdata,
            })
            logger.info("  Selected %s: case_id=%s (tier=%s)",
                        label, cid[:12], _baseline_tier_for(cdata, models))
        else:
            logger.warning("  Could not find example for: %s", label)

    return selected


# ---------------------------------------------------------------------------
# 2. JSON export
# ---------------------------------------------------------------------------

def _example_to_json(example: Dict[str, Any], models: List[str]) -> Dict[str, Any]:
    """Convert a selected example to a JSON-serialisable dict."""
    cdata = example["case_data"]
    c: Complaint = cdata["complaint"]

    result: Dict[str, Any] = {
        "label": example["label"],
        "description": example["description"],
        "case_id": c.case_id,
        "complaint_id": c.complaint_id,
        "narrative": c.narrative,
        "product": c.product,
        "issue": c.issue,
        "product_category": c.product_category,
        "issue_category": c.issue_category,
        "monetary_severity": c.monetary_severity,
        "issue_severity": c.issue_severity,
        "date_received": c.date_received,
        "models": {},
    }

    for m in models:
        model_data: Dict[str, Any] = {}

        # Persona estimate
        est = cdata["estimates"].get(m)
        if est:
            model_data["persona_estimate"] = {
                "persona_label": est.persona_label,
                "age": est.age,
                "geographic": est.geographic,
                "education": est.education,
                "socioeconomic": est.socioeconomic,
                "gender": est.gender,
                "esl": est.esl,
                "confidence": est.confidence,
                "linguistic_markers": est.linguistic_markers,
            }

        # Baseline decision
        bd = cdata["baseline"].get(m)
        if bd:
            model_data["baseline_decision"] = {
                "tier": bd.tier,
                "confidence": bd.confidence,
                "reasoning": bd.reasoning,
                "information_needed": bd.information_needed,
            }

        # Persona rewrites + decisions + re-estimated personas
        original_label = est.persona_label if est else ""
        baseline_tier = bd.tier if bd is not None else None
        rewrites_list = []
        rw_dict = cdata["rewrites"].get(m, {})
        pd_dict = cdata["persona_decs"].get(m, {})
        re_est_dict = cdata["rewrite_estimates"].get(m, {})
        for plabel in sorted(rw_dict.keys()):
            rw = rw_dict[plabel]
            dec = pd_dict.get(plabel)
            re_est = re_est_dict.get(plabel)
            entry: Dict[str, Any] = {
                "persona_label": plabel,
                "rewritten_complaint": rw.rewritten_narrative,
                "style_changes_made": rw.style_changes_made,
            }
            if re_est:
                entry["inferred_persona"] = {
                    "persona_label": re_est.persona_label,
                    "confidence": re_est.confidence,
                    "matches_target": re_est.persona_label == plabel,
                    "matches_original": re_est.persona_label == original_label,
                }
            if dec:
                entry["decision"] = {
                    "tier": dec.tier,
                    "confidence": dec.confidence,
                    "reasoning": dec.reasoning,
                    "information_needed": dec.information_needed,
                    "tier_matches_original": dec.tier == baseline_tier if baseline_tier is not None else None,
                }
            rewrites_list.append(entry)
        model_data["persona_rewrites"] = rewrites_list

        # Neutralised
        nc = cdata["neutral_text"].get(m)
        nd = cdata["neutral_decs"].get(m)
        n_est = cdata["neutral_estimates"].get(m)
        if nc:
            model_data["neutralised"] = {
                "neutralised_complaint": nc.neutralized_narrative,
                "markers_removed": nc.markers_removed,
            }
        if n_est:
            model_data["neutralised_persona_estimate"] = {
                "persona_label": n_est.persona_label,
                "confidence": n_est.confidence,
            }
        if nd:
            model_data["neutralised_decision"] = {
                "tier": nd.tier,
                "confidence": nd.confidence,
                "reasoning": nd.reasoning,
                "information_needed": nd.information_needed,
                "tier_matches_original": nd.tier == baseline_tier if baseline_tier is not None else None,
            }

        result["models"][m] = model_data

    return result


# ---------------------------------------------------------------------------
# 3. HTML generation
# ---------------------------------------------------------------------------

_TIER_LABELS = {0: "No Action", 1: "Non-Monetary", 2: "Monetary"}
_TIER_COLORS = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
_TIER_BG = {0: "#eafaf1", 1: "#fef9e7", 2: "#fdecea"}


def _esc(text: str) -> str:
    return html_mod.escape(str(text))


def _tier_badge(tier: int) -> str:
    label = _TIER_LABELS.get(tier, f"Tier {tier}")
    color = _TIER_COLORS.get(tier, "#999")
    bg = _TIER_BG.get(tier, "#f8f9fa")
    return (
        f'<span style="display:inline-block;padding:3px 10px;border-radius:12px;'
        f'font-size:0.85rem;font-weight:600;color:{color};background:{bg};'
        f'border:1px solid {color}">'
        f'Tier {tier} — {label}</span>'
    )


def _confidence_badge(conf: str) -> str:
    colors = {
        "confident_no_action": ("#27ae60", "#eafaf1"),
        "confident_action_needed": ("#e67e22", "#fef5e7"),
        "need_more_info": ("#e74c3c", "#fdecea"),
    }
    color, bg = colors.get(conf, ("#999", "#f8f9fa"))
    display = conf.replace("_", " ").title()
    return (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:10px;'
        f'font-size:0.8rem;color:{color};background:{bg}">{_esc(display)}</span>'
    )


def _match_flag(matches: bool, match_label: str, mismatch_label: str) -> str:
    """Render a small coloured match/mismatch indicator."""
    if matches:
        return (
            f'<span style="display:inline-block;padding:1px 7px;border-radius:8px;'
            f'font-size:0.75rem;font-weight:600;color:#27ae60;background:#eafaf1;'
            f'border:1px solid #27ae60;margin-left:4px">{_esc(match_label)}</span>'
        )
    return (
        f'<span style="display:inline-block;padding:1px 7px;border-radius:8px;'
        f'font-size:0.75rem;font-weight:600;color:#e74c3c;background:#fdecea;'
        f'border:1px solid #e74c3c;margin-left:4px">{_esc(mismatch_label)}</span>'
    )


def _severity_badge(level: str) -> str:
    colors = {"low": "#27ae60", "medium": "#f39c12", "high": "#e74c3c", "unknown": "#95a5a6"}
    color = colors.get(level, "#999")
    return (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:10px;'
        f'font-size:0.8rem;font-weight:600;color:white;background:{color}">'
        f'{_esc(level)}</span>'
    )


def _render_example_tab(example: Dict[str, Any], models: List[str]) -> str:
    """Render the content for one example tab."""
    cdata = example["case_data"]
    c: Complaint = cdata["complaint"]
    parts: List[str] = []

    # Description
    parts.append(f'<p style="color:#7f8c8d;margin-bottom:15px">{_esc(example["description"])}</p>')

    # --- Raw data section ---
    parts.append('<h3>Original Complaint</h3>')
    parts.append('<div style="display:flex;gap:15px;flex-wrap:wrap;margin-bottom:10px">')
    parts.append(f'<div><strong>Case ID:</strong> <code>{_esc(c.case_id[:12])}...</code></div>')
    parts.append(f'<div><strong>Product:</strong> {_esc(c.product)}</div>')
    parts.append(f'<div><strong>Issue:</strong> {_esc(c.issue)}</div>')
    parts.append(f'<div><strong>Date:</strong> {_esc(c.date_received)}</div>')
    parts.append('</div>')
    parts.append('<div style="display:flex;gap:10px;margin-bottom:10px">')
    parts.append(f'<div><strong>Monetary severity:</strong> {_severity_badge(c.monetary_severity)}</div>')
    parts.append(f'<div><strong>Issue severity:</strong> {_severity_badge(c.issue_severity)}</div>')
    parts.append('</div>')
    parts.append(
        f'<div style="background:#f8f9fa;border:1px solid #dee2e6;border-radius:6px;'
        f'padding:15px;margin-bottom:20px;font-size:0.9rem;line-height:1.7;'
        f'max-height:300px;overflow-y:auto;white-space:pre-wrap">{_esc(c.narrative)}</div>'
    )

    # --- Two-column model comparison ---
    n_models = len(models)
    col_width = f"{100 // max(n_models, 1)}%"
    parts.append(f'<div style="display:grid;grid-template-columns:repeat({n_models}, 1fr);gap:20px">')

    for m in models:
        parts.append(f'<div style="border:1px solid #dee2e6;border-radius:8px;padding:15px">')
        parts.append(f'<h3 style="text-align:center;margin-bottom:15px;color:#3498db">{_esc(m)}</h3>')

        # Persona estimate
        est = cdata["estimates"].get(m)
        if est:
            parts.append('<h4>Initial Persona Classification</h4>')
            parts.append(
                f'<div style="background:#eaf2f8;border-radius:6px;padding:10px;margin-bottom:12px">'
                f'<strong>{_esc(est.persona_label)}</strong>'
                f'<div style="font-size:0.85rem;color:#666;margin-top:4px">'
                f'Age: {_esc(est.age)} | Geo: {_esc(est.geographic)} | '
                f'Edu: {_esc(est.education)} | SES: {_esc(est.socioeconomic)} | '
                f'Gender: {_esc(est.gender)} | ESL: {_esc(est.esl)} | '
                f'Confidence: {est.confidence:.2f}</div>'
                f'</div>'
            )

        # Baseline decision
        bd = cdata["baseline"].get(m)
        if bd:
            parts.append('<h4>Initial Remedy Recommendation</h4>')
            parts.append(
                f'<div style="background:#f8f9fa;border-radius:6px;padding:10px;margin-bottom:12px">'
                f'{_tier_badge(bd.tier)} {_confidence_badge(bd.confidence)}'
                f'<div style="font-size:0.85rem;color:#555;margin-top:6px">{_esc(bd.reasoning)}</div>'
                f'</div>'
            )

        # Persona rewrites (collapsible)
        rw_dict = cdata["rewrites"].get(m, {})
        pd_dict = cdata["persona_decs"].get(m, {})
        re_est_dict = cdata["rewrite_estimates"].get(m, {})
        original_label = est.persona_label if est else ""
        baseline_tier = bd.tier if bd else None
        if rw_dict:
            parts.append('<h4>Alternate Personas (Causal Experiment)</h4>')
            parts.append(
                '<div style="max-height:500px;overflow-y:auto;border:1px solid #ecf0f1;'
                'border-radius:6px;padding:8px">'
            )
            for plabel in sorted(rw_dict.keys()):
                rw = rw_dict[plabel]
                dec = pd_dict.get(plabel)
                re_est = re_est_dict.get(plabel)

                # Tier badge + tier match flag
                if dec:
                    tier_html = _tier_badge(dec.tier)
                    if baseline_tier is not None:
                        tier_html += _match_flag(
                            dec.tier == baseline_tier,
                            "tier matches original",
                            "tier differs from original",
                        )
                else:
                    tier_html = '<span style="color:#999">No decision</span>'

                # Re-estimated persona + match flags
                inferred_html = ""
                if re_est:
                    matches_target = re_est.persona_label == plabel
                    matches_original = re_est.persona_label == original_label
                    inferred_html = (
                        f'<div style="font-size:0.82rem;color:#555;margin-top:4px">'
                        f'Inferred persona: <strong>{_esc(re_est.persona_label)}</strong> '
                        f'(confidence: {re_est.confidence:.2f})'
                        f'{_match_flag(matches_target, "matches target", "does not match target")}'
                        f'{_match_flag(matches_original, "matches original", "differs from original")}'
                        f'</div>'
                    )

                parts.append(
                    f'<details style="margin-bottom:6px;border-bottom:1px solid #f0f0f0;padding-bottom:6px">'
                    f'<summary style="cursor:pointer;font-weight:600;font-size:0.88rem;padding:4px 0">'
                    f'{_esc(plabel)} &nbsp; {tier_html}</summary>'
                    f'{inferred_html}'
                    f'<div style="padding:8px;font-size:0.85rem;background:#fafafa;border-radius:4px;'
                    f'margin-top:4px;max-height:200px;overflow-y:auto;white-space:pre-wrap">'
                    f'{_esc(rw.rewritten_narrative)}</div>'
                    f'</details>'
                )
            parts.append('</div>')

        # Neutralised complaint + decision
        nc = cdata["neutral_text"].get(m)
        nd = cdata["neutral_decs"].get(m)
        n_est = cdata["neutral_estimates"].get(m)
        if nc or nd:
            parts.append('<h4 style="margin-top:15px">Neutralised Complaint</h4>')
        if nc:
            parts.append(
                f'<div style="background:#f0f8f0;border:1px solid #c3e6cb;border-radius:6px;'
                f'padding:10px;margin-bottom:8px;font-size:0.88rem;max-height:200px;'
                f'overflow-y:auto;white-space:pre-wrap">{_esc(nc.neutralized_narrative)}</div>'
            )
            if nc.markers_removed:
                parts.append(
                    f'<div style="font-size:0.8rem;color:#666">Markers removed: '
                    f'{_esc(", ".join(nc.markers_removed))}</div>'
                )
        if n_est:
            parts.append(
                f'<div style="font-size:0.82rem;color:#555;margin-top:4px">'
                f'Inferred persona: <strong>{_esc(n_est.persona_label)}</strong> '
                f'(confidence: {n_est.confidence:.2f})</div>'
            )
        if nd:
            neutral_tier_flag = ""
            if baseline_tier is not None:
                neutral_tier_flag = _match_flag(
                    nd.tier == baseline_tier,
                    "tier matches original",
                    "tier differs from original",
                )
            parts.append(
                f'<div style="margin-top:6px">'
                f'{_tier_badge(nd.tier)} {_confidence_badge(nd.confidence)} {neutral_tier_flag}'
                f'<div style="font-size:0.85rem;color:#555;margin-top:4px">{_esc(nd.reasoning)}</div>'
                f'</div>'
            )

        parts.append('</div>')  # end model column

    parts.append('</div>')  # end grid

    return "\n".join(parts)


def _build_html(body_content: str, timestamp: str, models: List[str]) -> str:
    """Wrap body content in the full HTML document."""
    models_str = _esc(", ".join(models))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Worked Examples — LLM Fairness Experiment</title>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.6;
    color: #2c3e50;
    background: #f8f9fa;
    padding: 20px;
    max-width: 1400px;
    margin: 0 auto;
}}

h1 {{ font-size: 1.8rem; margin-bottom: 10px; color: #2c3e50; }}
h2 {{ font-size: 1.4rem; margin: 20px 0 10px; color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
h3 {{ font-size: 1.15rem; margin: 15px 0 8px; color: #2c3e50; }}
h4 {{ font-size: 1.0rem; margin: 10px 0 6px; color: #555; }}

.header {{
    background: linear-gradient(135deg, #2c3e50, #3498db);
    color: white;
    padding: 25px 30px;
    border-radius: 10px;
    margin-bottom: 20px;
}}
.header h1 {{ color: white; border: none; font-size: 2rem; }}
.header .subtitle {{ opacity: 0.85; font-size: 0.95rem; margin-top: 5px; }}

.tab-nav {{
    display: flex;
    flex-wrap: wrap;
    gap: 2px;
    background: #ecf0f1;
    border-radius: 8px 8px 0 0;
    padding: 4px 4px 0;
    margin-top: 10px;
}}
.tab-btn {{
    padding: 10px 18px;
    border: none;
    background: transparent;
    cursor: pointer;
    font-size: 0.9rem;
    font-weight: 500;
    color: #7f8c8d;
    border-radius: 6px 6px 0 0;
    transition: all 0.2s;
    white-space: nowrap;
}}
.tab-btn:hover {{ background: #dfe6e9; color: #2c3e50; }}
.tab-btn.active {{
    background: white;
    color: #3498db;
    font-weight: 600;
    box-shadow: 0 -2px 0 #3498db inset;
}}

.tab-content {{ display: none; background: white; padding: 25px; border-radius: 0 0 8px 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
.tab-content.active {{ display: block; }}

code {{ font-family: "SF Mono", "Consolas", "Liberation Mono", monospace; font-size: 0.88rem; background: #f0f0f0; padding: 1px 5px; border-radius: 3px; }}

details summary {{ list-style: none; }}
details summary::-webkit-details-marker {{ display: none; }}
details[open] summary {{ margin-bottom: 4px; }}

@media (max-width: 900px) {{
    .tab-btn {{ padding: 8px 12px; font-size: 0.8rem; }}
}}
</style>
</head>
<body>

<div class="header">
<h1>Worked Examples — LLM Fairness Experiment</h1>
<div class="subtitle">Models: {models_str} | Generated: {_esc(timestamp)}</div>
</div>

{body_content}

<script>
document.querySelectorAll('.tab-btn').forEach(function(btn) {{
    btn.addEventListener('click', function() {{
        var tabGroup = this.closest('.tab-group');
        tabGroup.querySelectorAll('.tab-btn').forEach(function(b) {{ b.classList.remove('active'); }});
        tabGroup.querySelectorAll('.tab-content').forEach(function(c) {{ c.classList.remove('active'); }});
        this.classList.add('active');
        var target = document.getElementById(this.getAttribute('data-tab'));
        if (target) target.classList.add('active');
    }});
}});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# 4. Public entry point
# ---------------------------------------------------------------------------

def generate_worked_examples(
    complaints: List[Complaint],
    persona_estimates: List[PersonaEstimate],
    baseline_decisions: List[RemedyDecision],
    persona_rewrites: List[PersonaRewrite],
    persona_decisions: List[RemedyDecision],
    neutralized: List[NeutralizedComplaint],
    neutralized_decisions: List[RemedyDecision],
    rewrite_persona_estimates: Optional[List[Tuple[str, str, PersonaEstimate]]] = None,
    neutralized_persona_estimates: Optional[List[PersonaEstimate]] = None,
    models: Optional[List[str]] = None,
    output_dir: str = "./output",
    timestamp: str = "",
) -> Tuple[str, str]:
    """Generate worked-examples HTML report and JSON data file.

    Parameters
    ----------
    complaints .. neutralized_decisions : Experiment result lists
    rewrite_persona_estimates : Re-estimated personas on rewritten text
        (list of (intended_label, model, PersonaEstimate) tuples)
    neutralized_persona_estimates : Re-estimated personas on neutralised text
    models : Model identifiers (e.g. ["gpt-4o-mini", "gpt-5-mini"])
    output_dir : Directory for output files
    timestamp : Experiment timestamp (must match dashboard filename)

    Returns
    -------
    (html_path, json_path) : Paths to the generated files
    """
    if models is None:
        models = []
    if rewrite_persona_estimates is None:
        rewrite_persona_estimates = []
    if neutralized_persona_estimates is None:
        neutralized_persona_estimates = []

    logger.info("Generating worked examples report...")

    # Select examples
    examples = select_worked_examples(
        complaints, persona_estimates, baseline_decisions,
        persona_rewrites, persona_decisions, neutralized,
        neutralized_decisions, rewrite_persona_estimates,
        neutralized_persona_estimates, models,
    )

    if not examples:
        logger.warning("No worked examples could be selected — skipping report generation")
        return ("", "")

    # --- Build HTML ---
    body_parts: List[str] = []
    body_parts.append('<div class="tab-group">')
    body_parts.append('<div class="tab-nav">')

    for i, ex in enumerate(examples):
        active = " active" if i == 0 else ""
        tid = f"ex-{i}"
        body_parts.append(
            f'<button class="tab-btn{active}" data-tab="{tid}">'
            f'{_esc(ex["label"])}</button>'
        )
    body_parts.append('</div>')

    for i, ex in enumerate(examples):
        active = " active" if i == 0 else ""
        tid = f"ex-{i}"
        body_parts.append(f'<div id="{tid}" class="tab-content{active}">')
        body_parts.append(_render_example_tab(ex, models))
        body_parts.append('</div>')

    body_parts.append('</div>')

    body = "\n".join(body_parts)
    full_html = _build_html(body, timestamp, models)

    # --- Write HTML ---
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = str(out_dir / f"worked-examples-{timestamp}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    logger.info("Worked examples HTML: %s (%d bytes)", html_path, len(full_html))

    # --- Build and write JSON ---
    json_data = [_example_to_json(ex, models) for ex in examples]
    json_path = str(out_dir / f"worked-examples-{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Worked examples JSON: %s", json_path)

    return html_path, json_path
