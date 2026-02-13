"""Debiased extraction consistency analysis.

Analyses whether the debiased fact-extraction pipeline produces consistent
results across the 36 persona-style rewrites of each complaint.

Called from Step 8 of the main experiment pipeline and produces structured
results suitable for table generation and paper bundle inclusion.

Five analyses:
1. Extraction consistency — per-complaint, % of 36 rewrites matching original
2. (omitted — divergence examples are console-only debug output)
3. Cross-model comparison — how much do the two models agree
4. Consistency by persona dimension — does consistency vary by age/geo/edu/ses
5. Cross-model agreement by dimension — cross-model agreement grouped by dimension

Plus a worked-example finder for the appendix table.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.debiased_triage import (
    FactSet,
    TriageDecision,
    compare_factsets,
)
from src.schemas import PERSONA_GRID, persona_label

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Human-readable label maps (for worked example)
# ------------------------------------------------------------------

_ENT = {
    "bank_or_lender": "Bank/Lender",
    "credit_bureau": "Credit Bureau",
    "debt_collector": "Debt Collector",
    "merchant": "Merchant",
    "fintech_payment_platform": "Fintech/Payment",
    "mortgage_servicer": "Mortgage Svc",
    "student_loan_servicer": "Student Loan Svc",
    "insurance_company": "Insurance Co",
    "other_third_party": "Other 3rd Party",
    "unknown": "Unknown",
}

_COND = {
    "failed_to_validate_debt": "No debt validation",
    "failed_to_investigate_dispute": "No dispute investigation",
    "inaccurate_credit_reporting": "Inaccurate reporting",
    "failed_to_correct_reporting": "No report correction",
    "failed_to_provide_required_notice": "No required notice",
    "failed_to_provide_provisional_credit": "No provisional credit",
    "unauthorized_account_opened": "Unauthorized account",
    "unauthorized_transaction_processed": "Unauthorized transaction",
    "unauthorized_inquiry_on_credit": "Unauthorized inquiry",
    "improper_fees_charged": "Improper fees",
    "undisclosed_fees_charged": "Undisclosed fees",
    "payment_not_posted": "Payment not posted",
    "payment_posted_late": "Late payment posting",
    "account_closed_without_notice": "Acct closed no notice",
    "account_restricted_or_frozen": "Acct restricted/frozen",
    "funds_held_beyond_permitted_period": "Funds held too long",
    "failed_to_process_autopay": "Autopay failure",
    "failed_fraud_dispute_process": "Failed fraud dispute",
    "poor_communication_or_responsiveness": "Poor communication",
    "incorrect_account_status": "Wrong acct status",
    "failed_to_apply_promotion": "No promo applied",
    "incorrect_balance_reported": "Wrong balance",
    "failed_loss_mitigation": "Failed loss mitigation",
    "wrongful_repossession_or_foreclosure": "Wrongful repossession",
    "failed_to_release_lien": "Failed lien release",
    "system_outage_preventing_access": "System outage",
    "improper_collection_practices": "Improper collections",
    "failed_identity_theft_process": "Failed ID theft process",
    "other_alleged_conduct": "Other conduct",
}

_HARM = {
    "fees_or_charges_incurred": "Fees/charges",
    "funds_lost_or_inaccessible": "Funds lost",
    "credit_score_damage": "Credit damage",
    "denied_credit_or_service": "Denied service",
    "forced_third_party_costs": "3rd-party costs",
    "lost_property_or_collateral": "Lost property",
    "consequential_financial_loss": "Financial loss",
    "no_financial_harm_claimed": "No harm claimed",
    "financial_harm_vague_or_unquantified": "Vague harm",
}


# ------------------------------------------------------------------
# Result dataclasses
# ------------------------------------------------------------------

@dataclass
class ConsistencyRow:
    """Per-complaint consistency metrics (original vs 36 rewrites)."""
    model: str
    case_id: str
    orig_tier: Any  # int or "" for AVERAGE row
    tier: float
    entity: float
    nexus: float
    conduct: float
    harm: float
    evid: float
    n: Any  # int or "" for AVERAGE row


@dataclass
class CrossModelRow:
    """Per-complaint cross-model agreement."""
    section: str
    case_id: str
    tier_m1: Any
    tier_m2: Any
    tier_agree: Any  # bool or float for summary
    entity_agree: Any
    nexus_agree: Any
    conduct_agree: Any
    harm_agree: Any


@dataclass
class DimensionRow:
    """Consistency grouped by persona dimension value."""
    model: str
    dimension: str
    value: str
    tier: float
    entity: float
    nexus: float
    conduct: float
    harm: float
    evid: float
    n: int


@dataclass
class CrossModelDimensionRow:
    """Cross-model agreement grouped by persona dimension value."""
    dimension: str
    value: str
    tier: float
    entity: float
    nexus: float
    conduct: float
    harm: float
    n: int


@dataclass
class WorkedExampleRow:
    """One row of the worked example (original or persona rewrite)."""
    variant: str
    text: str
    vals: Dict[str, Dict[str, str]]  # model -> {tier, entity, nexus, conduct, harm, diff}


@dataclass
class DebiasedConsistencyResults:
    """Complete results from the debiased consistency analysis."""
    consistency_rows: List[ConsistencyRow] = field(default_factory=list)
    cross_model_rows: List[CrossModelRow] = field(default_factory=list)
    dimension_rows: List[DimensionRow] = field(default_factory=list)
    cross_model_dimension_rows: List[CrossModelDimensionRow] = field(default_factory=list)
    worked_example_rows: List[WorkedExampleRow] = field(default_factory=list)
    model_averages: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dict for inclusion in analysis_results."""
        return {
            "consistency": [asdict(r) for r in self.consistency_rows],
            "cross_model": [asdict(r) for r in self.cross_model_rows],
            "by_dimension": [asdict(r) for r in self.dimension_rows],
            "cross_model_by_dimension": [asdict(r) for r in self.cross_model_dimension_rows],
            "worked_example": [asdict(r) for r in self.worked_example_rows],
            "model_averages": self.model_averages,
        }


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def _extract_vals(fs: FactSet, dec: TriageDecision) -> Dict[str, str]:
    """Extract FactSet + TriageDecision fields as a dict of human-readable strings."""
    ent = _ENT.get(fs.primary_entity_type.value, fs.primary_entity_type.value)
    nx = "Yes" if fs.bank_nexus else "No"
    conducts = sorted(
        {_COND.get(af.conduct.value, af.conduct.value) for af in fs.alleged_facts}
    )
    harms = sorted(
        {_HARM.get(h.value, h.value) for h in fs.harm_types}
    )
    return {
        "tier": str(dec.tier.value),
        "entity": ent,
        "nexus": nx,
        "conduct": " + ".join(conducts) if conducts else "-",
        "harm": " + ".join(harms) if harms else "-",
    }


def _vals_key(fs: FactSet, dec: TriageDecision) -> tuple:
    """Hashable key for comparing extractions (used for variation detection)."""
    return (
        dec.tier.value,
        fs.primary_entity_type.value,
        fs.bank_nexus,
        frozenset(af.conduct.value for af in fs.alleged_facts),
        frozenset(h.value for h in fs.harm_types),
    )


def _xl_vals(
    prefix: str,
    fs: Optional[FactSet],
    dec: Optional[TriageDecision],
    orig: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Return dict of extracted values keyed with prefix, plus diff column."""
    empty = {
        f"{prefix}_tier": "", f"{prefix}_entity": "",
        f"{prefix}_nexus": "", f"{prefix}_conduct": "",
        f"{prefix}_harm": "", f"{prefix}_diff": "",
    }
    if fs is None or dec is None:
        return empty
    vals = _extract_vals(fs, dec)
    result = {
        f"{prefix}_tier": vals["tier"],
        f"{prefix}_entity": vals["entity"],
        f"{prefix}_nexus": vals["nexus"],
        f"{prefix}_conduct": vals["conduct"],
        f"{prefix}_harm": vals["harm"],
    }
    if orig is not None:
        changed = [k for k in ("tier", "entity", "nexus", "conduct", "harm")
                   if vals[k] != orig[k]]
        result[f"{prefix}_diff"] = ", ".join(changed) if changed else ""
    else:
        result[f"{prefix}_diff"] = ""
    return result


# ------------------------------------------------------------------
# Persona dimensions
# ------------------------------------------------------------------

# Short key names matching PERSONA_GRID dict keys (e.g. "geo", "edu", "ses"),
# not the full dimension names from ESTIMATE_DIMS/CAUSAL_DIMS (e.g.
# "geographic", "education", "socioeconomic").
_DIMENSIONS = [
    ("age", ["young", "old"]),
    ("geo", ["urban", "suburban", "rural"]),
    ("edu", ["high_school", "college"]),
    ("ses", ["low", "middle", "high"]),
]


def _avg(xs):
    """Return the arithmetic mean of *xs*, or 0.0 if the sequence is empty."""
    return sum(xs) / len(xs) if xs else 0.0


# ------------------------------------------------------------------
# Analysis functions
# ------------------------------------------------------------------

def _analysis_1_consistency(
    orig_fs: Dict[str, Dict[str, Optional[FactSet]]],
    orig_dec: Dict[str, Dict[str, TriageDecision]],
    rw_fs: Dict[str, Dict[str, Dict[str, Optional[FactSet]]]],
    rw_dec: Dict[str, Dict[str, Dict[str, TriageDecision]]],
    complaints: list,
    models: List[str],
) -> Tuple[List[ConsistencyRow], Dict[str, Dict[str, float]]]:
    """Analysis 1: Extraction consistency per complaint (original vs 36 rewrites)."""
    rows: List[ConsistencyRow] = []
    model_averages: Dict[str, Dict[str, float]] = {}

    for model in models:
        model_tier: List[float] = []
        model_entity: List[float] = []
        model_nexus: List[float] = []
        model_conduct: List[float] = []
        model_harm: List[float] = []
        model_evid: List[float] = []

        for c in complaints:
            ofs = orig_fs.get(model, {}).get(c.case_id)
            odec = orig_dec.get(model, {}).get(c.case_id)
            if ofs is None or odec is None:
                continue

            n = 0
            tier_m = entity_m = nexus_m = conduct_m = harm_m = evid_m = 0
            for plabel, rfs in rw_fs.get(model, {}).get(c.case_id, {}).items():
                rdec = rw_dec.get(model, {}).get(c.case_id, {}).get(plabel)
                if rfs is None or rdec is None:
                    continue
                n += 1
                diffs = compare_factsets(ofs, rfs)
                if rdec.tier == odec.tier:
                    tier_m += 1
                if "entity_type" not in diffs:
                    entity_m += 1
                if "bank_nexus" not in diffs:
                    nexus_m += 1
                if "conduct_diff" not in diffs:
                    conduct_m += 1
                if "harm_diff" not in diffs:
                    harm_m += 1
                if "evidence_strength_diff" not in diffs:
                    evid_m += 1

            if n == 0:
                continue

            rates = [x / n for x in (tier_m, entity_m, nexus_m, conduct_m, harm_m, evid_m)]
            model_tier.append(rates[0])
            model_entity.append(rates[1])
            model_nexus.append(rates[2])
            model_conduct.append(rates[3])
            model_harm.append(rates[4])
            model_evid.append(rates[5])

            rows.append(ConsistencyRow(
                model=model, case_id=c.case_id,
                orig_tier=odec.tier.value,
                tier=rates[0], entity=rates[1], nexus=rates[2],
                conduct=rates[3], harm=rates[4], evid=rates[5],
                n=n,
            ))

        if model_tier:
            avgs = {
                "tier": _avg(model_tier), "entity": _avg(model_entity),
                "nexus": _avg(model_nexus), "conduct": _avg(model_conduct),
                "harm": _avg(model_harm), "evid": _avg(model_evid),
            }
            model_averages[model] = avgs
            rows.append(ConsistencyRow(
                model=model, case_id="AVERAGE", orig_tier="",
                tier=avgs["tier"], entity=avgs["entity"], nexus=avgs["nexus"],
                conduct=avgs["conduct"], harm=avgs["harm"], evid=avgs["evid"],
                n="",
            ))

    return rows, model_averages


def _analysis_3_cross_model(
    orig_fs: Dict[str, Dict[str, Optional[FactSet]]],
    orig_dec: Dict[str, Dict[str, TriageDecision]],
    rw_fs: Dict[str, Dict[str, Dict[str, Optional[FactSet]]]],
    rw_dec: Dict[str, Dict[str, Dict[str, TriageDecision]]],
    complaints: list,
    models: List[str],
) -> List[CrossModelRow]:
    """Analysis 3: Cross-model comparison on originals + rewrites."""
    if len(models) < 2:
        return []
    m1, m2 = models[0], models[1]
    rows: List[CrossModelRow] = []

    # --- Originals ---
    n_comp = 0
    agree: Dict[str, int] = defaultdict(int)
    for c in complaints:
        fs1 = orig_fs.get(m1, {}).get(c.case_id)
        fs2 = orig_fs.get(m2, {}).get(c.case_id)
        d1 = orig_dec.get(m1, {}).get(c.case_id)
        d2 = orig_dec.get(m2, {}).get(c.case_id)
        if fs1 is None or fs2 is None or d1 is None or d2 is None:
            continue
        n_comp += 1
        diffs = compare_factsets(fs1, fs2)
        checks = {
            "tier": d1.tier == d2.tier,
            "entity": "entity_type" not in diffs,
            "nexus": "bank_nexus" not in diffs,
            "conduct": "conduct_diff" not in diffs,
            "harm": "harm_diff" not in diffs,
        }
        for k, v in checks.items():
            if v:
                agree[k] += 1
        rows.append(CrossModelRow(
            section="originals", case_id=c.case_id,
            tier_m1=d1.tier.value, tier_m2=d2.tier.value,
            tier_agree=checks["tier"], entity_agree=checks["entity"],
            nexus_agree=checks["nexus"], conduct_agree=checks["conduct"],
            harm_agree=checks["harm"],
        ))

    if n_comp:
        rows.append(CrossModelRow(
            section="originals", case_id="AGREEMENT",
            tier_m1="", tier_m2="",
            tier_agree=agree["tier"] / n_comp,
            entity_agree=agree["entity"] / n_comp,
            nexus_agree=agree["nexus"] / n_comp,
            conduct_agree=agree["conduct"] / n_comp,
            harm_agree=agree["harm"] / n_comp,
        ))

    # --- Rewrites (aggregated tier agreement) ---
    rw_tier_agree: List[float] = []
    for c in complaints:
        n = 0
        tier_match = 0
        for plabel in rw_fs.get(m1, {}).get(c.case_id, {}):
            fs1 = rw_fs[m1][c.case_id].get(plabel)
            fs2 = rw_fs.get(m2, {}).get(c.case_id, {}).get(plabel)
            d1 = rw_dec.get(m1, {}).get(c.case_id, {}).get(plabel)
            d2 = rw_dec.get(m2, {}).get(c.case_id, {}).get(plabel)
            if fs1 is None or fs2 is None or d1 is None or d2 is None:
                continue
            n += 1
            if d1.tier == d2.tier:
                tier_match += 1
        if n > 0:
            rw_tier_agree.append(tier_match / n)

    if rw_tier_agree:
        avg_rw = sum(rw_tier_agree) / len(rw_tier_agree)
        rows.append(CrossModelRow(
            section="rewrites", case_id="TIER_AGREEMENT",
            tier_m1="", tier_m2="",
            tier_agree=avg_rw, entity_agree="",
            nexus_agree="", conduct_agree="", harm_agree="",
        ))

    return rows


def _analysis_4_by_dimension(
    orig_fs: Dict[str, Dict[str, Optional[FactSet]]],
    orig_dec: Dict[str, Dict[str, TriageDecision]],
    rw_fs: Dict[str, Dict[str, Dict[str, Optional[FactSet]]]],
    rw_dec: Dict[str, Dict[str, Dict[str, TriageDecision]]],
    complaints: list,
    models: List[str],
) -> List[DimensionRow]:
    """Analysis 4: Consistency by persona dimension (within-model)."""
    rows: List[DimensionRow] = []
    for model in models:
        for dim_key, dim_values in _DIMENSIONS:
            for dval in dim_values:
                tier_m = entity_m = nexus_m = conduct_m = harm_m = evid_m = 0
                n_total = 0

                for c in complaints:
                    ofs = orig_fs.get(model, {}).get(c.case_id)
                    odec = orig_dec.get(model, {}).get(c.case_id)
                    if ofs is None or odec is None:
                        continue

                    for persona in PERSONA_GRID:
                        if persona[dim_key] != dval:
                            continue
                        plabel = persona_label(**persona)
                        rfs = rw_fs.get(model, {}).get(c.case_id, {}).get(plabel)
                        rdec = rw_dec.get(model, {}).get(c.case_id, {}).get(plabel)
                        if rfs is None or rdec is None:
                            continue
                        n_total += 1
                        diffs = compare_factsets(ofs, rfs)
                        if rdec.tier == odec.tier:
                            tier_m += 1
                        if "entity_type" not in diffs:
                            entity_m += 1
                        if "bank_nexus" not in diffs:
                            nexus_m += 1
                        if "conduct_diff" not in diffs:
                            conduct_m += 1
                        if "harm_diff" not in diffs:
                            harm_m += 1
                        if "evidence_strength_diff" not in diffs:
                            evid_m += 1

                if n_total == 0:
                    continue
                rates = [x / n_total for x in (tier_m, entity_m, nexus_m, conduct_m, harm_m, evid_m)]
                rows.append(DimensionRow(
                    model=model, dimension=dim_key, value=dval,
                    tier=rates[0], entity=rates[1], nexus=rates[2],
                    conduct=rates[3], harm=rates[4], evid=rates[5],
                    n=n_total,
                ))
    return rows


def _analysis_5_cross_model_by_dimension(
    rw_fs: Dict[str, Dict[str, Dict[str, Optional[FactSet]]]],
    rw_dec: Dict[str, Dict[str, Dict[str, TriageDecision]]],
    complaints: list,
    models: List[str],
) -> List[CrossModelDimensionRow]:
    """Analysis 5: Cross-model agreement by persona dimension."""
    if len(models) < 2:
        return []
    m1, m2 = models[0], models[1]
    rows: List[CrossModelDimensionRow] = []

    for dim_key, dim_values in _DIMENSIONS:
        for dval in dim_values:
            tier_a = entity_a = nexus_a = conduct_a = harm_a = 0
            n_total = 0

            for c in complaints:
                for persona in PERSONA_GRID:
                    if persona[dim_key] != dval:
                        continue
                    plabel = persona_label(**persona)
                    fs1 = rw_fs.get(m1, {}).get(c.case_id, {}).get(plabel)
                    fs2 = rw_fs.get(m2, {}).get(c.case_id, {}).get(plabel)
                    d1 = rw_dec.get(m1, {}).get(c.case_id, {}).get(plabel)
                    d2 = rw_dec.get(m2, {}).get(c.case_id, {}).get(plabel)
                    if fs1 is None or fs2 is None or d1 is None or d2 is None:
                        continue
                    n_total += 1
                    diffs = compare_factsets(fs1, fs2)
                    if d1.tier == d2.tier:
                        tier_a += 1
                    if "entity_type" not in diffs:
                        entity_a += 1
                    if "bank_nexus" not in diffs:
                        nexus_a += 1
                    if "conduct_diff" not in diffs:
                        conduct_a += 1
                    if "harm_diff" not in diffs:
                        harm_a += 1

            if n_total == 0:
                continue
            rates = [x / n_total for x in (tier_a, entity_a, nexus_a, conduct_a, harm_a)]
            rows.append(CrossModelDimensionRow(
                dimension=dim_key, value=dval,
                tier=rates[0], entity=rates[1], nexus=rates[2],
                conduct=rates[3], harm=rates[4],
                n=n_total,
            ))
    return rows


def _find_worked_example(
    orig_fs: Dict[str, Dict[str, Optional[FactSet]]],
    orig_dec: Dict[str, Dict[str, TriageDecision]],
    rw_fs: Dict[str, Dict[str, Dict[str, Optional[FactSet]]]],
    rw_dec: Dict[str, Dict[str, Dict[str, TriageDecision]]],
    rewrite_texts: Dict[str, Dict[str, Optional[str]]],
    complaints: list,
    models: List[str],
) -> List[WorkedExampleRow]:
    """Find a complaint showing cross-model + within-model disagreement.

    Returns the worked-example rows (original + 36 persona rewrites), or
    an empty list if no qualifying complaint is found.
    """
    if len(models) < 2:
        return []
    m1, m2 = models[0], models[1]

    example = None
    for c in complaints:
        fs1 = orig_fs.get(m1, {}).get(c.case_id)
        fs2 = orig_fs.get(m2, {}).get(c.case_id)
        d1 = orig_dec.get(m1, {}).get(c.case_id)
        d2 = orig_dec.get(m2, {}).get(c.case_id)
        if any(x is None for x in (fs1, fs2, d1, d2)):
            continue

        # Criterion 1: cross-model disagreement on original
        cross_diffs = compare_factsets(fs1, fs2)
        if d1.tier == d2.tier and not cross_diffs:
            continue

        # Criterion 2: within-m1 variation across rewrites
        m1_keys = set()
        for plabel_i, rfs_i in rw_fs.get(m1, {}).get(c.case_id, {}).items():
            rdec_i = rw_dec.get(m1, {}).get(c.case_id, {}).get(plabel_i)
            if rfs_i is not None and rdec_i is not None:
                m1_keys.add(_vals_key(rfs_i, rdec_i))
        if len(m1_keys) <= 1:
            continue

        # Criterion 3: within-m2 variation across rewrites
        m2_keys = set()
        for plabel_i, rfs_i in rw_fs.get(m2, {}).get(c.case_id, {}).items():
            rdec_i = rw_dec.get(m2, {}).get(c.case_id, {}).get(plabel_i)
            if rfs_i is not None and rdec_i is not None:
                m2_keys.add(_vals_key(rfs_i, rdec_i))
        if len(m2_keys) <= 1:
            continue

        example = c
        break

    if example is None:
        logger.info("Worked example: no complaint found meeting all three disagreement criteria")
        return []

    cid = example.case_id
    rows: List[WorkedExampleRow] = []

    # Original extraction values
    orig_v: Dict[str, Dict[str, str]] = {}
    model_vals: Dict[str, Dict[str, str]] = {}
    for model in models:
        efs = orig_fs.get(model, {}).get(cid)
        edec = orig_dec.get(model, {}).get(cid)
        if efs is not None and edec is not None:
            vals = _extract_vals(efs, edec)
            orig_v[model] = vals
            model_vals[model] = _xl_vals(model, efs, edec)
        else:
            model_vals[model] = _xl_vals(model, None, None)

    rows.append(WorkedExampleRow(
        variant="ORIGINAL",
        text=example.narrative,
        vals=model_vals,
    ))

    # Persona rewrites
    for persona in PERSONA_GRID:
        plabel = persona_label(**persona)
        rtext = rewrite_texts.get(cid, {}).get(plabel, "")

        model_vals = {}
        for model in models:
            rfs = rw_fs.get(model, {}).get(cid, {}).get(plabel)
            rdec = rw_dec.get(model, {}).get(cid, {}).get(plabel)
            model_vals[model] = _xl_vals(model, rfs, rdec, orig_v.get(model))

        rows.append(WorkedExampleRow(
            variant=plabel,
            text=rtext or "",
            vals=model_vals,
        ))

    logger.info("Worked example found: %s", cid)
    return rows


# ------------------------------------------------------------------
# Top-level analysis entry point
# ------------------------------------------------------------------

def analyze_extraction_consistency(
    orig_fs: Dict[str, Dict[str, Optional[FactSet]]],
    orig_dec: Dict[str, Dict[str, TriageDecision]],
    rw_fs: Dict[str, Dict[str, Dict[str, Optional[FactSet]]]],
    rw_dec: Dict[str, Dict[str, Dict[str, TriageDecision]]],
    rewrite_texts: Dict[str, Dict[str, Optional[str]]],
    complaints: list,
    models: List[str],
) -> DebiasedConsistencyResults:
    """Run all debiased extraction consistency analyses.

    Parameters
    ----------
    orig_fs : model -> case_id -> FactSet (or None)
    orig_dec : model -> case_id -> TriageDecision
    rw_fs : model -> case_id -> persona_label -> FactSet (or None)
    rw_dec : model -> case_id -> persona_label -> TriageDecision
    rewrite_texts : case_id -> persona_label -> rewritten text (for worked example)
    complaints : list of Complaint objects from the causal sample
    models : list of model identifiers

    Returns
    -------
    DebiasedConsistencyResults with all analysis outputs.
    """
    logger.info("Running debiased extraction consistency analysis "
                "(%d complaints, %d model(s))", len(complaints), len(models))

    consistency_rows, model_averages = _analysis_1_consistency(
        orig_fs, orig_dec, rw_fs, rw_dec, complaints, models
    )
    cross_model_rows = _analysis_3_cross_model(
        orig_fs, orig_dec, rw_fs, rw_dec, complaints, models
    )
    dimension_rows = _analysis_4_by_dimension(
        orig_fs, orig_dec, rw_fs, rw_dec, complaints, models
    )
    cross_model_dimension_rows = _analysis_5_cross_model_by_dimension(
        rw_fs, rw_dec, complaints, models
    )
    worked_example_rows = _find_worked_example(
        orig_fs, orig_dec, rw_fs, rw_dec, rewrite_texts, complaints, models
    )

    logger.info("Debiased consistency: %d consistency rows, %d cross-model rows, "
                "%d dimension rows, %d worked-example rows",
                len(consistency_rows), len(cross_model_rows),
                len(dimension_rows), len(worked_example_rows))

    return DebiasedConsistencyResults(
        consistency_rows=consistency_rows,
        cross_model_rows=cross_model_rows,
        dimension_rows=dimension_rows,
        cross_model_dimension_rows=cross_model_dimension_rows,
        worked_example_rows=worked_example_rows,
        model_averages=model_averages,
    )
