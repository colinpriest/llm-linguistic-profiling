"""Slot comparison logic: determine preservation status per family."""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Set

from . import SlotComparisonResult, SlotStatus


def compare_set_slots(
    source_canonicals: Set[str] | FrozenSet[str] | List[str],
    rewrite_canonicals: Set[str] | FrozenSet[str] | List[str],
    family: str,
) -> SlotComparisonResult:
    """Compare two canonical-string sets for a non-polarity slot family.

    Status rules
    -------------
    * Both empty → PRESERVED_ABSENCE
    * Source non-empty, rewrite empty → OMISSION
    * Source empty, rewrite non-empty → ADDITION
    * Both non-empty, sets equal → PRESERVED_PRESENT
    * Both non-empty, not equal → MUTATION
    """
    src = frozenset(source_canonicals)
    rw = frozenset(rewrite_canonicals)

    if not src and not rw:
        status = SlotStatus.PRESERVED_ABSENCE
    elif src and not rw:
        status = SlotStatus.OMISSION
    elif not src and rw:
        status = SlotStatus.ADDITION
    elif src == rw:
        status = SlotStatus.PRESERVED_PRESENT
    else:
        status = SlotStatus.MUTATION

    return SlotComparisonResult(
        family=family,
        status=status,
        source_canonicals=src,
        rewrite_canonicals=rw,
    )


def compare_polarity(
    source_events: Dict[str, str],
    rewrite_events: Dict[str, str],
) -> SlotComparisonResult:
    """Compare polarity maps with contradiction detection.

    Polarity-specific rules (when both non-empty):
    * Any shared key with opposite polarity → MUTATION_POLARITY_CONTRADICTION
    * Maps identical → PRESERVED_PRESENT
    * Otherwise → MUTATION
    """
    if not source_events and not rewrite_events:
        return SlotComparisonResult(
            family="polarity",
            status=SlotStatus.PRESERVED_ABSENCE,
            source_canonicals=frozenset(),
            rewrite_canonicals=frozenset(),
        )

    src_canon = frozenset(f"{k}:{v}" for k, v in source_events.items())
    rw_canon = frozenset(f"{k}:{v}" for k, v in rewrite_events.items())

    if source_events and not rewrite_events:
        return SlotComparisonResult(
            family="polarity",
            status=SlotStatus.OMISSION,
            source_canonicals=src_canon,
            rewrite_canonicals=frozenset(),
        )

    if not source_events and rewrite_events:
        return SlotComparisonResult(
            family="polarity",
            status=SlotStatus.ADDITION,
            source_canonicals=frozenset(),
            rewrite_canonicals=rw_canon,
        )

    # Both non-empty — check for contradictions
    has_contradiction = False
    shared_keys = set(source_events) & set(rewrite_events)
    for key in shared_keys:
        sp, rp = source_events[key], rewrite_events[key]
        if sp != rp and "AMBIG" not in (sp, rp):
            has_contradiction = True
            break

    if has_contradiction:
        status = SlotStatus.MUTATION_POLARITY_CONTRADICTION
    elif src_canon == rw_canon:
        status = SlotStatus.PRESERVED_PRESENT
    else:
        status = SlotStatus.MUTATION

    return SlotComparisonResult(
        family="polarity",
        status=status,
        source_canonicals=src_canon,
        rewrite_canonicals=rw_canon,
        has_contradiction=has_contradiction,
    )
