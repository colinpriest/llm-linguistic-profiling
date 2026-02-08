"""
Diagnostic tracking for experiment errors, warnings, and fallback actions.

Collects issues throughout the experiment pipeline for reporting in the HTML dashboard.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DiagnosticLevel(Enum):
    """Severity level of a diagnostic message."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class DiagnosticCategory(Enum):
    """Category of diagnostic for grouping in reports."""
    MODEL_FIT = "model_fit"           # Statistical model fitting issues
    DATA_QUALITY = "data_quality"     # Data issues (missing values, insufficient N)
    LLM_API = "llm_api"               # LLM API errors
    DEPENDENCY = "dependency"         # Missing optional dependencies
    COMPUTATION = "computation"       # Numerical issues (convergence, singular matrix)
    VALIDATION = "validation"         # Input validation failures
    IO = "io"                         # File I/O issues
    OTHER = "other"


@dataclass
class DiagnosticEntry:
    """A single diagnostic entry."""

    level: DiagnosticLevel
    category: DiagnosticCategory
    module: str                       # Source module (e.g., "mitigation_analysis")
    message: str                      # Human-readable description
    fallback_action: Optional[str] = None  # What fallback was taken, if any
    details: Dict[str, Any] = field(default_factory=dict)  # Additional context
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "level": self.level.value,
            "category": self.category.value,
            "module": self.module,
            "message": self.message,
            "fallback_action": self.fallback_action,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class DiagnosticCollector:
    """Thread-safe collector for diagnostic messages throughout the experiment."""

    _instance: Optional["DiagnosticCollector"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "DiagnosticCollector":
        """Singleton pattern for global access."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._entries: List[DiagnosticEntry] = []
                    cls._instance._entry_lock = threading.Lock()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (for testing or new experiments)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._entries = []

    def add(
        self,
        level: DiagnosticLevel,
        category: DiagnosticCategory,
        module: str,
        message: str,
        fallback_action: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a diagnostic entry."""
        entry = DiagnosticEntry(
            level=level,
            category=category,
            module=module,
            message=message,
            fallback_action=fallback_action,
            details=details or {},
        )
        with self._entry_lock:
            self._entries.append(entry)

        # Also log to standard logger
        log_msg = f"[{category.value}] {message}"
        if fallback_action:
            log_msg += f" (fallback: {fallback_action})"

        if level == DiagnosticLevel.ERROR:
            logger.error(log_msg)
        elif level == DiagnosticLevel.WARNING:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

    def error(
        self,
        category: DiagnosticCategory,
        module: str,
        message: str,
        fallback_action: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Convenience method for adding an error."""
        self.add(DiagnosticLevel.ERROR, category, module, message, fallback_action, details)

    def warning(
        self,
        category: DiagnosticCategory,
        module: str,
        message: str,
        fallback_action: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Convenience method for adding a warning."""
        self.add(DiagnosticLevel.WARNING, category, module, message, fallback_action, details)

    def info(
        self,
        category: DiagnosticCategory,
        module: str,
        message: str,
        fallback_action: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Convenience method for adding an info message."""
        self.add(DiagnosticLevel.INFO, category, module, message, fallback_action, details)

    def get_entries(
        self,
        level: Optional[DiagnosticLevel] = None,
        category: Optional[DiagnosticCategory] = None,
    ) -> List[DiagnosticEntry]:
        """Get entries, optionally filtered by level or category."""
        with self._entry_lock:
            entries = list(self._entries)

        if level is not None:
            entries = [e for e in entries if e.level == level]
        if category is not None:
            entries = [e for e in entries if e.category == category]

        return entries

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all diagnostics for the report."""
        with self._entry_lock:
            entries = list(self._entries)

        # Count by level
        level_counts = {level.value: 0 for level in DiagnosticLevel}
        for entry in entries:
            level_counts[entry.level.value] += 1

        # Count by category
        category_counts = {cat.value: 0 for cat in DiagnosticCategory}
        for entry in entries:
            category_counts[entry.category.value] += 1

        # Count fallbacks vs no-fallback
        with_fallback = sum(1 for e in entries if e.fallback_action)
        without_fallback = len(entries) - with_fallback

        # Group entries for display
        errors = [e.to_dict() for e in entries if e.level == DiagnosticLevel.ERROR]
        warnings = [e.to_dict() for e in entries if e.level == DiagnosticLevel.WARNING]
        infos = [e.to_dict() for e in entries if e.level == DiagnosticLevel.INFO]

        return {
            "total_count": len(entries),
            "level_counts": level_counts,
            "category_counts": category_counts,
            "with_fallback": with_fallback,
            "without_fallback": without_fallback,
            "errors": errors,
            "warnings": warnings,
            "infos": infos,
            "all_entries": [e.to_dict() for e in entries],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert all entries to a dictionary."""
        return self.get_summary()


# Global accessor
def get_diagnostics() -> DiagnosticCollector:
    """Get the global diagnostic collector instance."""
    return DiagnosticCollector()


# Convenience functions for common patterns
def record_model_fit_error(
    module: str,
    model_type: str,
    target: str,
    error: str,
    fallback: Optional[str] = None,
) -> None:
    """Record a model fitting error."""
    get_diagnostics().error(
        category=DiagnosticCategory.MODEL_FIT,
        module=module,
        message=f"{model_type} model failed for target '{target}': {error}",
        fallback_action=fallback,
        details={"model_type": model_type, "target": target, "error": error},
    )


def record_model_fit_warning(
    module: str,
    model_type: str,
    target: str,
    warning: str,
    fallback: Optional[str] = None,
) -> None:
    """Record a model fitting warning."""
    get_diagnostics().warning(
        category=DiagnosticCategory.MODEL_FIT,
        module=module,
        message=f"{model_type} model warning for target '{target}': {warning}",
        fallback_action=fallback,
        details={"model_type": model_type, "target": target, "warning": warning},
    )


def record_data_quality_issue(
    module: str,
    issue: str,
    fallback: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Record a data quality issue."""
    get_diagnostics().warning(
        category=DiagnosticCategory.DATA_QUALITY,
        module=module,
        message=issue,
        fallback_action=fallback,
        details=details or {},
    )


def record_dependency_missing(
    module: str,
    dependency: str,
    fallback: Optional[str] = None,
) -> None:
    """Record a missing optional dependency."""
    get_diagnostics().warning(
        category=DiagnosticCategory.DEPENDENCY,
        module=module,
        message=f"Optional dependency '{dependency}' not installed",
        fallback_action=fallback,
        details={"dependency": dependency},
    )


def record_computation_error(
    module: str,
    operation: str,
    error: str,
    fallback: Optional[str] = None,
) -> None:
    """Record a computation error (convergence, numerical issues)."""
    get_diagnostics().error(
        category=DiagnosticCategory.COMPUTATION,
        module=module,
        message=f"{operation} failed: {error}",
        fallback_action=fallback,
        details={"operation": operation, "error": error},
    )


def record_llm_error(
    module: str,
    model: str,
    error: str,
    fallback: Optional[str] = None,
) -> None:
    """Record an LLM API error."""
    get_diagnostics().error(
        category=DiagnosticCategory.LLM_API,
        module=module,
        message=f"LLM API error ({model}): {error}",
        fallback_action=fallback,
        details={"model": model, "error": error},
    )
