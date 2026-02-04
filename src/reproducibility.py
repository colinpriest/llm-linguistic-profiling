"""Reproducibility bundle generation for the paper output pipeline.

Generates manifest files, git state, environment freeze, and prompt
fingerprints to ensure every run can be fully reproduced.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    """Return hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except (OSError, IOError):
        return "file_not_found"


def _sha256_str(text: str) -> str:
    """Return hex SHA-256 digest of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def generate_run_config_json(
    run_config: Dict[str, Any],
    manifest_dir: Path,
) -> Path:
    """Write run_config.json with all configuration parameters."""
    config = {
        "run_id": run_config.get("run_id", ""),
        "timestamp": run_config.get("timestamp", datetime.now().isoformat()),
        "seed": run_config.get("seed"),
        "num_complaints": run_config.get("num_complaints"),
        "causal_sample_pct": run_config.get("causal_sample_pct"),
        "materiality_threshold": run_config.get("materiality_threshold"),
        "models": run_config.get("models", []),
        "data_path": str(run_config.get("data_path", "")),
        "threads": run_config.get("threads"),
        "skip_steps": list(run_config.get("skip_steps", set())),
    }
    path = manifest_dir / "run_config.json"
    path.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    logger.info("Written: %s", path)
    return path


def generate_data_manifest_csv(
    complaints: list,
    data_path: str | Path,
    manifest_dir: Path,
) -> Path:
    """Write data_manifest.csv with input file hashes and sampling stats."""
    data_path = Path(data_path)
    rows = [
        {
            "file": str(data_path),
            "sha256": _sha256_file(data_path),
            "n_rows_sampled": len(complaints),
            "timestamp": datetime.now().isoformat(),
        }
    ]
    path = manifest_dir / "data_manifest.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "sha256", "n_rows_sampled", "timestamp"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Written: %s", path)
    return path


def generate_git_state_json(manifest_dir: Path) -> Path:
    """Capture git commit hash, branch, and dirty flag."""
    git_state: Dict[str, Any] = {
        "commit": "unknown",
        "branch": "unknown",
        "dirty": False,
        "timestamp": datetime.now().isoformat(),
    }
    try:
        git_state["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
        git_state["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
        git_state["dirty"] = status != ""
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Could not capture git state")

    path = manifest_dir / "git_state.json"
    path.write_text(json.dumps(git_state, indent=2), encoding="utf-8")
    logger.info("Written: %s", path)
    return path


def generate_env_freeze_txt(manifest_dir: Path) -> Path:
    """Write pip freeze output."""
    path = manifest_dir / "env_freeze.txt"
    try:
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True, stderr=subprocess.DEVNULL,
        )
        path.write_text(freeze, encoding="utf-8")
    except (subprocess.CalledProcessError, FileNotFoundError):
        path.write_text("# pip freeze unavailable\n", encoding="utf-8")
        logger.warning("Could not capture pip freeze")
    logger.info("Written: %s", path)
    return path


def generate_prompts_fingerprints_json(manifest_dir: Path) -> Path:
    """Write prompt templates with SHA-256 fingerprints."""
    from src.prompts import (
        PROMPT_VERSIONS,
        PERSONA_ESTIMATE_SYSTEM,
        PERSONA_ESTIMATE_USER,
        REMEDY_DECISION_SYSTEM,
        REMEDY_DECISION_USER,
        NEUTRALIZE_SYSTEM,
        NEUTRALIZE_USER,
        DOLLAR_EXTRACT_SYSTEM,
        DOLLAR_EXTRACT_USER,
        PERSONA_REWRITE_SYSTEM,
    )

    prompts = {
        "prompt_versions": PROMPT_VERSIONS,
        "templates": {
            "persona_estimate_system": {
                "sha256": _sha256_str(PERSONA_ESTIMATE_SYSTEM),
                "length": len(PERSONA_ESTIMATE_SYSTEM),
            },
            "persona_estimate_user": {
                "sha256": _sha256_str(PERSONA_ESTIMATE_USER),
                "length": len(PERSONA_ESTIMATE_USER),
            },
            "remedy_decision_system": {
                "sha256": _sha256_str(REMEDY_DECISION_SYSTEM),
                "length": len(REMEDY_DECISION_SYSTEM),
            },
            "remedy_decision_user": {
                "sha256": _sha256_str(REMEDY_DECISION_USER),
                "length": len(REMEDY_DECISION_USER),
            },
            "persona_rewrite_system": {
                "sha256": _sha256_str(PERSONA_REWRITE_SYSTEM),
                "length": len(PERSONA_REWRITE_SYSTEM),
            },
            "neutralize_system": {
                "sha256": _sha256_str(NEUTRALIZE_SYSTEM),
                "length": len(NEUTRALIZE_SYSTEM),
            },
            "neutralize_user": {
                "sha256": _sha256_str(NEUTRALIZE_USER),
                "length": len(NEUTRALIZE_USER),
            },
            "dollar_extract_system": {
                "sha256": _sha256_str(DOLLAR_EXTRACT_SYSTEM),
                "length": len(DOLLAR_EXTRACT_SYSTEM),
            },
            "dollar_extract_user": {
                "sha256": _sha256_str(DOLLAR_EXTRACT_USER),
                "length": len(DOLLAR_EXTRACT_USER),
            },
        },
    }

    path = manifest_dir / "prompts_fingerprints.json"
    path.write_text(json.dumps(prompts, indent=2), encoding="utf-8")
    logger.info("Written: %s", path)
    return path


def generate_manifest_json(
    run_config: Dict[str, Any],
    output_root: Path,
    manifest_dir: Path,
) -> Path:
    """Generate manifest.json with all output file paths and SHA-256 hashes.

    Should be called *after* all other outputs have been written so that
    file hashes are accurate.
    """
    manifest: Dict[str, Any] = {
        "run_id": run_config.get("run_id", ""),
        "generated_at": datetime.now().isoformat(),
        "config": {
            "seed": run_config.get("seed"),
            "num_complaints": run_config.get("num_complaints"),
            "models": run_config.get("models", []),
        },
        "outputs": {},
    }

    # Walk output tree and hash every file
    for file_path in sorted(output_root.rglob("*")):
        if file_path.is_file() and file_path.name != "manifest.json":
            rel = file_path.relative_to(output_root)
            manifest["outputs"][str(rel)] = {
                "sha256": _sha256_file(file_path),
                "size_bytes": file_path.stat().st_size,
            }

    path = manifest_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Written: %s", path)
    return path


def generate_reproducibility_bundle(
    run_config: Dict[str, Any],
    complaints: list,
    output_root: Path,
) -> Dict[str, Path]:
    """Generate all reproducibility files. Call manifest last."""
    manifest_dir = output_root / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    files: Dict[str, Path] = {}
    files["run_config"] = generate_run_config_json(run_config, manifest_dir)
    files["data_manifest"] = generate_data_manifest_csv(
        complaints, run_config.get("data_path", ""), manifest_dir,
    )
    files["git_state"] = generate_git_state_json(manifest_dir)
    files["env_freeze"] = generate_env_freeze_txt(manifest_dir)
    files["prompts_fingerprints"] = generate_prompts_fingerprints_json(manifest_dir)
    # manifest.json generated last (after all other outputs exist)
    return files
