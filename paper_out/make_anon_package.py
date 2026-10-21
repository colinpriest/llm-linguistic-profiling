"""Stage an anonymised copy of the code, prompts, pre-registrations and results for blind review.

    python paper_out/make_anon_package.py [--out D:/tmp/anon_pkg]

Copies a curated file list (no git history, no .env, no raw CFPB data, no caches, no LLM outputs) and
redacts identifying strings. Prints the file list and any residual matches for a final manual check.
Publishing (e.g. via anonymous.4open.science) is done by the author, not by this script.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

INCLUDE = [
    "requirements.txt", "llm_fairness_experiment.py", "holdout_panel.py",
    "jev_pilot.py", "jev_arms.py", "jev_attributes.py",
    "src/*.py",
    "prereg/*.py", "prereg/*.md", "prereg/*.json", "prereg/*.txt",
    "paper_out/*.py", "paper_out/appendix_A_prompts.md",
    "paper_out/tables/*", "paper_out/figures/*.png", "paper_out/figures/*.pdf",
]
EXCLUDE_NAMES = {"README.md", "make_anon_package.py"}            # prereg/README.md references internal branch names; replaced by README_ANON.md
REDACT = [
    (re.compile(r"Colin( Priest)?", re.I), "the author"),
    (re.compile(r"colinpriest", re.I), "anonymous"),
    (re.compile(r"claude/project-thread-[a-z0-9]+"), "main"),
    (re.compile(r"[A-Za-z]:[\\/]+dev[\\/]+[^\s\"']+"), "<path>"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<email>"),
]
CHECK = re.compile(r"colin|priest|@gmail|D:[\\/]dev|project-thread", re.I)

README = """# Same Facts, Different Remedy: code, prompts, pre-registrations and results (anonymised)

This package accompanies a submission under blind review. It contains:

- `src/`, `holdout_panel.py`, `jev_*.py`, `llm_fairness_experiment.py`: data pipeline, rewrite and decision clients,
  and the Jev (System One) scoring client.
- `prereg/`: the five pre-registration documents and their frozen analysis scripts (`*.md` + `*.py`), the result file
  of every one-shot confirmatory analysis (`*_results.json`), and post hoc analyses, which are labelled as such.
- `paper_out/`: scripts that build every figure and table from the committed result files, the figures and tables,
  and Appendix A (verbatim prompts, generated from the code).

Not included: the CFPB narratives (public; https://www.consumerfinance.gov/data-research/consumer-complaints/),
rewritten texts and raw model outputs (available after review), API keys, and version-control history. The
pre-registration chronology (commit times and hashes) will be released with the de-anonymised repository.

Environment: Python 3.10+, `pip install -r requirements.txt`; keys for OpenAI, TypeSafe and (for the fidelity audit)
Anthropic are read from a `.env` file.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="D:/tmp/anon_pkg")
    out = Path(ap.parse_args().out)
    if out.exists():
        shutil.rmtree(out)
    files = sorted({p for pat in INCLUDE for p in Path(".").glob(pat) if p.is_file() and p.name not in EXCLUDE_NAMES
                    and "__pycache__" not in p.parts})
    for f in files:
        dest = out / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        if f.suffix in {".py", ".md", ".txt", ".json", ".csv", ".tex"}:
            txt = f.read_text(encoding="utf-8", errors="replace")
            for rx, rep in REDACT:
                txt = rx.sub(rep, txt)
            dest.write_text(txt, encoding="utf-8")
        else:
            shutil.copy2(f, dest)
    (out / "README_ANON.md").write_text(README, encoding="utf-8")
    residual = []
    for f in out.rglob("*"):
        if f.is_file() and f.suffix in {".py", ".md", ".txt", ".json", ".csv", ".tex"}:
            for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if CHECK.search(line):
                    residual.append(f"{f.relative_to(out)}:{i}: {line.strip()[:120]}")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"staged {len(files) + 1} files, {size / 1e6:.1f} MB, in {out}")
    for f in files:
        print("  ", f.as_posix())
    print("residual identifying matches:", len(residual))
    for r in residual[:50]:
        print("  ", r)


if __name__ == "__main__":
    main()
