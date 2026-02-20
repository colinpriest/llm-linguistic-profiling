# Reproducibility Bundle: Targeted Matched-Guise Tests of LLM Decision Pipelines

This bundle contains the complete experimental infrastructure and cached results for the paper **"Targeted Matched-Guise Tests of Linguistic-Style Sensitivity in LLM Decision Pipelines with Rewrite-Fidelity Filters"**. All results correspond to the canonical frozen run `20260220_101746`.

## 1. Directory Structure

* `/data`: The stratified sample of 20,000 CFPB complaints.
* `/prompts`: Versioned templates for persona estimation, matched-guise rewriting, and remedy allocation.
* `/schemas`: JSON schemas for validating structured model outputs.
* `/cache`: Frozen LLM response logs (JSON) enabling reproduction without API calls.
* `/audit_pipeline`: Python scripts for the deterministic hard-slot preservation audit.
* `/analysis`: Scripts for calculation, randomization tests, and GEE models.
* `/bayesian`: Stan model files and approximate-inference outputs.

## 2. Quick Start: Reproducing Table 5 (Primary Results)

To reproduce the primary dispersion metrics without an OpenAI API key:

1. Ensure Python 3.10+ is installed.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the primary analysis script:
   ```bash
   python analysis/reproduce_table_5.py --cache_path ./cache/20260220_101746/
   ```
4. The script will load the frozen responses, apply the fidelity filters, and output the metrics and p-values reported in the manuscript.

## 3. Integrity Audit Verification

Reviewers can verify the **Fidelity Audit** results (Section 5.4) by running the deterministic extractor over the raw source-rewrite pairs:

```bash
python audit_pipeline/run_audit.py --input_data ./data/matched_guise_sample.csv
```

This script uses regex-based rules for numeric, temporal, polarity, and anchor slots.

## 4. Synthetic Demonstration

For a high-level overview of the pipeline logic, see `demo_pipeline.ipynb`. This notebook processes a single synthetic complaint through the 36-guise rewrite operator and integrity filter to demonstrate the "targeted sensitivity" logic.

---

### Final Reviewer Note

All empirical claims in the paper trace directly to these folders. If you encounter any discrepancies between the cached outputs and the reported figures, please refer to the `provenance_id` stamped in the JSON headers.
