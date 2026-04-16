# Reproducibility Pack — File Listing

All files compiled into this paper-submission reproducibility bundle, with
raw file sizes (bytes) and the category each file belongs to.

Categories used below correspond one-to-one with the category list in the
submission instructions:

- **Source code** — Python scripts/modules for data loading, LLM orchestration, rewriting, analysis, and figure/table generation.
- **Prompt files** — Prompt templates and one-shot exemplars used by the pipeline.
- **Schema / validation** — Pydantic models, JSON Schema files, response-validation logic.
- **Rewrite-fidelity audit** — Deterministic audit code for numeric, temporal, polarity, and anchor checks.
- **Robustness-subset definition** — Code or config defining HF, NO_POL, NO_NUM, and related subset logic.
- **Bayesian model files** — Stan model sources and Bayesian runner code.
- **Sample-reconstruction / preprocessing** — Scripts that reconstruct the study sample and derived non-LLM artifacts from the public CFPB source data.
- **Software-version manifest** — Plain text/Markdown files listing software/package versions.
- **Setup instructions** — README and companion methodology/step documentation explaining how to run the code without secrets.
- **Configuration** — YAML/JSON/TOML/INI configuration data for audits and subset definitions.
- **Figure/table driver** — Scripts that regenerate manuscript figures and tables.
- **Lightweight derived-data spec** — Small metadata/manifests describing expected derived artifacts.

| File | Size (bytes) | Category |
|------|-------------:|----------|
| README.md | 12,087 | Setup instructions |
| cmdstan_version.txt | 8 | Software-version manifest |
| install-stan.py | 926 | Setup instructions |
| llm_fairness_experiment.py | 18,429 | Source code |
| make_subset_data.py | 1,450 | Sample-reconstruction / preprocessing |
| matched-guise-slot-preservation-audit.py | 757 | Rewrite-fidelity audit |
| matched_guise_robustness_subsets.py | 75,326 | Robustness-subset definition |
| regen_robustness_ladder_figure.py | 2,106 | Figure/table driver |
| remedy-reasons.py | 40,725 | Source code |
| requirements.lock | 1,732 | Software-version manifest |
| requirements.txt | 791 | Software-version manifest |
| small_simulation_illustration.py | 6,494 | Figure/table driver |
| contracts/__init__.py | 62 | Schema / validation |
| contracts/backfill_panels.py | 7,603 | Schema / validation |
| contracts/publication_manifest.json | 403 | Lightweight derived-data spec |
| contracts/reason_code_map.json | 772 | Lightweight derived-data spec |
| contracts/validate_publication_clean.py | 27,676 | Schema / validation |
| contracts/schemas/panel_ipw_robustness.schema.json | 4,728 | Schema / validation |
| contracts/schemas/panel_process_amplification.schema.json | 1,919 | Schema / validation |
| contracts/tests/__init__.py | 35 | Schema / validation |
| contracts/tests/test_render_contract.py | 8,811 | Schema / validation |
| contracts/validators/__init__.py | 451 | Schema / validation |
| contracts/validators/reason_notes.py | 3,978 | Schema / validation |
| contracts/validators/render_contract.py | 11,728 | Schema / validation |
| docs/bayesian-analysis.md | 35,418 | Bayesian model files |
| docs/cfpb-data.md | 11,455 | Sample-reconstruction / preprocessing |
| docs/llm-api-calls.md | 21,236 | Setup instructions |
| docs/llm-prompts.md | 21,992 | Prompt files |
| docs/matched-guise-robustness-subsets.md | 15,920 | Robustness-subset definition |
| docs/matched-guise-slot-preservation-audit.md | 12,513 | Rewrite-fidelity audit |
| docs/step-01-persona-estimation.md | 10,159 | Setup instructions |
| docs/step-02-remedy-decision.md | 6,155 | Setup instructions |
| docs/step-03-correlation-analysis.md | 7,634 | Setup instructions |
| docs/step-04-persona-rewriting.md | 9,877 | Setup instructions |
| docs/step-05-remedy-decision-rewritten.md | 12,685 | Setup instructions |
| docs/step-06-neutralised-complaint.md | 8,494 | Setup instructions |
| docs/step-07-remedy-decision-neutralised.md | 8,174 | Setup instructions |
| docs/step-08-severity-amplification.md | 11,235 | Setup instructions |
| docs/step-09-process-discrimination.md | 8,919 | Setup instructions |
| docs/text-attributes.md | 36,411 | Lightweight derived-data spec |
| docs/tiers-and-severities.md | 13,003 | Lightweight derived-data spec |
| models/stan/tier2_fixed_logit.stan | 1,041 | Bayesian model files |
| models/stan/tier2_hier_logit.stan | 1,824 | Bayesian model files |
| models/stan/tier2_hier_logit_flex.stan | 1,179 | Bayesian model files |
| models/stan/tier3_fixed_ordinal.stan | 1,817 | Bayesian model files |
| models/stan/tier3_hier_ordinal.stan | 2,708 | Bayesian model files |
| models/stan/tier3_hier_ordinal_flex.stan | 1,890 | Bayesian model files |
| one-shot-examples/one-shot-example-ai-decision-making.md | 3,280 | Prompt files |
| one-shot-examples/one-shot-example-bias-mitigation.md | 19,545 | Prompt files |
| one-shot-examples/one-shot-example-bridging-commentary.md | 2,705 | Prompt files |
| one-shot-examples/one-shot-example-correlation-vs-cause.md | 6,476 | Prompt files |
| one-shot-examples/one-shot-example-language-neutralisation.md | 5,742 | Prompt files |
| one-shot-examples/one-shot-example-linguistic-differences.md | 4,297 | Prompt files |
| one-shot-examples/one-shot-example-robustness-quality.md | 4,807 | Prompt files |
| one-shot-examples/one-shot-example-severity-stakes.md | 6,611 | Prompt files |
| one-shot-examples/one-shot-examples-conclusion.md | 7,833 | Prompt files |
| reproducibility/schemas/commentary-response.md | 1,397 | Schema / validation |
| reproducibility/schemas/dollar-amount-response.md | 1,881 | Schema / validation |
| reproducibility/schemas/neutralize-response.md | 1,434 | Schema / validation |
| reproducibility/schemas/persona-estimate-response.md | 2,548 | Schema / validation |
| reproducibility/schemas/persona-rewrite-response.md | 1,535 | Schema / validation |
| reproducibility/schemas/remedy-decision-response.md | 1,778 | Schema / validation |
| reproducibility/schemas/text-quality-rating-response.md | 2,842 | Schema / validation |
| slot_audit/__init__.py | 2,004 | Rewrite-fidelity audit |
| slot_audit/__main__.py | 104 | Rewrite-fidelity audit |
| slot_audit/aggregate_report.py | 21,557 | Rewrite-fidelity audit |
| slot_audit/cache_loader.py | 9,218 | Rewrite-fidelity audit |
| slot_audit/compare_slots.py | 3,444 | Rewrite-fidelity audit |
| slot_audit/extract_anchors.py | 2,702 | Rewrite-fidelity audit |
| slot_audit/extract_numeric.py | 5,128 | Rewrite-fidelity audit |
| slot_audit/extract_polarity.py | 5,868 | Rewrite-fidelity audit |
| slot_audit/extract_temporal.py | 4,113 | Rewrite-fidelity audit |
| slot_audit/slot_audit.py | 12,957 | Rewrite-fidelity audit |
| slot_audit/configs/event_lexicon.yaml | 2,639 | Configuration |
| slot_audit/configs/issue_anchors.yaml | 2,019 | Configuration |
| slot_audit/tests/__init__.py | 0 | Rewrite-fidelity audit |
| slot_audit/tests/conftest.py | 678 | Rewrite-fidelity audit |
| slot_audit/tests/pairs_fixture.csv | 4,466 | Rewrite-fidelity audit |
| slot_audit/tests/test_anchors.py | 3,536 | Rewrite-fidelity audit |
| slot_audit/tests/test_compare_logic.py | 14,636 | Rewrite-fidelity audit |
| slot_audit/tests/test_numeric.py | 3,270 | Rewrite-fidelity audit |
| slot_audit/tests/test_polarity.py | 3,630 | Rewrite-fidelity audit |
| slot_audit/tests/test_temporal.py | 2,420 | Rewrite-fidelity audit |
| src/__init__.py | 0 | Source code |
| src/analysis_cache.py | 26,607 | Source code |
| src/bayes_robustness.py | 132,295 | Bayesian model files |
| src/cache.py | 7,136 | Source code |
| src/data_loader.py | 26,905 | Sample-reconstruction / preprocessing |
| src/debiased_analysis.py | 27,643 | Source code |
| src/debiased_triage.py | 55,666 | Source code |
| src/diagnostics.py | 9,958 | Source code |
| src/estimable.py | 13,882 | Source code |
| src/experiment.py | 233,787 | Source code |
| src/figure_generator.py | 91,021 | Figure/table driver |
| src/ipw_robustness.py | 32,268 | Source code |
| src/json_utils.py | 7,774 | Source code |
| src/llm_client.py | 33,196 | Source code |
| src/mitigation_analysis.py | 86,240 | Source code |
| src/paper_bundle.py | 32,764 | Figure/table driver |
| src/persona_analysis.py | 17,149 | Source code |
| src/prereg_compliance.py | 22,603 | Source code |
| src/process_discrimination.py | 14,706 | Source code |
| src/prompts.py | 13,507 | Prompt files |
| src/regression_diagnostics.py | 16,742 | Source code |
| src/repeated_measures.py | 62,128 | Source code |
| src/reporter.py | 320,189 | Figure/table driver |
| src/reproducibility.py | 9,049 | Source code |
| src/schemas.py | 20,347 | Schema / validation |
| src/severity_analysis.py | 105,039 | Source code |
| src/sparsity_analysis.py | 30,350 | Source code |
| src/statistics.py | 53,342 | Source code |
| src/table_generator.py | 155,367 | Figure/table driver |
| src/text_features.py | 16,380 | Source code |
| src/text_rewriting_analysis.py | 26,535 | Source code |
| src/token_usage.py | 1,164 | Source code |
| src/worked_examples.py | 34,620 | Figure/table driver |

## Category roll-up

| Category | File count |
|----------|-----------:|
| Source code | 30 |
| Prompt files | 11 |
| Schema / validation | 17 |
| Rewrite-fidelity audit | 18 |
| Robustness-subset definition | 2 |
| Bayesian model files | 8 |
| Sample-reconstruction / preprocessing | 3 |
| Software-version manifest | 3 |
| Setup instructions | 10 |
| Configuration | 2 |
| Figure/table driver | 8 |
| Lightweight derived-data spec | 4 |
