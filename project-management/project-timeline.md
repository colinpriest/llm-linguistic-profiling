## **Project Timeline: February 1 - March 1, 2026**

**Target Submission: February 25 (3-day buffer before March 1)**

### **Phase 1: Setup & Data Preparation (Feb 1-7)**

**February 1-2 (Weekend): Infrastructure Setup**

* Set up API access for both GPT-4o-mini and GPT-5.2-mini
* Design prompt templates for all 7 steps
* Create demographic persona definitions (36 combinations)
* Test API rate limits and cost estimation

**February 3-4: Complaint Sample Preparation**

* Select 5,000 complaints from CFPB database
* Stratify by severity (Low/Medium/High)
* Clean and standardize format
* Create unique identifiers for tracking

**February 5-7: Pilot Testing**

* Run pilot with 100 complaints through all 7 steps
* Debug prompt templates and response parsing
* Refine demographic persona language patterns
* Validate neutralization approach

### **Phase 2: Core Data Collection (Feb 8-21)**

**February 8-10: Steps 1-3 (Association Testing)**

* **Step 1** : Persona estimation on 5K complaints (both models)
* **Step 2** : Independent decision-making on same 5K complaints (both models)
* **Step 3** : Statistical association analysis
* **Daily target** : ~1,700 complaints processed per day

**February 11-14: Steps 4-5 (Causality Testing)**

* **Step 4** : Rewrite complaints in persona-specific styles (36 personas × severity stratified sample)
* **Step 5** : Decision-making on rewritten complaints + association testing
* **Focus** : Representative sample (~1K complaints × key personas)

**February 15-17: Steps 6-7 (Mitigation Testing)**

* **Step 6** : Neutral language rewrites of full 5K sample
* **Step 7** : Decision-making on neutral versions + association comparison
* **Quality check** : Manual review of neutralization effectiveness

**February 18-21: Data Validation & Supplementary Analysis**

* Cross-model comparison analysis
* Severity amplification analysis across all conditions
* Intersectional analysis (which demographics most affected)
* Data quality checks and missing value handling

### **Phase 3: Analysis & Writing (Feb 22-25)**

**February 22: Statistical Analysis**

* Association strength calculations (Steps 1-3)
* Causal effect sizes (Steps 4-5)
* Mitigation effectiveness (Steps 6-7)
* Model comparison statistics
* Intersectional demographic analysis

**February 23: Results Visualization**

* Create all figures and tables
* Effect size visualizations
* Demographic bias heatmaps
* Before/after mitigation comparisons

**February 24: Writing Sprint**

* Complete first draft of all sections
* Focus on clear management journal narrative
* Emphasize practical implications

**February 25: Final Review & Submission**

* Proofread and format
* Submit to Journal of Financial Econometrics ML Special Issue
* **Buffer days available: Feb 26-28**

### **Critical Success Factors**

**Daily Monitoring Checkpoints:**

* API costs tracking (could get expensive with 5K × multiple conditions)
* Response quality validation (ensure prompts working as intended)
* Statistical power checks (ensure sufficient sample sizes)

**Risk Mitigation:**

* **API limits** : Stagger requests across multiple days
* **Cost overruns** : Start with smaller sample if needed, scale up
* **Technical issues** : 3-day buffer allows for troubleshooting

**Parallel Processing Opportunities:**

* Run both models simultaneously where possible
* Batch process similar prompt types
* Pre-write introduction/methods while data collection runs
