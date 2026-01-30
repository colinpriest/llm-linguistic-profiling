
**Paper Overview**

This paper investigates whether large language models used in financial complaint resolution engage in linguistic profiling that leads to discriminatory outcomes, and whether such discrimination can be mitigated through language neutralization.

**Research Context and Gap**

While existing research has documented LLM discrimination based on dialect (particularly African American English versus Standard English), systematic testing across broader demographic dimensions—age, socioeconomic status, education level, and geographic background—remains sparse, especially in financial services contexts where decisions carry significant consumer welfare implications.

**Experimental Design**

The study employs a seven-step methodology designed to establish not merely association but causation between linguistic profiling and discriminatory outcomes:

*Steps 1-3* establish whether LLMs can infer demographic characteristics from complaint text and whether such inferences predict differential treatment. Importantly, the persona estimation and decision-making tasks are performed by separate model instances to prevent contamination.

*Steps 4-5* provide causal identification through a matched-guise approach: the same complaint is rewritten to reflect different demographic linguistic patterns while holding substantive content constant. If outcomes differ across rewrites, this proves that linguistic cues—not complaint merit—drive the disparity.

*Steps 6-7* test whether rewriting complaints into neutral language eliminates the discrimination, offering a practical mitigation strategy.

**Factorial Structure**

The experiment tests 36 demographic persona combinations (2 age groups × 3 geographic settings × 2 education levels × 3 socioeconomic classes) plus a neutral baseline across multiple models, three severity levels, and three complaint versions (original, persona-rewritten, neutralized), using a base sample of 5,000 complaints.

**Research Questions**

1. Do LLMs engage in linguistic profiling when processing financial complaints?
2. Is the relationship between linguistic profiling and differential outcomes causal?
3. Do newer models exhibit less linguistic discrimination than older models?
4. Does complaint severity amplify linguistic discrimination?
5. Does language neutralization effectively mitigate discrimination?
6. Which demographic intersections face the greatest discrimination, and do compound disadvantages emerge?

**Intended Contributions**

The paper aims to provide the first systematic evidence of linguistic discrimination in financial services AI, extend discrimination testing beyond the Black/White dialect binary to broader demographic dimensions, identify whether high-stakes decisions amplify bias, and offer practitioners an actionable mitigation strategy with measurable effectiveness.
