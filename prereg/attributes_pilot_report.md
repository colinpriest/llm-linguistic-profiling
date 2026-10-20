# Attribute pilot results

Complaints 200, guises 14400

## Go / no-go
```
{
  "G1_discriminating_attributes": [
    "vocabulary",
    "grammar",
    "formality",
    "confidence",
    "politeness",
    "rudeness",
    "emotion",
    "entitlement",
    "coherence",
    "chronology",
    "argument",
    "request_clarity",
    "concision",
    "harm",
    "bank_error",
    "prior_attempts"
  ],
  "G1_pass": true,
  "G2_manipulated_delivery_structure": [
    "argument_E",
    "chronology_E",
    "coherence_E",
    "concision_E",
    "emotion_E",
    "entitlement_E",
    "formality_E",
    "grammar_E",
    "politeness_E",
    "rudeness_E",
    "vocabulary_E"
  ],
  "G2_pass": true,
  "G3_shares_gpt4o_tier": {
    "gpt-4o-mini:edu": 0.7138927643689851,
    "gpt-4o-mini:ses": 0.8190387527244342
  },
  "G3_pass": true,
  "drop_for_confirmatory_unreliable": [],
  "GO": true
}
```
## Certainty and reliability
```
{
  "vocabulary": {
    "family": "delivery",
    "mean_norm_entropy": 0.3592465064611207,
    "within_complaint_sd_E": 0.1125351154856147,
    "retest_spearman_E": 0.997983636526577,
    "retest_choice_agree": 0.988
  },
  "grammar": {
    "family": "delivery",
    "mean_norm_entropy": 0.531418756366935,
    "within_complaint_sd_E": 0.14059643056722745,
    "retest_spearman_E": 0.9957574275416028,
    "retest_choice_agree": 0.982
  },
  "formality": {
    "family": "delivery",
    "mean_norm_entropy": 0.35850960052817277,
    "within_complaint_sd_E": 0.29128499768266836,
    "retest_spearman_E": 0.9988890907785872,
    "retest_choice_agree": 0.982
  },
  "confidence": {
    "family": "delivery",
    "mean_norm_entropy": 0.32189306656577393,
    "within_complaint_sd_E": 0.09988935710973455,
    "retest_spearman_E": 0.9983598560022777,
    "retest_choice_agree": 0.99
  },
  "politeness": {
    "family": "delivery",
    "mean_norm_entropy": 0.3942436257937346,
    "within_complaint_sd_E": 0.13969752302556643,
    "retest_spearman_E": 0.9970440557659047,
    "retest_choice_agree": 0.99
  },
  "rudeness": {
    "family": "delivery",
    "mean_norm_entropy": 0.31404899624968896,
    "within_complaint_sd_E": 0.07867378534202042,
    "retest_spearman_E": 0.9963828177089071,
    "retest_choice_agree": 0.986
  },
  "emotion": {
    "family": "delivery",
    "mean_norm_entropy": 0.404593934854127,
    "within_complaint_sd_E": 0.12360740152384442,
    "retest_spearman_E": 0.9982480473154626,
    "retest_choice_agree": 0.99
  },
  "entitlement": {
    "family": "delivery",
    "mean_norm_entropy": 0.49908505585469554,
    "within_complaint_sd_E": 0.11900831952552235,
    "retest_spearman_E": 0.9979733793668345,
    "retest_choice_agree": 0.97
  },
  "coherence": {
    "family": "structure",
    "mean_norm_entropy": 0.5028110794311328,
    "within_complaint_sd_E": 0.06770735340153196,
    "retest_spearman_E": 0.9959809915748249,
    "retest_choice_agree": 0.984
  },
  "chronology": {
    "family": "structure",
    "mean_norm_entropy": 0.48083181344443104,
    "within_complaint_sd_E": 0.07329694597722353,
    "retest_spearman_E": 0.9961183736720118,
    "retest_choice_agree": 0.972
  },
  "argument": {
    "family": "structure",
    "mean_norm_entropy": 0.5620440674294175,
    "within_complaint_sd_E": 0.05022139446651904,
    "retest_spearman_E": 0.9938158720261184,
    "retest_choice_agree": 0.98
  },
  "request_clarity": {
    "family": "structure",
    "mean_norm_entropy": 0.4218771298366154,
    "within_complaint_sd_E": 0.06217126198711637,
    "retest_spearman_E": 0.9979664186991759,
    "retest_choice_agree": 0.974
  },
  "concision": {
    "family": "structure",
    "mean_norm_entropy": 0.6726247135867311,
    "within_complaint_sd_E": 0.10763768775212991,
    "retest_spearman_E": 0.9958003405075119,
    "retest_choice_agree": 0.962
  },
  "evidence": {
    "family": "content",
    "mean_norm_entropy": 0.4531851293893749,
    "within_complaint_sd_E": 0.042084871130905886,
    "retest_spearman_E": 0.9975873146759281,
    "retest_choice_agree": 0.978
  },
  "harm": {
    "family": "content",
    "mean_norm_entropy": 0.46963180611158734,
    "within_complaint_sd_E": 0.053323595756587176,
    "retest_spearman_E": 0.9975841806852438,
    "retest_choice_agree": 0.976
  },
  "bank_error": {
    "family": "content",
    "mean_norm_entropy": 0.509250168453358,
    "within_complaint_sd_E": 0.06294530270267774,
    "retest_spearman_E": 0.9976373539622364,
    "retest_choice_agree": 0.966
  },
  "money_demand": {
    "family": "content",
    "mean_norm_entropy": 0.1652800305538769,
    "within_complaint_sd_E": 0.04246774008992135,
    "retest_spearman_E": 0.9836919999140781,
    "retest_choice_agree": 0.974
  },
  "legal_basis": {
    "family": "content",
    "mean_norm_entropy": 0.17108575512518617,
    "within_complaint_sd_E": 0.04499154338198841,
    "retest_spearman_E": 0.9939818521717364,
    "retest_choice_agree": 0.996
  },
  "prior_attempts": {
    "family": "content",
    "mean_norm_entropy": 0.2903611593016459,
    "within_complaint_sd_E": 0.06724231811299591,
    "retest_spearman_E": 0.9975357966889019,
    "retest_choice_agree": 0.988
  }
}
```