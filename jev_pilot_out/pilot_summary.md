# Jev pilot summary

- Complaints: 2000; texts scored: 150000; model versions: jev-1.13.0
- Cost: $12.6019 (2000 input tokens/text); projected full 2,000-complaint run: $12.60
- Latency p50/p95: 0.2792362000036519/0.3675215600072987 s

## Option-order sensitivity
```
{
  "argmax_agreement_fwd_vs_rev": 0.90888,
  "mean_abs_diff_e_tier": 0.06126186666666667,
  "mean_e_tier_fwd": 0.5306932666666667,
  "mean_e_tier_rev": 0.5872615333333334
}
```
## Originals: Jev vs GPT
```
{
  "jev_argmax_dist": {
    "0": 0.603,
    "1": 0.317,
    "2": 0.08
  },
  "jev_mean_probs": {
    "p0_fwd": 0.55653,
    "p1_fwd": 0.34267,
    "p2_fwd": 0.10077
  },
  "agreement": {
    "gpt-4o-mini": {
      "n": 2000,
      "gpt_tier_distribution": {
        "0": 0.372,
        "1": 0.5345,
        "2": 0.0935
      },
      "exact_agreement": 0.631,
      "kappa": 0.383734087103926,
      "spearman_gpt_tier_vs_jev_e_tier": 0.6320042497708301
    },
    "gpt-5-mini": {
      "n": 2000,
      "gpt_tier_distribution": {
        "0": 0.3365,
        "1": 0.526,
        "2": 0.1375
      },
      "exact_agreement": 0.656,
      "kappa": 0.4445776489327092,
      "spearman_gpt_tier_vs_jev_e_tier": 0.6705570678672071
    }
  }
}
```
## Style sensitivity across 36 guises (SD of persona means, permutation p)
### Rewrites by gpt-4o-mini
| measure | SD | p | complaints |
|---|---|---|---|
| jev_e_tier | 0.0099 | 0.0005 | 2000 |
| jev_argmax_fwd | 0.0171 | 0.0005 | 2000 |
| jev_median_tier | 0.0139 | 0.0005 | 2000 |
| jev_p_cum2 | 0.0017 | 0.0005 | 2000 |
| jev_argmax_tier2 | 0.0042 | 0.0005 | 2000 |
| gpt-4o-mini_tier | 0.0282 | 0.0005 | 2000 |
| gpt-4o-mini_tier2 | 0.0072 | 0.0005 | 2000 |

Dimension gaps (max-min level mean): age Jev 0.013 vs GPT 0.021; geo Jev 0.000 vs GPT 0.015; edu Jev 0.007 vs GPT 0.036; ses Jev 0.006 vs GPT 0.037

### Rewrites by gpt-5-mini
| measure | SD | p | complaints |
|---|---|---|---|
| jev_e_tier | 0.0046 | 0.0005 | 2000 |
| jev_argmax_fwd | 0.0078 | 0.0005 | 2000 |
| jev_median_tier | 0.0064 | 0.0005 | 2000 |
| jev_p_cum2 | 0.0015 | 0.0005 | 2000 |
| jev_argmax_tier2 | 0.0029 | 0.0130 | 2000 |
| gpt-5-mini_tier | 0.0079 | 0.3898 | 2000 |
| gpt-5-mini_tier2 | 0.0038 | 0.6362 | 2000 |

Dimension gaps (max-min level mean): age Jev 0.003 vs GPT 0.003; geo Jev 0.001 vs GPT 0.007; edu Jev 0.006 vs GPT 0.005; ses Jev 0.006 vs GPT 0.006

## Manipulation check (Jev persona recovery on guises)
| rewrite model | dim | chance | argmax=target | mean P(target) |
|---|---|---|---|---|
| gpt-4o-mini | age | 0.50 | 0.651 | 0.631 |
| gpt-4o-mini | geo | 0.33 | 0.343 | 0.340 |
| gpt-4o-mini | edu | 0.50 | 0.601 | 0.588 |
| gpt-4o-mini | ses | 0.33 | 0.360 | 0.366 |
| gpt-5-mini | age | 0.50 | 0.540 | 0.527 |
| gpt-5-mini | geo | 0.33 | 0.337 | 0.337 |
| gpt-5-mini | edu | 0.50 | 0.537 | 0.531 |
| gpt-5-mini | ses | 0.33 | 0.340 | 0.342 |

## Persona entropy (normalised; 1 = no signal)
```
{
  "original": {
    "age": 0.7475561436003508,
    "geo": 0.6375988811912066,
    "edu": 0.5858719034100385,
    "ses": 0.4689646763887211
  },
  "neutralized_by_gpt-4o-mini": {
    "age": 0.8083384419740289,
    "geo": 0.5869069836482399,
    "edu": 0.23244322546849103,
    "ses": 0.4289490215352095
  },
  "neutralized_by_gpt-5-mini": {
    "age": 0.8149735278686514,
    "geo": 0.6019673969364792,
    "edu": 0.3456488717897133,
    "ses": 0.4632202519422574
  },
  "guise_by_gpt-4o-mini": {
    "age": 0.6148247804005864,
    "geo": 0.5754948448653667,
    "edu": 0.5311074943064158,
    "ses": 0.42837311267104494
  },
  "guise_by_gpt-5-mini": {
    "age": 0.7371354358701416,
    "geo": 0.6092653276982354,
    "edu": 0.5776672533511873,
    "ses": 0.46403728248868464
  }
}
```
## Uncertainty vs sensitivity
```
{
  "gpt-4o-mini": {
    "spearman_orig_entropy_vs_guise_sd": 0.694017913148875,
    "n": 2000
  },
  "gpt-5-mini": {
    "spearman_orig_entropy_vs_guise_sd": 0.7191658816763796,
    "n": 2000
  }
}
```