# Jev pilot summary

- Complaints: 200; texts scored: 15000; model versions: jev-1.13.0
- Cost: $1.2593 (1999 input tokens/text); projected full 2,000-complaint run: $12.59
- Latency p50/p95: 0.2997513000154868/0.3847566699827439 s

## Option-order sensitivity
```
{
  "argmax_agreement_fwd_vs_rev": 0.9142666666666667,
  "mean_abs_diff_e_tier": 0.06162666666666667,
  "mean_e_tier_fwd": 0.5461686666666667,
  "mean_e_tier_rev": 0.6016126666666666
}
```
## Originals: Jev vs GPT
```
{
  "jev_argmax_dist": {
    "0": 0.57,
    "1": 0.335,
    "2": 0.095
  },
  "jev_mean_probs": {
    "p0_fwd": 0.53335,
    "p1_fwd": 0.36460000000000004,
    "p2_fwd": 0.10195
  },
  "agreement": {
    "gpt-4o-mini": {
      "n": 200,
      "gpt_tier_distribution": {
        "0": 0.305,
        "1": 0.595,
        "2": 0.1
      },
      "exact_agreement": 0.585,
      "kappa": 0.327744704977119,
      "spearman_gpt_tier_vs_jev_e_tier": 0.6225845830817994
    },
    "gpt-5-mini": {
      "n": 200,
      "gpt_tier_distribution": {
        "0": 0.29,
        "1": 0.565,
        "2": 0.145
      },
      "exact_agreement": 0.65,
      "kappa": 0.4458956700704505,
      "spearman_gpt_tier_vs_jev_e_tier": 0.7047045763715937
    }
  }
}
```
## Style sensitivity across 36 guises (SD of persona means, permutation p)
### Rewrites by gpt-4o-mini
| measure | SD | p | complaints |
|---|---|---|---|
| jev_e_tier | 0.0118 | 0.0005 | 200 |
| jev_argmax_fwd | 0.0231 | 0.0325 | 200 |
| jev_median_tier | 0.0244 | 0.0005 | 200 |
| jev_p_cum2 | 0.0032 | 0.0295 | 200 |
| jev_argmax_tier2 | 0.0081 | 0.3643 | 200 |
| gpt-4o-mini_tier | 0.0381 | 0.0005 | 200 |
| gpt-4o-mini_tier2 | 0.0153 | 0.0010 | 200 |

Dimension gaps (max-min level mean): age Jev 0.014 vs GPT 0.008; geo Jev 0.003 vs GPT 0.022; edu Jev 0.010 vs GPT 0.049; ses Jev 0.012 vs GPT 0.042

### Rewrites by gpt-5-mini
| measure | SD | p | complaints |
|---|---|---|---|
| jev_e_tier | 0.0058 | 0.0085 | 200 |
| jev_argmax_fwd | 0.0175 | 0.3878 | 200 |
| jev_median_tier | 0.0152 | 0.2534 | 200 |
| jev_p_cum2 | 0.0024 | 0.0160 | 200 |
| jev_argmax_tier2 | 0.0078 | 0.1999 | 200 |
| gpt-5-mini_tier | 0.0231 | 0.4238 | 200 |
| gpt-5-mini_tier2 | 0.0134 | 0.1969 | 200 |

Dimension gaps (max-min level mean): age Jev 0.001 vs GPT 0.006; geo Jev 0.001 vs GPT 0.012; edu Jev 0.005 vs GPT 0.007; ses Jev 0.004 vs GPT 0.017

## Manipulation check (Jev persona recovery on guises)
| rewrite model | dim | chance | argmax=target | mean P(target) |
|---|---|---|---|---|
| gpt-4o-mini | age | 0.50 | 0.641 | 0.628 |
| gpt-4o-mini | geo | 0.33 | 0.341 | 0.339 |
| gpt-4o-mini | edu | 0.50 | 0.598 | 0.587 |
| gpt-4o-mini | ses | 0.33 | 0.361 | 0.366 |
| gpt-5-mini | age | 0.50 | 0.540 | 0.528 |
| gpt-5-mini | geo | 0.33 | 0.335 | 0.337 |
| gpt-5-mini | edu | 0.50 | 0.533 | 0.528 |
| gpt-5-mini | ses | 0.33 | 0.337 | 0.342 |

## Persona entropy (normalised; 1 = no signal)
```
{
  "original": {
    "age": 0.7490874090941004,
    "geo": 0.6197340934980055,
    "edu": 0.5185172786417493,
    "ses": 0.45085391962286225
  },
  "neutralized_by_gpt-4o-mini": {
    "age": 0.8076388674076846,
    "geo": 0.5806237069683075,
    "edu": 0.21651893032872213,
    "ses": 0.42562671952278974
  },
  "neutralized_by_gpt-5-mini": {
    "age": 0.8093901247802678,
    "geo": 0.5934205747297113,
    "edu": 0.3083354203414762,
    "ses": 0.43989628664066915
  },
  "guise_by_gpt-4o-mini": {
    "age": 0.6098649604843329,
    "geo": 0.5640087273045851,
    "edu": 0.5105335595119948,
    "ses": 0.40625131722360575
  },
  "guise_by_gpt-5-mini": {
    "age": 0.7388174316435744,
    "geo": 0.5990730861419649,
    "edu": 0.5244196092220595,
    "ses": 0.4445408748219587
  }
}
```
## Uncertainty vs sensitivity
```
{
  "gpt-4o-mini": {
    "spearman_orig_entropy_vs_guise_sd": 0.7021710594055934,
    "n": 200
  },
  "gpt-5-mini": {
    "spearman_orig_entropy_vs_guise_sd": 0.722129799930907,
    "n": 200
  }
}
```