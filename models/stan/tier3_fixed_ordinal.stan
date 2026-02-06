// B2-fixed: Fixed-effect ordinal cumulative-logit for Tier 0/1/2
// Used when most complaints are singletons (>50% with exactly 1 obs),
// making complaint-level random intercepts unidentifiable.
//
// Note: ordered_logistic absorbs the intercept into the cutpoints.
// No separate alpha/intercept to avoid redundancy with cutpoints.
//
// Priors (tighter for stability)
//   beta      ~ Normal(0, 0.8)   [stronger shrinkage on coefficients]
//   c         ~ Normal(0, 1.5)   [tighter on cutpoints to avoid inf proposals]
//
// Uses reparameterization to guarantee ordered cutpoints:
//   c[1] = c_base
//   c[2] = c_base + softplus(c_diff_raw) + 0.1  (guaranteed > c[1])

data {
  int<lower=1> N;                       // observations
  int<lower=1> K;                       // predictors
  matrix[N, K] X;                       // design matrix
  array[N] int<lower=1, upper=3> y;     // ordinal tiers (1/2/3)
}

parameters {
  vector[K] beta;
  real c_base;                          // first cutpoint
  real c_diff_raw;                      // raw difference (unconstrained)
}

transformed parameters {
  ordered[2] c;
  c[1] = c_base;
  // softplus(x) = log(1 + exp(x)) is always positive; add 0.1 minimum gap
  c[2] = c_base + log1p_exp(c_diff_raw) + 0.1;
}

model {
  // Stronger shrinkage than Normal(0,1) to stabilise many interaction terms.
  // OR ~ 2.2 at 1 SD, still allows meaningful effects.
  beta ~ normal(0, 0.8);
  // Priors on reparameterized cutpoints
  c_base ~ normal(0, 1.5);
  c_diff_raw ~ normal(0, 1.0);  // softplus(0) ≈ 0.69, so typical gap ≈ 0.8

  y ~ ordered_logistic(X * beta, c);
}

generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = ordered_logistic_lpmf(y[n] | dot_product(X[n], beta), c);
  }
}
