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

data {
  int<lower=1> N;                       // observations
  int<lower=1> K;                       // predictors
  matrix[N, K] X;                       // design matrix
  array[N] int<lower=1, upper=3> y;     // ordinal tiers (1/2/3)
}

parameters {
  vector[K] beta;
  ordered[2] c;                         // cutpoints (absorb intercept)
}

model {
  // Stronger shrinkage than Normal(0,1) to stabilise many interaction terms.
  // OR ~ 2.2 at 1 SD, still allows meaningful effects.
  beta ~ normal(0, 0.8);
  // Tighter cutpoint prior keeps c in reasonable logit-scale range (±3 at 2 SD),
  // reducing "Cut-points not valid ordered vector" and "inf" proposal warnings.
  c ~ normal(0, 1.5);

  y ~ ordered_logistic(X * beta, c);
}

generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = ordered_logistic_lpmf(y[n] | dot_product(X[n], beta), c);
  }
}
