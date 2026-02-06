// B1-fixed: Fixed-effect logistic regression for Tier-2 indicator
// Used when most complaints are singletons (>50% with exactly 1 obs),
// making complaint-level random intercepts unidentifiable.
//
// Priors (tighter for stability)
//   beta      ~ Normal(0, 0.8)   [stronger shrinkage on coefficients]
//   alpha     ~ Normal(0, 2)

data {
  int<lower=1> N;                       // observations
  int<lower=1> K;                       // predictors
  matrix[N, K] X;                       // design matrix
  array[N] int<lower=0, upper=1> y;     // Tier-2 indicator
}

parameters {
  vector[K] beta;
  real alpha;
}

model {
  // Stronger shrinkage than Normal(0,1) to stabilise many interaction terms.
  // OR ~ 2.2 at 1 SD, still allows meaningful effects.
  beta ~ normal(0, 0.8);
  alpha ~ normal(0, 2);

  y ~ bernoulli_logit(alpha + X * beta);
}

generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = bernoulli_logit_lpmf(y[n] | alpha + dot_product(X[n], beta));
  }
}
