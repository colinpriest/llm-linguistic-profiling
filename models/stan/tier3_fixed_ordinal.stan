// B2-fixed: Fixed-effect ordinal cumulative-logit for Tier 0/1/2
// Used when most complaints are singletons (>50% with exactly 1 obs),
// making complaint-level random intercepts unidentifiable.
//
// Note: ordered_logistic absorbs the intercept into the cutpoints.
//
// Priors
//   beta      ~ Normal(0, 1)
//   c         ~ Normal(0, 2)          [ordered cutpoints]

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
  beta ~ normal(0, 1);
  c ~ normal(0, 2);

  y ~ ordered_logistic(X * beta, c);
}

generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = ordered_logistic_lpmf(y[n] | dot_product(X[n], beta), c);
  }
}
