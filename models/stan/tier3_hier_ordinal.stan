// B2: Hierarchical ordinal cumulative-logit for Tier 0/1/2
// Maps tiers: tier 0 -> y=1, tier 1 -> y=2, tier 2 -> y=3
// Non-centred parameterisation for complaint-level random intercept.
//
// Note: ordered_logistic absorbs the intercept into the cutpoints,
// so no separate alpha parameter is needed (unlike the logistic model).
//
// Priors
//   beta      ~ Normal(0, 1)
//   c         ~ Normal(0, 2)          [ordered cutpoints]
//   sigma_u   ~ Normal(0, 0.5)       [half-normal, lower=0]
//   z_u       ~ Normal(0, 1)          [non-centred]

data {
  int<lower=1> N;                       // observations
  int<lower=1> K;                       // predictors
  matrix[N, K] X;                       // design matrix
  int<lower=1> J;                       // number of complaints
  array[N] int<lower=1, upper=J> jj;    // complaint index
  array[N] int<lower=1, upper=3> y;     // ordinal tiers (1/2/3)
}

parameters {
  vector[K] beta;
  ordered[2] c;                         // cutpoints (absorb intercept)
  vector[J] z_u;
  real<lower=0> sigma_u;
}

transformed parameters {
  vector[J] u = z_u * sigma_u;
}

model {
  // Priors — tightened from Normal(0, 1.5) / Normal(0, 3) after
  // convergence failures (cutpoints overflowing to inf, rhat>1.6, ess<10).
  // Normal(0, 1) regularises the many interaction dummies while still
  // allowing OR ≈ 2.7 at 1 SD.  Cutpoint prior Normal(0, 2) keeps c
  // in a reasonable logit-scale range (±4 at 2 SD).
  beta ~ normal(0, 1);
  c ~ normal(0, 2);
  z_u ~ std_normal();
  // Half-normal(0, 0.5): 95% of mass below ~1.0 on the logit scale.
  // Exponential(1) (mean=1.0) was too diffuse — singleton complaints
  // (only one persona observation) have nearly unidentifiable random
  // effects, which drifted to extreme values and forced the ordered
  // cutpoints to ±inf.  Half-normal(0, 0.5) concentrates complaint-
  // level heterogeneity in a plausible range while still allowing the
  // data to push sigma_u higher if warranted.
  sigma_u ~ normal(0, 0.5);

  y ~ ordered_logistic(X * beta + u[jj], c);
}

generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = ordered_logistic_lpmf(y[n] | dot_product(X[n], beta) + u[jj[n]], c);
  }
}
