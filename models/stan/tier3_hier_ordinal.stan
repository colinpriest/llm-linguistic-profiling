// B2: Hierarchical ordinal cumulative-logit for Tier 0/1/2
// Maps tiers: tier 0 -> y=1, tier 1 -> y=2, tier 2 -> y=3
// Non-centred parameterisation for complaint-level random intercept.
//
// Note: ordered_logistic absorbs the intercept into the cutpoints,
// so no separate alpha parameter is needed (unlike the logistic model).
//
// Priors (tighter for stability)
//   beta      ~ Normal(0, 0.8)   [stronger shrinkage on coefficients]
//   c         ~ Normal(0, 1.5)   [tighter on cutpoints to avoid inf proposals]
//   sigma_u   ~ Normal(0, 0.5)   [half-normal, lower=0]
//   z_u       ~ Normal(0, 1)     [non-centred]

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
  // Stronger shrinkage than Normal(0,1) to stabilise many interaction terms.
  // OR ~ 2.2 at 1 SD, still allows meaningful effects.
  beta ~ normal(0, 0.8);
  // Tighter cutpoint prior keeps c in reasonable logit-scale range (±3 at 2 SD),
  // reducing "Cut-points not valid ordered vector" and "inf" proposal warnings.
  c ~ normal(0, 1.5);
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
