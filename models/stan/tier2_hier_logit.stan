// B1: Hierarchical logistic regression for Tier-2 indicator
// Non-centred parameterisation for complaint-level random intercept.
//
// Priors
//   beta      ~ Normal(0, 1)
//   alpha     ~ Normal(0, 2)
//   sigma_u   ~ Normal(0, 0.5)       [half-normal, lower=0]
//   z_u       ~ Normal(0, 1)        [non-centred]

data {
  int<lower=1> N;                       // observations
  int<lower=1> K;                       // predictors
  matrix[N, K] X;                       // design matrix
  int<lower=1> J;                       // number of complaints
  array[N] int<lower=1, upper=J> jj;    // complaint index
  array[N] int<lower=0, upper=1> y;     // Tier-2 indicator
}

parameters {
  vector[K] beta;
  real alpha;
  vector[J] z_u;
  real<lower=0> sigma_u;
}

transformed parameters {
  vector[J] u = z_u * sigma_u;
}

model {
  // Priors — tightened from Normal(0, 1.5) / Normal(0, 3) after
  // convergence failures (rhat>1.2, ess<15) on issue_severity.
  // X includes main-effect + severity + interaction dummies (~30-40 cols);
  // Normal(0, 1) still allows OR ≈ 2.7 at 1 SD, which is a large effect
  // on the logit scale, while regularising the many near-zero interactions.
  beta ~ normal(0, 1);
  alpha ~ normal(0, 2);
  z_u ~ std_normal();
  // Half-normal(0, 0.5): 95% of mass below ~1.0 on the logit scale.
  // Exponential(1) (mean=1.0) was too diffuse — singleton complaints
  // (only one persona observation) have nearly unidentifiable random
  // effects, causing non-convergence (rhat>1.2, ess<15).
  sigma_u ~ normal(0, 0.5);

  // Likelihood
  y ~ bernoulli_logit(alpha + X * beta + u[jj]);
}

generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = bernoulli_logit_lpmf(y[n] | alpha + dot_product(X[n], beta) + u[jj[n]]);
  }
}
