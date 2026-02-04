// B2: Hierarchical ordinal cumulative-logit for Tier 0/1/2
// Maps tiers: tier 0 -> y=1, tier 1 -> y=2, tier 2 -> y=3
// Non-centred parameterisation for complaint-level random intercept.
//
// Priors
//   beta      ~ Normal(0, 1.5)
//   alpha     ~ Normal(0, 3)
//   c         ~ Normal(0, 3)          [ordered cutpoints]
//   sigma_u   ~ Exponential(1)
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
  real alpha;
  ordered[2] c;                         // cutpoints
  vector[J] z_u;
  real<lower=0> sigma_u;
}

transformed parameters {
  vector[J] u = z_u * sigma_u;
}

model {
  beta ~ normal(0, 1.5);
  alpha ~ normal(0, 3);
  c ~ normal(0, 3);
  z_u ~ std_normal();
  sigma_u ~ exponential(1);

  y ~ ordered_logistic(alpha + X * beta + u[jj], c);
}

generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = ordered_logistic_lpmf(y[n] | alpha + dot_product(X[n], beta) + u[jj[n]], c);
  }
}
