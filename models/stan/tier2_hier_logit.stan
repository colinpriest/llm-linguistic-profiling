// B1: Hierarchical logistic regression for Tier-2 indicator
// Non-centred parameterisation for complaint-level random intercept.
//
// Priors
//   beta      ~ Normal(0, 1.5)
//   alpha     ~ Normal(0, 3)
//   sigma_u   ~ Exponential(1)
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
  // Priors
  beta ~ normal(0, 1.5);
  alpha ~ normal(0, 3);
  z_u ~ std_normal();
  sigma_u ~ exponential(1);

  // Likelihood
  y ~ bernoulli_logit(alpha + X * beta + u[jj]);
}

generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = bernoulli_logit_lpmf(y[n] | alpha + dot_product(X[n], beta) + u[jj[n]]);
  }
}
