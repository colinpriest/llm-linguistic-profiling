// B1-flex: Hierarchical logistic with per-column prior scales.
// Used in fallback 3 (stronger priors) — allows tighter regularisation
// on interaction terms (prior_sd=0.5) vs main effects (prior_sd=1.0).
//
// Priors
//   beta[k]   ~ Normal(0, prior_sd[k])   [per-column]
//   alpha     ~ Normal(0, 2)
//   sigma_u   ~ Normal(0, 0.5)            [half-normal, lower=0]
//   z_u       ~ Normal(0, 1)              [non-centred]

data {
  int<lower=1> N;
  int<lower=1> K;
  matrix[N, K] X;
  int<lower=1> J;
  array[N] int<lower=1, upper=J> jj;
  array[N] int<lower=0, upper=1> y;
  vector<lower=0>[K] prior_sd;            // per-column prior SD for beta
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
  beta ~ normal(0, prior_sd);
  alpha ~ normal(0, 2);
  z_u ~ std_normal();
  sigma_u ~ normal(0, 0.5);

  y ~ bernoulli_logit(alpha + X * beta + u[jj]);
}

generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = bernoulli_logit_lpmf(y[n] | alpha + dot_product(X[n], beta) + u[jj[n]]);
  }
}
