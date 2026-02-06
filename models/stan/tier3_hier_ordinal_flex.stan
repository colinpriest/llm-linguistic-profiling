// B2-flex: Hierarchical ordinal with per-column prior scales.
// Used in fallback 4 (stronger priors) — allows tighter regularisation
// on interaction terms (prior_sd=0.4) vs main effects (prior_sd=0.8).
//
// Note: ordered_logistic absorbs the intercept into the cutpoints.
//
// Uses reparameterization to guarantee ordered cutpoints:
//   c[1] = c_base
//   c[2] = c_base + softplus(c_diff_raw) + 0.1  (guaranteed > c[1])
//
// Priors
//   beta[k]   ~ Normal(0, prior_sd[k])   [per-column]
//   c_base    ~ Normal(0, 1.5)           [first cutpoint]
//   c_diff_raw ~ Normal(0, 1.0)          [controls gap]
//   sigma_u   ~ Normal(0, 0.5)           [half-normal, lower=0]
//   z_u       ~ Normal(0, 1)             [non-centred]

data {
  int<lower=1> N;
  int<lower=1> K;
  matrix[N, K] X;
  int<lower=1> J;
  array[N] int<lower=1, upper=J> jj;
  array[N] int<lower=1, upper=3> y;
  vector<lower=0>[K] prior_sd;            // per-column prior SD for beta
}

parameters {
  vector[K] beta;
  real c_base;                            // first cutpoint
  real c_diff_raw;                        // raw difference (unconstrained)
  vector[J] z_u;
  real<lower=0> sigma_u;
}

transformed parameters {
  vector[J] u = z_u * sigma_u;
  ordered[2] c;
  c[1] = c_base;
  // softplus(x) = log(1 + exp(x)) is always positive; add 0.1 minimum gap
  c[2] = c_base + log1p_exp(c_diff_raw) + 0.1;
}

model {
  beta ~ normal(0, prior_sd);
  // Priors on reparameterized cutpoints
  c_base ~ normal(0, 1.5);
  c_diff_raw ~ normal(0, 1.0);  // softplus(0) ≈ 0.69, so typical gap ≈ 0.8
  z_u ~ std_normal();
  sigma_u ~ normal(0, 0.5);

  y ~ ordered_logistic(X * beta + u[jj], c);
}

generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = ordered_logistic_lpmf(y[n] | dot_product(X[n], beta) + u[jj[n]], c);
  }
}
