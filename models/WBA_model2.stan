// removed the weights in favor of kappa and rho i think?

data {
  int<lower=1> n;
  array[n] int<lower=0, upper=7> trust1;
  array[n] int<lower=0, upper=7> trust2;
  array[n] int<lower=0, upper=7> group_trust_mean;
  int<lower=1> n_total_rating;  // pass 7 from R

  // prior hyperparameters
  real<lower=0> prior_kappa_mu;
  real<lower=0> prior_kappa_sigma;
}

parameters {
  // rho and kappa do what w_direct and w_social do?
  real<lower=0, upper=1> rho;
  real<lower=0>          kappa;
}

transformed parameters {
  vector[n] alpha_post = 0.5 + kappa * (
    (1 - rho) * to_vector(trust1) +
    rho        * to_vector(group_trust_mean)
  );
  vector[n] beta_post = 0.5 + kappa * (
    (1 - rho) * (n_total_rating - to_vector(trust1)) +
    rho        * (n_total_rating - to_vector(group_trust_mean))
  );
}

model {
  rho   ~ beta(2, 2);
  kappa ~ lognormal(log(prior_kappa_mu), prior_kappa_sigma);
  target += beta_binomial_lpmf(trust2 | n_total_rating, alpha_post, beta_post);
}

generated quantities {
  real rho_prior   = beta_rng(2, 2);
  real kappa_prior = lognormal_rng(log(prior_kappa_mu), prior_kappa_sigma);
  array[n] int  post_pred;
  array[n] int  prior_pred;
  array[n] real log_lik;
  for (i in 1:n) {
    // posterior predictive
    real theta_post  = beta_rng(alpha_post[i], beta_post[i]);
    post_pred[i]     = binomial_rng(n_total_rating, theta_post);
    // prior predictive — consistent with model structure
    real a_pr = 0.5 + kappa_prior * (
      (1 - rho_prior) * trust1[i] +
      rho_prior        * group_trust_mean[i]);
    real b_pr = 0.5 + kappa_prior * (
      (1 - rho_prior) * (n_total_rating - trust1[i]) +
      rho_prior        * (n_total_rating - group_trust_mean[i]));
    real theta_prior = beta_rng(fmax(0.01, a_pr), fmax(0.01, b_pr));
    prior_pred[i]    = binomial_rng(n_total_rating, theta_prior);
    // log likelihood for LOO
    log_lik[i] = beta_binomial_lpmf(trust2[i] | n_total_rating,
                                     alpha_post[i], beta_post[i]);
  }
}
