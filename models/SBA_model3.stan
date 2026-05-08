/// Variability only in kappa (concentration of evidence), rho (evidenc weighing) fixed
/// closer to being actual SBA?

/// testing an alternative model
data {
  int<lower=1> n;
  array[n] int<lower=0, upper=7> trust1;
  array[n] int<lower=0, upper=7> trust2;
  array[n] int<lower=0, upper=7> group_trust_mean;
  int<lower=1> n_total_rating;
  real<lower=0> prior_kappa_mu;
  real<lower=0> prior_kappa_sigma;
}
parameters {
  real<lower=0, upper=50> kappa;
}

//the same as the WBA but with hardcoded rho as 0.5
transformed parameters {
  vector[n] alpha_post = 0.5 + kappa * (
    0.5 * to_vector(trust1) +
    0.5 * to_vector(group_trust_mean)
  );
  vector[n] beta_post = 0.5 + kappa * (
    0.5 * (n_total_rating - to_vector(trust1)) +
    0.5 * (n_total_rating - to_vector(group_trust_mean))
  );
}

model {
  kappa ~ lognormal(log(prior_kappa_mu), prior_kappa_sigma);
  target += beta_binomial_lpmf(trust2 | n_total_rating, alpha_post, beta_post);
}
generated quantities {
  real kappa_prior = lognormal_rng(log(prior_kappa_mu), prior_kappa_sigma);
  array[n] int  post_pred;
  array[n] int  prior_pred;
  array[n] real log_lik;
  
  for (i in 1:n) {
    post_pred[i] = binomial_rng(n_total_rating,
                     beta_rng(alpha_post[i], beta_post[i]));

    real a_pr = 0.5 + kappa_prior * (
      0.5 * trust1[i] + 0.5 * group_trust_mean[i]);
    real b_pr = 0.5 + kappa_prior * (
      0.5 * (n_total_rating - trust1[i]) +
      0.5 * (n_total_rating - group_trust_mean[i]));
    prior_pred[i] = binomial_rng(n_total_rating,
                      beta_rng(fmax(0.01, a_pr), fmax(0.01, b_pr)));

    log_lik[i] = beta_binomial_lpmf(trust2[i] | n_total_rating,
                                     alpha_post[i], beta_post[i]);
  }
}

