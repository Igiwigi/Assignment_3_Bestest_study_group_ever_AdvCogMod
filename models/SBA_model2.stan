

// removed the weights in favor of kappa and rho i think?
data {
  int<lower=1> n;
  array[n] int<lower=0, upper=7> trust1;
  array[n] int<lower=0, upper=7> trust2;
  array[n] int<lower=0, upper=7> group_trust_mean;
  int<lower=1> n_total_rating;  // pass 7 from R
}

transformed parameters {
  vector[n] alpha_post = 0.5 + (
    to_vector(trust1) +
    to_vector(group_trust_mean)
  );
  vector[n] beta_post = 0.5 + (
    (n_total_rating - to_vector(trust1)) +
    (n_total_rating - to_vector(group_trust_mean))
  );
}

model {
  target += beta_binomial_lpmf(trust2 | n_total_rating, alpha_post, beta_post);
}

generated quantities {
  array[n] int  post_pred;
  array[n] real log_lik;
  for (i in 1:n) {
    real theta   = beta_rng(alpha_post[i], beta_post[i]);
    post_pred[i] = binomial_rng(n_total_rating, theta);
    log_lik[i]   = beta_binomial_lpmf(trust2[i] | n_total_rating,
                                       alpha_post[i], beta_post[i]);
  }
}
