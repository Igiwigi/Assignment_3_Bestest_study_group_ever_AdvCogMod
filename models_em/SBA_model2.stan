// removed the weights in favor of kappa and rho i think?
// this is equivalent to WBA with specific rho & kappa param (so kind of redundant)
// currently blowing up (has high pareto-k) making weighing impossible


data {
  int<lower=1> n;
  array[n] int<lower=1, upper=8> trust1;      // ← now 1-8
  array[n] int<lower=1, upper=8> trust2;      // ← now 1-8
  array[n] int<lower=1, upper=8> group_trust_mean;//NOTE: there is a mismatch here! 
  //group trust updates in our simulation, participant after participant, but here it assumes there's one "true" stable value
  //ideally, would fix this by either making this one adaptive or by changing the data simulation logic (so they match)
  //Not enough time for it right now though
  int<lower=1> n_total_rating;                // ← set to 8
}

parameters {
  real<lower=0, upper=1> rho;      // ADD THIS
  real<lower=0> kappa;             // ADD THIS
}

transformed parameters {
  vector<lower=0.001>[n] alpha_post = fmax(0.001, 0.5 + kappa * (
    (1 - rho) * to_vector(trust1) +
    rho * to_vector(group_trust_mean)
  ));
  vector<lower=0.001>[n] beta_post = fmax(0.001, 0.5 + kappa * (
    (1 - rho) * (n_total_rating - to_vector(trust1)) +
    rho * (n_total_rating - to_vector(group_trust_mean))
  ));
}

model {
  rho ~ beta(1, 1);
  kappa ~ exponential(1);
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

