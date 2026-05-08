
// this is equivalent to WBA with specific rho & kappa param (so kind of redundant)
// currently blowing up (has high pareto-k) making weighing impossible

data {
  int<lower=1> n;
  array[n] int<lower=1, upper=8> trust1;      // ← change to 1-8
  array[n] int<lower=1, upper=8> trust2;      // ← change to 1-8
  array[n] int<lower=1, upper=8> group_trust_mean; // ← change to 1-8
  //NOTE: there is a mismatch here! 
  //group trust updates in our simulation, participant after participant, but here it assumes there's one "true" stable value
  //ideally, would fix this by either making this one adaptive or by changing the data simulation logic (so they match)
  //Not enough time for it right now though
  int<lower=1> n_total_rating;                // ← set to 8
}

transformed parameters {
   //hardcoded 0.5 prior, like in examples
  vector[n] alpha_post = 0.5 + to_vector(group_trust_mean) + to_vector(trust1);
  vector[n] beta_post  = 0.5 + (n_total_rating - to_vector(group_trust_mean)) + (n_total_rating - to_vector(trust1));
}


model {
  target += beta_binomial_lpmf(trust2 | n_total_rating, alpha_post, beta_post);
}

generated quantities {
  array[n] int  post_pred;
  array[n] int  prior_pred;
  array[n] real log_lik;
  
  for (i in 1:n) {
    // posterior predictive
    real theta = beta_rng(alpha_post[i], beta_post[i]);
    post_pred[i] = binomial_rng(n_total_rating, theta);
    
    // prior predictive
    real a_pr = 0.5 + group_trust_mean[i] + trust1[i];
    real b_pr = 0.5 + (n_total_rating - group_trust_mean[i]) + (n_total_rating - trust1[i]);
    real theta_prior = beta_rng(fmax(0.01, a_pr), fmax(0.01, b_pr));
    prior_pred[i] = binomial_rng(n_total_rating, theta_prior);
    
    // log likelihood for LOO
    log_lik[i] = beta_binomial_lpmf(trust2[i] | n_total_rating,
                                     alpha_post[i], beta_post[i]);
  }
}

