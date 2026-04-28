//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=1> n;
  array[n] int<lower=0, upper=8> trust1; // FirstRating (fixed in plots)
  array[n] int<lower=0, upper=8> trust2; // SecondRating (fixed in plots)
  array[n] int<lower=0, upper=7> group_trust_mean;
  int<lower=1> n_total_rating;
  
  // prior hyperparameters passed from R
  real          prior_wd_mu;
  real<lower=0> prior_wd_sigma;
  real          prior_ws_mu;
  real<lower=0> prior_ws_sigma;
  real<lower=0> prior_kappa_mu;
  real<lower=0> prior_kappa_sigma;
}

parameters {
  real<lower=0, upper=1> rho;   // social weight share
  real<lower=0>          kappa; // overall precision
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
  kappa ~ lognormal(log(2), 0.5); //centered near 2 (hopefully correct)
  target += beta_binomial_lpmf(trust2 | n_total_rating, alpha_post, beta_post);
}

generated quantities {
  // prior samples for recovery plots
  real w_direct_prior = normal_rng(prior_wd_mu, prior_wd_sigma);
  real w_social_prior = normal_rng(prior_ws_mu, prior_ws_sigma);
  real rho_prior      = beta_rng(2, 2);
  real kappa_prior    = normal_rng(prior_kappa_mu, prior_kappa_sigma);

  array[n] int  post_pred;
  array[n] int  prior_pred;
  array[n] real log_lik;

  for (i in 1:n) {
    // posterior predictive
    real theta_post  = beta_rng(alpha_post[i], beta_post[i]);
    post_pred[i]     = binomial_rng(n_total_rating, theta_post);

    // prior predictive
    real eff_wd_pr = fmax(0, w_direct_prior) * (1 - rho_prior);
    real eff_ws_pr = fmax(0, w_social_prior) * rho_prior;
    real a_pr = 0.5 + fmax(0, kappa_prior) * (
      eff_wd_pr * trust1[i] + eff_ws_pr * group_trust_mean[i]);
    real b_pr = 0.5 + fmax(0, kappa_prior) * (
      eff_wd_pr * (n_total_rating - trust1[i]) +
      eff_ws_pr * (n_total_rating - group_trust_mean[i]));
    real theta_prior = beta_rng(fmax(0.01, a_pr), fmax(0.01, b_pr));
    prior_pred[i]    = binomial_rng(n_total_rating, theta_prior);
    
    // log likelihood for LOO
    log_lik[i] = beta_binomial_lpmf(trust2[i] | n_total_rating,
                                     alpha_post[i], beta_post[i]);
  }
}
