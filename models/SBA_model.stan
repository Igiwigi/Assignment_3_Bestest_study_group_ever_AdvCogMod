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
  array[n] int<lower=0, upper=8> trust2;           // SecondRating (fixed in plots)
  array[n] int<lower=0, upper=7> group_trust_mean; // GroupTrustMean
  int<lower=1> n_total_rating;                     // = 7
}

//parameters {
  // SBA: no free parameters
//}

transformed parameters {
  // fixed agent values
  real w_direct = 1.0;
  real w_social = 1.0;
  real rho      = 0.5;
  real kappa    = 2.0;

  real eff_w_direct = w_direct * (1 - rho);  // = 0.5
  real eff_w_social = w_social * rho;         // = 0.5

  vector[n] alpha_post = 0.5 + kappa * (
    eff_w_direct * to_vector(trust1) +
    eff_w_social * to_vector(group_trust_mean)
  );

  vector[n] beta_post = 0.5 + kappa * (
    eff_w_direct * (n_total_rating - to_vector(trust1)) +
    eff_w_social * (n_total_rating - to_vector(group_trust_mean))
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
