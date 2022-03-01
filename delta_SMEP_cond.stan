//
// This Stan program defines a simple RL model, with a classic delta learning rule
// and a standard softmax choice rule with an additional exploration bonus that 
// scales with the estimated uncertainty of the chosen arm and a perservaration
// bonus that advantages the arm chosen on the immediate last trial. 
// Parameters are estimated hierarchically for participants in each condition.
//

// The input data include subject, trial number, choices, and rewards
data {
  int<lower=1> nSubjects;
  int<lower=1> totalTrials;   
  int<lower=1> trialNum[totalTrials];
  int<lower=1> subject[totalTrials]; 
  int<lower=0,upper=4> choices[totalTrials];     
  real<lower=0, upper=100> rewards[totalTrials]; 
  vector[4] eb[totalTrials]; // exploration bonus = the # of trials it's been since this arm was last chosen
  int<lower=1, upper=2> condition[nSubjects]; // 1 = imperative, 2 = interrogative
  int<lower=1> nSubCond1; // number of subjects in condition 1, imperative
  }

// parameters accepted by the model
parameters {
  real eta_mu[2]; // hyperparameter for the mean of eta (needed for alphas)
  real<lower=0> eta_sigma; // hyperparameter for the standard deviation of eta (needed for alphas)
  real<lower=0> beta_mu[2]; // hyperparameter for the mean of the distribution of beta parameters
  real<lower=0> beta_sigma; // hyperparameter for the standard deviation of the distribution of beta parameters
  real phi_mu[2]; // hyperparameter for the mean of phi (directed exploration bonus)
  real<lower=0> phi_sigma; // hyperparameter for the standard deviation of distribution of phi parameters
  real persev_mu[2]; // hyperparameter for perserveration
  real<lower=0> persev_sigma;
  
  vector[nSubjects] eta_raw; // these are used to remove dependencies between parameters
  vector[nSubjects] beta_raw;
  vector[nSubjects] phi_raw;
  vector[nSubjects] persev_raw;
}

// Transformed parameters (for things that don't need priors; for ease of inserting into likelihood)
transformed parameters {
  // Q value - learned from reward
  vector[4] Q[totalTrials];
  // perserveration bonus
  vector[4] pb[totalTrials];
  
  // actual beta and eta, determined by mu and sigma hyperparameters
  vector[nSubjects] eta;
  vector[nSubjects] beta;
  vector[nSubjects] phi;
  vector[nSubjects] persev;

  for (s in 1:nSubjects){
    eta[s] = eta_mu[condition[s]] + eta_sigma * eta_raw[s];
    beta[s] = beta_mu[condition[s]] + beta_sigma * beta_raw[s];
    phi[s] = phi_mu[condition[s]] + phi_sigma * phi_raw[s];
    persev[s] = persev_mu[condition[s]] + persev_sigma * persev_raw[s];
  }
  
  for (t in 1:totalTrials){
    if (trialNum[t]==1){ // first trial for a given subject
      for (arm in 1:4){
        Q[t][arm] = 50; // initial expected utilities for all arms
        pb[t][arm] = 0;
      }
    } else{
      for (arm in 1:4){
        Q[t][arm] = Q[t-1][arm]; // inherit previous value
        pb[t][arm] = 0;
      }
      //if (choices[t-1] != 0){ // if the previous trial was not a missed trial, update estimated reward and perserv bonus
       Q[t][choices[t-1]] = Q[t-1][choices[t-1]] + inv_logit(eta[subject[t]]) * (rewards[t-1] - Q[t-1][choices[t-1]]);
       pb[t][choices[t-1]] = persev[subject[t]];
      //}
    }
  }
}

model {
  eta_mu ~ normal(0,5); 
  eta_sigma ~ cauchy(0,1); 
  beta_mu ~ normal(0,5); 
  beta_sigma ~ cauchy(0,1); 
  phi_mu ~ normal(0,1);
  phi_sigma ~ cauchy(0,1);
  persev_mu ~ normal(0,1);
  persev_sigma ~ cauchy(0,1);
  
  beta_raw ~ normal(0,1);
  eta_raw ~ normal(0,1);
  phi_raw ~ normal(0,1);
  persev_raw ~ normal(0,1);
  
  for (t in 1:totalTrials){
   // if (choices[t] != 0){ // adding this to avoid issues with missed trials
      choices[t] ~ categorical_logit(beta[subject[t]] * (Q[t] + phi[subject[t]] * eb[t] + pb[t])); // the probability of the choices on each trial given utilities and exploration bonus
   // }
  }
}

// convert eta back to alpha
generated quantities{
  vector[nSubjects] alpha;
  vector [nSubCond1]alpha_1;
  vector [nSubjects - nSubCond1]alpha_2;
  int alpha_1_counter;
  int alpha_2_counter;
  real alpha_mu[2];
  real alpha_sigma[2];
  vector[totalTrials] log_lik; // log likelihood for model comparison
  real beta_diff;  // difference between beta_mu[1] (imperative) and beta_mu[2] (interrogative)
  real phi_diff; // directed exploration
  real persev_diff;
  real alpha_diff;

  alpha_1_counter = 0;
  alpha_2_counter = 0;

  for (s in 1:nSubjects){
    alpha[s] = inv_logit(eta[s]);
    if(condition[s]==1){
      alpha_1_counter = alpha_1_counter + 1;
      alpha_1[alpha_1_counter] = alpha[s];
    }else{
      alpha_2_counter = alpha_2_counter + 1;
      alpha_2[alpha_2_counter] = alpha[s];
    }
  }

  alpha_mu[1] = mean(alpha_1);
  alpha_mu[2] = mean(alpha_2);
  alpha_sigma[1] = sd(alpha_1);
  alpha_sigma[2] = sd(alpha_2);
  
  for (t in 1:totalTrials){
    if (choices[t] != 0){ // adding this to avoid issues with missed trials
      log_lik[t] = categorical_logit_lpmf(choices[t] | beta[subject[t]] * (Q[t] + phi[subject[t]] * eb[t]));
    }
  }
  beta_diff = beta_mu[1] - beta_mu[2];
  phi_diff = phi_mu[1] - phi_mu[2];
  persev_diff = persev_mu[1] - persev_mu[2];
  alpha_diff = alpha_mu[1] - alpha_mu[2];
}

