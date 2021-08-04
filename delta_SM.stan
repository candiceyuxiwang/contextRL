//
// This Stan program defines a simple RL model, with a classic delta learning rule
// and a standard softmax choice rule. Parameters are estimated hierarchically.
//

// The input data include subject, trial number, choices, and rewards
data {
  int<lower=1> nSubjects;
  int<lower=1> totalTrials;   
  int<lower=1> trialNum[totalTrials];
  int<lower=0> subject[totalTrials]; 
  int<lower=0,upper=4> choices[totalTrials];     
  real<lower=0, upper=100> rewards[totalTrials]; 
  }

// parameters accepted by the model
parameters {
  real eta_mu; // hyperparameter for the mean of eta (needed for alphas)
  real<lower=0> eta_sigma; // hyperparameter for the standard deviation of eta (needed for alphas)
  real beta_mu; // hyperparameter for the mean of the distribution of beta parameters
  real<lower=0> beta_sigma; // hyperparameter for the standard deviation of the distribution of beta parameters
  
  vector[nSubjects] eta_raw; // these are used to remove dependencies between parameters
  vector[nSubjects] beta_raw;
}

// Transformed parameters (for things that don't need priors; for ease of inserting into likelihood)
transformed parameters {
  // Q value - learned from reward
  vector[4] Q[totalTrials];
  
  // actual beta and eta, determined by mu and sigma hyperparameters
  vector[nSubjects] eta;
  vector[nSubjects] beta;

  for (s in 1:nSubjects){
    eta[s] = eta_mu + eta_sigma * eta_raw[s];
    beta[s] = beta_mu + beta_sigma * beta_raw[s];
  }
  
  for (t in 1:totalTrials){
    
    if (trialNum[t]==1){ // first trial for a given subject
      for (arm in 1:4){
        Q[t][arm] = 50; // initial expected utilities for all arms
      }
    } else{
      for (arm in 1:4){
        Q[t][arm] = Q[t-1][arm]; // inherit previous value
      }
      if (choices[t-1] != 0){ // if the previous trial was not a missed trial
       Q[t][choices[t-1]] = Q[t-1][choices[t-1]] + inv_logit(eta[subject[t]]) * (rewards[t-1] - Q[t-1][choices[t-1]]);
      }
    }
  }
}

model {
  eta_mu ~ normal(0,5); 
  eta_sigma ~ cauchy(0,1); 
  beta_mu ~ normal(0,5); 
  beta_sigma ~ cauchy(0,1); 
  
  beta_raw ~ normal(0,1);
  eta_raw ~ normal(0,1);
  
  for (t in 1:totalTrials){
    if (choices[t] != 0){ // adding this to avoid issues with missed trials
      choices[t] ~ categorical_logit(beta[subject[t]] * Q[t]); // the probability of the choices on each trial given utilities
    }
  }
}

// convert eta back to alpha
generated quantities{
  vector[nSubjects] alpha;
  real alpha_mu;
  real alpha_sigma;
  vector[totalTrials] log_lik; // log likelihood for model comparison

  for (s in 1:nSubjects){
    alpha[s] = inv_logit(eta[s]);
  }

  alpha_mu = mean(alpha);
  alpha_sigma = sd(alpha);
  
  for (t in 1:totalTrials){
    if (choices[t] != 0){ // adding this to avoid issues with missed trials
      log_lik[t] = categorical_logit_lpmf(choices[t] | beta[subject[t]] * Q[t]);
    }
  }
  
}

