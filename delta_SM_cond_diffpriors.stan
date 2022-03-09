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
  int<lower=1, upper=2> condition[nSubjects]; // 1 = imperative, 2 = interrogative
  int<lower=1> nSubCond1; // number of subjects in condition 1, imperative
  }

// parameters accepted by the model
parameters {
  real eta_mu; // hyperparameter for the mean of eta (needed for alphas)
  real eta_mu_diff;
  real<lower=0> eta_sigma; // hyperparameter for the standard deviation of eta (needed for alphas)
  real beta_mu; // hyperparameter for the mean of the distribution of beta parameters
  real beta_mu_diff;
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
    if(condition[s]==1){
      eta[s] = eta_mu + eta_mu_diff/2 + eta_sigma * eta_raw[s];
      beta[s] = beta_mu + beta_mu_diff/2 + beta_sigma * beta_raw[s];
    }else{
      eta[s] = eta_mu - eta_mu_diff/2 + eta_sigma * eta_raw[s];
      beta[s] = beta_mu - beta_mu_diff/2 + beta_sigma * beta_raw[s];
    }
    
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
  eta_mu_diff ~ normal(0,1);
  eta_sigma ~ normal(0,1); 
  beta_mu ~ normal(0,1); 
  beta_mu_diff ~ normal(0,1);
  beta_sigma ~ normal(0,1); 
  
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
  vector [nSubCond1]alpha_1;
  vector [nSubjects - nSubCond1]alpha_2;
  int alpha_1_counter;
  int alpha_2_counter;
  real alpha_mu[2];
  real alpha_sigma;
  vector[totalTrials] log_lik; // log likelihood for model comparison
  real alpha_mu_diff;

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
  alpha_mu_diff = alpha_mu[1] - alpha_mu[2];
  
  alpha_sigma = sd(alpha);
   
  for (t in 1:totalTrials){
    if (choices[t] != 0){ // adding this to avoid issues with missed trials
      log_lik[t] = categorical_logit_lpmf(choices[t] | beta[subject[t]] * Q[t]);
    }
  }
  
}

