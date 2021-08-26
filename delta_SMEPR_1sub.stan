//
// This Stan program defines a simple RL model, with a classic delta learning rule
// and a standard softmax choice rule with an additional exploration bonus that 
// scales with the estimated uncertainty of the chosen arm, a perservaration
// bonus that advantages the arm chosen on the immediate last trial, and a random
// exploration bonus that is discounted by the summed uncertainty across all arms. 
// Parameters are estimated for 1 subject only.
//

// The input data include subject, trial number, choices, and rewards
data {
//  int<lower=1> nSubjects;
  int<lower=1> totalTrials;   
  int<lower=1> trialNum[totalTrials];
//  int<lower=0> subject[totalTrials]; 
  int<lower=0,upper=4> choices[totalTrials];     
  real<lower=0, upper=100> rewards[totalTrials]; 
  vector[4] eb[totalTrials]; // exploration bonus = the # of trials it's been since this arm was last chosen
  }

// parameters accepted by the model
parameters {
//  real eta_mu; // hyperparameter for the mean of eta (needed for alphas)
//  real<lower=0> eta_sigma; // hyperparameter for the standard deviation of eta (needed for alphas)
    real alpha;
    
//  real beta_mu; // hyperparameter for the mean of the distribution of beta parameters
//  real<lower=0> beta_sigma; // hyperparameter for the standard deviation of the distribution of beta parameters
    real<lower=0> beta;
    
//  real phi_mu; // hyperparameter for the mean of phi (directed exploration bonus)
//  real<lower=0> phi_sigma; // hyperparameter for the standard deviation of distribution of phi parameters
    real phi;
    
//  real persev_mu; // hyperparameter for perserveration
//  real<lower=0> persev_sigma;
    real persev;
    
//  real gamma_mu; // random exploration bonus
//  real<lower=0> gamma_sigma;
    real gamma;
  
//  vector[nSubjects] eta_raw; // these are used to remove dependencies between parameters
//  vector[nSubjects] beta_raw;
//  vector[nSubjects] phi_raw;
//  vector[nSubjects] persev_raw;
//  vector[nSubjects] gamma_raw;
}

// Transformed parameters (for things that don't need priors; for ease of inserting into likelihood)
transformed parameters {
  // Q value - learned from reward
  vector[4] Q[totalTrials];
  
  // actual beta and eta, determined by mu and sigma hyperparameters
//  vector[nSubjects] eta;
//  vector[nSubjects] beta;
//  vector[nSubjects] phi;
//  vector[nSubjects] persev;
//  vector[nSubjects] gamma;

//  for (s in 1:nSubjects){
//    eta[s] = eta_mu + eta_sigma * eta_raw[s];
//    beta[s] = beta_mu + beta_sigma * beta_raw[s];
//    phi[s] = phi_mu + phi_sigma * phi_raw[s];
//    persev[s] = persev_mu + persev_sigma * persev_raw[s];
//    gamma[s] = gamma_mu + gamma_sigma * gamma_raw[s];
//  }
  
  for (t in 1:totalTrials){
    
    if (trialNum[t]==1){ // first trial for a given subject
      for (arm in 1:4){
        Q[t][arm] = 50; // initial expected utilities for all arms
      }
    } else{
      for (arm in 1:4){
        Q[t][arm] = Q[t-1][arm]; // inherit previous value
      }
      if (choices[t-1] != 0){ // if the previous trial was not a missed trial, update estimated reward and perserv bonus
//       Q[t][choices[t-1]] = Q[t-1][choices[t-1]] + inv_logit(eta[subject[t]]) * (rewards[t-1] - Q[t-1][choices[t-1]]) + persev[subject[t]];
         Q[t][choices[t-1]] = Q[t-1][choices[t-1]] + alpha * (rewards[t-1] - Q[t-1][choices[t-1]]) + persev;
      }
    }
  }
}

model {
//  eta_mu ~ normal(0,5); 
//  eta_sigma ~ cauchy(0,1); 
  alpha ~ beta(1,1);
//  beta_mu ~ normal(0,5); 
//  beta_sigma ~ cauchy(0,1);
  beta ~ normal(0,5);
//  phi_mu ~ normal(0,1);
//  phi_sigma ~ cauchy(0,1);
  phi ~ normal(0,1);
//  persev_mu ~ normal(0,1);
//  persev_sigma ~ cauchy(0,1);
  persev ~ normal(0,1);
//  gamma_mu ~ normal(0,1);
//  gamma_sigma ~ cauchy(0,1);
  gamma ~ normal(0,1);
  
//  beta_raw ~ normal(0,1);
//  eta_raw ~ normal(0,1);
//  phi_raw ~ normal(0,1);
//  persev_raw ~ normal(0,1);
//  gamma_raw ~ normal(0,1);
  
  for (t in 1:totalTrials){
    if (choices[t] != 0){ // adding this to avoid issues with missed trials
//      choices[t] ~ categorical_logit(beta[subject[t]] * (Q[t] + phi[subject[t]] * eb[t] + gamma[subject[t]]*(Q[t]/sum(eb[t]))));// the probability of the choices on each trial given utilities and exploration bonus
        choices[t] ~ categorical_logit(beta * (Q[t] + phi * eb[t] + gamma * (Q[t]/sum(eb[t]))));
    }
  }
}

// convert eta back to alpha
generated quantities{
//  vector[nSubjects] alpha;
//  real alpha_mu;
//  real alpha_sigma;
//  vector[totalTrials] log_lik; // log likelihood for model comparison

//  for (s in 1:nSubjects){
//    alpha[s] = inv_logit(eta[s]);
//  }

//  alpha_mu = mean(alpha);
//  alpha_sigma = sd(alpha);
  
//  for (t in 1:totalTrials){
//    if (choices[t] != 0){ // adding this to avoid issues with missed trials
//      log_lik[t] = categorical_logit_lpmf(choices[t] | beta[subject[t]] * (Q[t] + phi[subject[t]] * eb[t] + gamma[subject[t]]*(Q[t]/sum(eb[t]))));
//    }
//  }
  
}

