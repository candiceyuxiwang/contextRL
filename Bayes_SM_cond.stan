//
// This Stan program defines a RL model with a Bayesian learning rule
// and a standard softmax choice rule. 
// Parameters are estimated hierarchically for participants in each condition.
//

// The input data include subject, trial number, choices, and rewards
data {
  int<lower=1> nSubjects;
  int<lower=1> totalTrials;   
  int<lower=1> trialNum[totalTrials];
  int<lower=1> subject[totalTrials]; 
  int<lower=0, upper=4> choices[totalTrials];     
  real<lower=0, upper=100> rewards[totalTrials]; 
  int<lower=1, upper=2> condition[nSubjects]; // 1 = imperative, 2 = interrogative
  }
  
// fixed paramteres from Chakroun et al., 2020
transformed data {
  real<lower=0, upper=100> v1;
  real<lower=0> sig1;
  real<lower=0> sigO;
  real<lower=0> sigD;
  real<lower=0,upper=1> decay;
  real<lower=0, upper=100> decay_center;
  
  // random walk parameters 
  v1   = 50.0;        // prior belief mean reward value trial 1
  sig1 = 4.0;         // prior belief variance trial 1
  sigO = 4.0;         // observation variance
  sigD = 2.8;         // diffusion variance
  decay = 0.9836;     // decay parameter
  decay_center = 50;  // decay center
}

// parameters accepted by the model
parameters {
  real<lower=0> beta_mu; // hyperparameter for the mean of the distribution of beta parameters
  real beta_mu_diff;
  real<lower=0> beta_sigma; // hyperparameter for the standard deviation of the distribution of beta parameters
  
  vector[nSubjects] beta_raw;
}

// Transformed parameters (for things that don't need priors; for ease of inserting into likelihood)
transformed parameters {
  real pe; // prediction error
  vector[totalTrials] Kgain; // kalman gain

  vector[4] v[totalTrials];   // value (mu)
  vector<lower=0>[4] sig; // sigma
  
  // actual beta parameters, determined by mu and sigma hyperparameters
  vector[nSubjects] beta;
 
  for (s in 1:nSubjects){
    if(condition[s]==1){
      beta[s] = beta_mu + beta_mu_diff/2 + beta_sigma * beta_raw[s];
    }else{
      beta[s] = beta_mu - beta_mu_diff/2 + beta_sigma * beta_raw[s];
    }  }
  
  for (t in 1:totalTrials){
    if (trialNum[t]==1){ // first trial for a given subject
      v[t]   = rep_vector(v1, 4);
      sig = rep_vector(sig1, 4);

    } 
    pe    = rewards[t] - v[t][choices[t]];                       // prediction error 
    Kgain[t] = sig[choices[t]]^2 / (sig[choices[t]]^2 + sigO^2); // Kalman gain

    if(t<totalTrials){ // update prior distribution for all 4 bandits
      for (j in 1:4) {
        if(choices[t]==j){
          v[t+1][choices[t]]   = v[t][choices[t]] + Kgain[t] * pe;             // posterior value/mu updating (learning)
          sig[choices[t]] = sqrt( (1-Kgain[t]) * sig[choices[t]]^2 );  // sigma updating
          v[t+1][choices[t]] = decay * v[t+1][choices[t]] + (1-decay) * decay_center; 
          sig[choices[t]] = sqrt( decay^2 * sig[choices[t]]^2 + sigD^2 );
        }else{
          v[t+1][j] = decay * v[t][j] + (1-decay) * decay_center; 
          sig[j] = sqrt( decay^2 * sig[j]^2 + sigD^2 );
        }
      }
    }
  }
}

model {
  
  beta_mu ~ normal(0,1); 
  beta_mu_diff ~ normal(0,1);
  beta_sigma ~ normal(0,1); 
  
  beta_raw ~ normal(0,1);
  
  for (t in 1:totalTrials){
      choices[t] ~ categorical_logit(beta[subject[t]] * v[t]);
  }

}

generated quantities{
  //log_lik
  vector[totalTrials] log_lik; // log likelihood for model comparison

  for (t in 1:totalTrials){
   log_lik[t] = categorical_logit_lpmf(choices[t] | beta[subject[t]] * v[t]);
  }

}

