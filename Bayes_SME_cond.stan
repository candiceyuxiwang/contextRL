//
// This Stan program defines a RL model with a Bayesian learning rule
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
  real phi_mu; // hyperparameter for the mean of phi (directed exploration bonus)
  real phi_mu_diff;
  real<lower=0> phi_sigma; // hyperparameter for the standard deviation of distribution of phi parameters
 // real persev_mu; // hyperparameter for perserveration
 // real persev_mu_diff;
 // real<lower=0> persev_sigma;
  
  vector[nSubjects] beta_raw;
  vector[nSubjects] phi_raw;
 // vector[nSubjects] persev_raw;
}

// Transformed parameters (for things that don't need priors; for ease of inserting into likelihood)
transformed parameters {
  real pe; // prediction error
  vector[totalTrials] Kgain; // kalman gain
  vector[4] eb[totalTrials]; // exploration bonus   
  
//  vector[4] pb;  // perseveration bonus
  vector[4] v[totalTrials];   // value (mu)
  vector<lower=0>[4] sig; // sigma
  
  // actual beta, phi, and persev parameters, determined by mu and sigma hyperparameters
  vector[nSubjects] beta;
  vector[nSubjects] phi;
 // vector[nSubjects] persev;
 
  for (s in 1:nSubjects){
    if(condition[s]==1){
      beta[s] = beta_mu + beta_mu_diff/2 + beta_sigma * beta_raw[s];
      phi[s] = phi_mu + phi_mu_diff/2 + phi_sigma * phi_raw[s];
     // persev[s] = persev_mu + persev_mu_diff/2 + persev_sigma * persev_raw[s];
    }else{
      beta[s] = beta_mu - beta_mu_diff/2 + beta_sigma * beta_raw[s];
      phi[s] = phi_mu - phi_mu_diff/2 + phi_sigma * phi_raw[s];
     // persev[s] = persev_mu - persev_mu_diff/2 + persev_sigma * persev_raw[s];
    }  }
  
  for (t in 1:totalTrials){

    if (trialNum[t]==1){ // first trial for a given subject
      v[t]   = rep_vector(v1, 4);
      sig = rep_vector(sig1, 4);
      
    } 
    pe    = rewards[t] - v[t][choices[t]];                       // prediction error 
    Kgain[t] = sig[choices[t]]^2 / (sig[choices[t]]^2 + sigO^2); // Kalman gain

    eb[t] = phi[subject[t]] * sig;

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
  // using priors from the estimated posteriors from Chakroun et al. 2020
  // beta_mu ~ normal(0.2,0.1); 
  // beta_sigma ~ normal(0.1,0.1); 
  // phi_mu ~ normal(1,0.1);
  // phi_sigma ~ normal(0.7,0.1);
  // persev_mu ~ normal(5,1);
  // persev_sigma ~ normal(0.1,0.2);
  // 
  // beta_raw ~ normal(0,0.1);
  // phi_raw ~ normal(0,0.1);
  // persev_raw ~ normal(0,1);
  
  beta_mu ~ normal(0,1); 
  beta_mu_diff ~ normal(0,1);
  beta_sigma ~ normal(0,1); 
  phi_mu ~ normal(0,1);
  phi_mu_diff ~ normal(0,1);
  phi_sigma ~ normal(0,1);
 // persev_mu ~ normal(0,1);
 // persev_mu_diff ~ normal(0,1);
 // persev_sigma ~ normal(0,1);
  
  beta_raw ~ normal(0,1);
  phi_raw ~ normal(0,1);
 // persev_raw ~ normal(0,1);
  
  for (t in 1:totalTrials){
  //  vector[4] pb;  // perseveration bonus
  //  if (choices[t] != 0){ // adding this to avoid issues with missed trials
  //    pb = rep_vector(0.0, 4);
  //    if (trialNum[t]>1) {
          //if (choices[t-1] !=0) {
  //          pb[choices[t-1]] = persev[subject[t]];
          //} 
   //   }
  //    choices[t] ~ categorical_logit(beta[subject[t]] * (v[t] + eb[t] + pb)); // the probability of the choices on each trial given utilities and exploration bonus
      choices[t] ~ categorical_logit(beta[subject[t]] * v[t] + eb[t]);
 //   }
  }

}

generated quantities{
  //log_lik
  vector[totalTrials] log_lik; // log likelihood for model comparison
 // vector[4] pb;


  for (t in 1:totalTrials){
  //    pb = rep_vector(0.0, 4);
  //    if (trialNum[t]>1) {
  //       pb[choices[t-1]] = persev[subject[t]];
  //    }
   //   log_lik[t] = categorical_logit_lpmf(choices[t] | beta[subject[t]] * (v[t] + eb[t] + pb));
   log_lik[t] = categorical_logit_lpmf(choices[t] | beta[subject[t]] * v[t] + eb[t]);
  }

}

