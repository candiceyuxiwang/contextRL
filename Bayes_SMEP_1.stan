//
// This Stan program defines a RL model with a Bayesian learning rule
// and a standard softmax choice rule with an additional exploration bonus that 
// scales with the estimated uncertainty of the chosen arm and a perservaration
// bonus that advantages the arm chosen on the immediate last trial. 
// Parameters are estimated hierarchically.
// Adapted from Chakroun et al. 2020
//

data {
  int<lower=1> nSubjects;
  int<lower=1> nTrials;       
  int<lower=0,upper=4> choices[nSubjects, nTrials];     
  real<lower=0, upper=100> rewards[nSubjects, nTrials]; 
  }

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

parameters {
  real<lower=0> beta_mu;
  real phi_mu;
  real persev_mu;

  real<lower=0> beta_sigma;
  real<lower=0> phi_sigma;
  real<lower=0> persev_sigma;
  
  real beta_raw[nSubjects]; 
  real phi_raw[nSubjects];
  real persev_raw[nSubjects];
}

transformed parameters{
  // actual beta, phi, and persev parameters, determined by mu and sigma hyperparameters
  vector[nSubjects] beta;
  vector[nSubjects] phi;
  vector[nSubjects] persev;

  for (s in 1:nSubjects){
    beta[s] = beta_mu + beta_sigma * beta_raw[s];
    phi[s] = phi_mu + phi_sigma * phi_raw[s];
    persev[s] = persev_mu + persev_sigma * persev_raw[s];
  }
}

model {
  beta_mu ~ normal(0,5); 
  phi_mu ~ normal(0,1);
  persev_mu ~ normal(0,1);
  beta_sigma    ~ cauchy(0,1);
  phi_sigma     ~ cauchy(0,1);
  persev_sigma  ~ cauchy(0,1);
  beta_raw ~ normal(0,1);
  phi_raw ~ normal(0,1);
  persev_raw ~ normal(0,1);

  for (s in 1:nSubjects) {

      vector[4] v;   // value (mu)
      vector[4] sig; // sigma
      vector[4] eb;  // exploration bonus
      vector[4] pb;  // perseveration bonus
      real pe;       // prediction error
      real Kgain;    // Kalman gain
  
      v   = rep_vector(v1, 4);
      sig = rep_vector(sig1, 4);
  
      for (t in 1:nTrials) {        
      
      if (choices[s,t] != 0) {
        
        eb = phi[s] * sig;
        pb = rep_vector(0.0, 4);
        
        if (t>1) {
          if (choices[s,t-1] !=0) {
            pb[choices[s,t-1]] = persev[s];
          } 
        }
        
        choices[s,t] ~ categorical_logit( beta[s] * (v + eb + pb) ); // compute action probabilities
  
        pe    = rewards[s,t] - v[choices[s,t]];                       // prediction error 
        Kgain = sig[choices[s,t]]^2 / (sig[choices[s,t]]^2 + sigO^2); // Kalman gain
        
        v[choices[s,t]]   = v[choices[s,t]] + Kgain * pe;             // value/mu updating (learning)
        sig[choices[s,t]] = sqrt( (1-Kgain) * sig[choices[s,t]]^2 );  // sigma updating
        
      }
      
      v = decay * v + (1-decay) * decay_center;  
      for (j in 1:4) {
          sig[j] = sqrt( decay^2 * sig[j]^2 + sigD^2 );
      }
      }
    }
}
