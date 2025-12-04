# src/bocpd_model.py
import numpy as np
from scipy.stats import t

class BOCPD:
    def __init__(self, hazard_rate=1/100, mean_prior=0, var_prior=1, var_data=1):
        """
        Bayesian Online Changepoint Detection with Gaussian predictive distribution.
        
        Args:
            hazard_rate: Constant hazard rate (lambda).
            mean_prior: Prior mean (mu_0).
            var_prior: Prior variance (sigma_0^2).
            var_data: Data variance (sigma_noise^2).
        """
        self.hazard_rate = hazard_rate
        self.mean_prior = mean_prior
        self.var_prior = var_prior
        self.var_data = var_data
        
        # Initialize run length probabilities
        # R[t, r] is P(run_length_t = r | data_1:t)
        # We only store the current timestep's distribution to save memory if needed,
        # but for analysis we usually want the full matrix.
        self.R = np.array([[1.0]]) # R[0, 0] = 1
        
        # Sufficient statistics for Gaussian:
        # mu_n, kappa_n, alpha_n, beta_n (for Normal-Gamma/Inverse-Gamma)
        # But here we use a simpler fixed-variance Gaussian for simplicity as per Adams & MacKay 
        # (or the conjugate Normal-Normal if variance is known).
        # Let's use the standard conjugate Normal-Gamma for unknown mean and precision if possible,
        # or just Normal-Normal if we assume fixed data variance.
        # The prompt implies a "Gaussian predictive distribution".
        # Let's stick to the simpler Normal-Normal (known variance) for stability first, 
        # or Normal-Inverse-Gamma (unknown mean & variance) for robustness.
        
        # Let's use Normal-Inverse-Gamma (NIG) priors for unknown mean and variance.
        # Parameters: mu (mean), kappa (precision of mean), alpha (shape of var), beta (scale of var)
        self.mu0 = 0
        self.kappa0 = 1
        self.alpha0 = 1
        self.beta0 = 1
        
        # Store current sufficient stats for all active run lengths
        self.muT = np.array([self.mu0])
        self.kappaT = np.array([self.kappa0])
        self.alphaT = np.array([self.alpha0])
        self.betaT = np.array([self.beta0])
        
    def update(self, x):
        """
        Process one new data point x.
        """
        # 1. Evaluate Predictive Probability pi_t = P(x_t | r_{t-1}, x_{t-r:t-1})
        # Posterior predictive of NIG is a Student's t-distribution
        # t_{2*alpha} (mu, beta * (kappa+1) / (alpha * kappa))
        
        df = 2 * self.alphaT
        loc = self.muT
        scale = np.sqrt(self.betaT * (self.kappaT + 1) / (self.alphaT * self.kappaT))
        
        pred_probs = t.pdf(x, df, loc=loc, scale=scale)
        
        # 2. Calculate Growth Probabilities: P(r_t = r_{t-1} + 1 | ...)
        # growth_probs[r] \propto R[t-1, r] * pred_probs[r] * (1 - hazard)
        hazard = self.hazard_rate
        growth_probs = self.R[-1] * pred_probs * (1 - hazard)
        
        # 3. Calculate Changepoint Probability: P(r_t = 0 | ...)
        # cp_prob \propto sum(R[t-1, r] * pred_probs[r] * hazard)
        cp_prob = np.sum(self.R[-1] * pred_probs * hazard)
        
        # 4. Normalize
        evidence = np.sum(growth_probs) + cp_prob
        new_R = np.append(cp_prob, growth_probs)
        new_R /= evidence
        
        # Update R matrix
        # Pad previous R with 0 to match shape if we were storing full matrix
        # But here we just append the new row.
        # To make it a matrix, we need to handle the growing size.
        # For efficiency, let's just append to a list of arrays
        # self.R = np.vstack([self.R, new_R]) # This is slow. 
        # Better: just store the list and stack at the end if needed.
        # But self.R needs to be accessible as the "last row".
        
        # Let's re-structure: self.R_history list of arrays
        if not hasattr(self, 'R_history'):
            self.R_history = [self.R[-1]]
        self.R_history.append(new_R)
        self.R = np.array([new_R]) # Keep only current for next step calculation logic
        
        # 5. Update Sufficient Statistics
        # Update for existing runs (r > 0)
        new_muT = (self.kappaT * self.muT + x) / (self.kappaT + 1)
        new_kappaT = self.kappaT + 1
        new_alphaT = self.alphaT + 0.5
        new_betaT = self.betaT + (self.kappaT * (x - self.muT)**2) / (2 * (self.kappaT + 1))
        
        # Append prior for new run (r = 0)
        self.muT = np.append(self.mu0, new_muT)
        self.kappaT = np.append(self.kappa0, new_kappaT)
        self.alphaT = np.append(self.alpha0, new_alphaT)
        self.betaT = np.append(self.beta0, new_betaT)
        
    def get_run_length_matrix(self):
        """
        Return the full R matrix (T x T).
        Pad with zeros.
        """
        T = len(self.R_history)
        R_mat = np.zeros((T, T))
        for t, probs in enumerate(self.R_history):
            R_mat[t, :len(probs)] = probs
        return R_mat
