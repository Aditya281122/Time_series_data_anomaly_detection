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
        self.R = np.array([[1.0]]) # R[0, 0] = 1
        self.R_history = [self.R[-1]]
        
        # Normal-Inverse-Gamma (NIG) priors
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
        # 1. Evaluate Predictive Probability
        df = 2 * self.alphaT
        loc = self.muT
        scale = np.sqrt(self.betaT * (self.kappaT + 1) / (self.alphaT * self.kappaT))
        
        pred_probs = t.pdf(x, df, loc=loc, scale=scale)
        
        # 2. Calculate Growth Probabilities
        hazard = self.hazard_rate
        growth_probs = self.R[-1] * pred_probs * (1 - hazard)
        
        # 3. Calculate Changepoint Probability
        cp_prob = np.sum(self.R[-1] * pred_probs * hazard)
        
        # 4. Normalize
        evidence = np.sum(growth_probs) + cp_prob
        new_R = np.append(cp_prob, growth_probs)
        new_R /= (evidence + 1e-9)
        
        # Update R history
        self.R_history.append(new_R)
        self.R = np.array([new_R])
        
        # 5. Update Sufficient Statistics
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
        """
        T = len(self.R_history)
        R_mat = np.zeros((T, T))
        for t, probs in enumerate(self.R_history):
            R_mat[t, :len(probs)] = probs
        return R_mat

def detect_changepoints(values: np.ndarray, hazard_rate: float = 1/100) -> np.ndarray:
    """
    Run BOCPD on a series and return changepoint probabilities (r=0).
    """
    # Normalize
    mean_val = np.mean(values)
    std_val = np.std(values)
    values_norm = (values - mean_val) / (std_val + 1e-9)
    
    bocpd = BOCPD(hazard_rate=hazard_rate)
    for x in values_norm:
        bocpd.update(x)
        
    R_mat = bocpd.get_run_length_matrix()
    # CP prob is column 0 (run length = 0)
    # Align with data: R_mat includes initial prior, so we skip row 0?
    # R_history[0] is initial state. R_history[1] is after x[0].
    # So cp_probs[t] corresponds to x[t-1]?
    # Let's align carefully.
    # R_history has length T+1.
    # cp_probs = R_mat[1:, 0] has length T.
    cp_probs = R_mat[1:, 0]
    return cp_probs
