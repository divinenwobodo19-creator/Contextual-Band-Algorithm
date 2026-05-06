import numpy as np
from typing import Dict, List, Optional

class LinTSDisjoint:
    """
    LinTS (Linear Thompson Sampling) Disjoint algorithm.
    Reference: Agrawal & Goyal, 2013.
    
    Explores by sampling from the posterior distribution of weights (theta).
    """
    def __init__(self, n_features: int, v: float = 0.1, l2_lambda: float = 1.0):
        """
        Initialize the LinTS algorithm.
        
        Args:
            n_features (int): Context dimension (d).
            v (float): Exploration parameter (nu).
            l2_lambda (float): L2 regularization (lambda).
        """
        self.n_features = n_features
        self.v = v
        self.l2_lambda = l2_lambda
        self.arms: Dict[str, Dict] = {}

    def _init_arm(self, arm_id: str):
        if arm_id not in self.arms:
            self.arms[arm_id] = {
                'B': np.identity(self.n_features) * self.l2_lambda, # Precision matrix
                'B_inv': np.identity(self.n_features) * (1.0 / self.l2_lambda), # Covariance matrix
                'f': np.zeros((self.n_features, 1)), # Reward accumulator
                'mu_hat': np.zeros((self.n_features, 1)) # Mean estimate
            }

    def select(self, arm_ids: List[str], context: np.ndarray) -> str:
        """Select best arm by sampling from posterior distributions."""
        x = context.reshape(-1, 1)
        best_sample = -float('inf')
        best_arm = None
        
        for arm_id in arm_ids:
            self._init_arm(arm_id)
            arm = self.arms[arm_id]
            
            # 1. Sample theta from N(mu_hat, v^2 * B_inv)
            # Efficient sampling: mu_hat + v * L * epsilon where epsilon ~ N(0, I) and LL^T = B_inv
            # Since B_inv is small (d x d), we can use multivariate_normal
            # or Cholesky decomposition.
            
            try:
                # covariance = (self.v ** 2) * arm['B_inv']
                # Sample a weight vector theta_a for this arm
                theta_a = np.random.multivariate_normal(
                    arm['mu_hat'].flatten(), 
                    (self.v ** 2) * arm['B_inv']
                ).reshape(-1, 1)
            except (ValueError, np.linalg.LinAlgError):
                # Fallback to mean if sampling fails
                theta_a = arm['mu_hat']
            
            # 2. Prediction = theta_a.T * x
            p = (theta_a.T @ x).item()
            
            if p > best_sample:
                best_sample = p
                best_arm = arm_id
                
        return best_arm

    def update(self, arm_id: str, context: np.ndarray, reward: float):
        """Update posterior parameters."""
        self._init_arm(arm_id)
        arm = self.arms[arm_id]
        x = context.reshape(-1, 1)
        
        # 1. Update B (precision)
        arm['B'] += x @ x.T
        
        # 2. Update f (accumulator)
        arm['f'] += reward * x
        
        # 3. Update B_inv (covariance) using Sherman-Morrison
        inv_x = arm['B_inv'] @ x
        denom = 1.0 + (x.T @ inv_x).item()
        arm['B_inv'] -= (inv_x @ inv_x.T) / denom
        
        # 4. Update mu_hat = B_inv * f
        arm['mu_hat'] = arm['B_inv'] @ arm['f']
