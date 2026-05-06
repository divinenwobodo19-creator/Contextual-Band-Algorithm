import numpy as np
from typing import Dict, Optional

class LinUCBDisjoint:
    """
    LinUCB Disjoint algorithm implementation with support for Discounting (forgetfulness).
    Reference: Li et al., 2010.
    """
    def __init__(self, n_features: int, alpha: float = 1.0, l2_lambda: float = 1.0, gamma: float = 1.0):
        """
        Initialize the LinUCB algorithm.
        
        Args:
            n_features (int): Context vector dimension (d).
            alpha (float): Exploration-exploitation tradeoff.
            l2_lambda (float): L2 Regularization parameter.
            gamma (float): Discount factor for non-stationary environments (0 < gamma <= 1.0).
                           1.0 = standard LinUCB (no forgetfulness).
        """
        self.n_features = n_features
        self.alpha = alpha
        self.l2_lambda = l2_lambda
        self.gamma = gamma
        
        # Maps arm_id (str) to its (A, b) matrices
        self.arms: Dict[str, Dict] = {}

    def _init_arm(self, arm_id: str):
        """Initialize matrices for a new arm with L2 regularization."""
        if arm_id not in self.arms:
            self.arms[arm_id] = {
                'A': np.identity(self.n_features) * self.l2_lambda,
                'A_inv': np.identity(self.n_features) * (1.0 / self.l2_lambda), # Maintain inverse for efficiency
                'b': np.zeros((self.n_features, 1))
            }

    def select(self, arm_ids: list[str], context: np.ndarray) -> str:
        """
        Select the best arm based on highest UCB.
        
        Args:
            arm_ids (list of str): IDs of available arms.
            context (np.ndarray): Context vector (d,).
            
        Returns:
            str: ID of the chosen arm.
        """
        x = context.reshape(-1, 1)
        best_ucb = -float('inf')
        best_arm = None
        
        for arm_id in arm_ids:
            self._init_arm(arm_id)
            A_inv = self.arms[arm_id]['A_inv']
            b = self.arms[arm_id]['b']
            
            # Compute theta = A_inv @ b
            theta = A_inv @ b
            
            # Compute UCB = theta.T @ x + alpha * sqrt(x.T @ A_inv @ x)
            # O(d^2) operation. Ensure non-negative inside sqrt for numerical stability.
            variance = (x.T @ A_inv @ x).item()
            p = (theta.T @ x).item() + self.alpha * np.sqrt(max(0, variance))
            
            if p > best_ucb:
                best_ucb = p
                best_arm = arm_id
                
        if best_arm is None:
            raise ValueError("No arms provided for selection.")
            
        return best_arm

    def update(self, arm_id: str, context: np.ndarray, reward: float):
        """
        Update the model matrices for a chosen arm using Sherman-Morrison formula.
        Supports Discounted LinUCB for non-stationary environments.
        """
        self._init_arm(arm_id)
        x = context.reshape(-1, 1)
        
        arm = self.arms[arm_id]
        
        if self.gamma < 1.0:
            # Discounted LinUCB Update:
            # A_new = gamma * A + x * x.T + (1-gamma) * lambda * I
            # b_new = gamma * b + reward * x
            
            # 1. Update A and b with discount
            arm['A'] = self.gamma * arm['A'] + x @ x.T + (1 - self.gamma) * self.l2_lambda * np.identity(self.n_features)
            arm['b'] = self.gamma * arm['b'] + reward * x
            
            # 2. Re-compute A_inv (Discounting breaks rank-1 update efficiency for A_inv)
            # For O(d^2) updates with discounting, we'd need more complex math.
            # Here we use standard inv since d is small.
            arm['A_inv'] = np.linalg.inv(arm['A'])
        else:
            # Standard LinUCB (Gamma = 1.0) with Sherman-Morrison (O(d^2))
            A_inv = arm['A_inv']
            inv_x = A_inv @ x
            denominator = 1.0 + (x.T @ inv_x).item()
            arm['A_inv'] -= (inv_x @ inv_x.T) / denominator
            arm['A'] += x @ x.T
            arm['b'] += reward * x
