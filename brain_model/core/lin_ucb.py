import numpy as np

class LinUCBDisjoint:
    """
    LinUCB Disjoint Contextual Bandit algorithm.
    
    Each arm maintains its own set of parameters (A_a, b_a).
    """
    def __init__(self, n_arms, n_features, alpha=1.0):
        """
        Initialize the LinUCB algorithm.
        
        Args:
            n_arms (int): Number of available arms (actions).
            n_features (int): Number of context features.
            alpha (float): Exploration parameter (higher alpha = more exploration).
        """
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        
        # A_a = D_a^T D_a + I_d (Identity matrix for each arm)
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        # b_a = D_a^T c_a (Reward vector for each arm)
        self.b = [np.zeros((n_features, 1)) for _ in range(n_arms)]
        
        # Cache for A_inv
        self.A_inv = [np.identity(n_features) for _ in range(n_arms)]

    def predict(self, context):
        """
        Predict the best arm for a given context.
        
        Args:
            context (array-like): Context features for the current learner (size: n_features).
            
        Returns:
            int: Index of the chosen arm.
        """
        x = np.array(context).reshape(-1, 1)
        p = np.zeros(self.n_arms)
        
        for a in range(self.n_arms):
            # Calculate theta = A_inv * b
            theta = self.A_inv[a] @ self.b[a]
            
            # Calculate predicted reward + UCB
            # p_a = theta^T * x + alpha * sqrt(x^T * A_inv * x)
            ucb = self.alpha * np.sqrt(x.T @ self.A_inv[a] @ x)
            p[a] = (theta.T @ x + ucb).item()
            
        return np.argmax(p)

    def update(self, arm_index, context, reward):
        """
        Update the model with the observed reward for a chosen arm.
        
        Args:
            arm_index (int): Index of the chosen arm.
            context (array-like): Context features used for prediction.
            reward (float): Observed reward (e.g., 0 to 1).
        """
        x = np.array(context).reshape(-1, 1)
        
        # Update A_a and b_a
        self.A[arm_index] += x @ x.T
        self.b[arm_index] += reward * x
        
        # Update inverse of A_a using Woodbury Identity or simply np.linalg.inv
        # Since n_features is usually small, np.linalg.inv is fine.
        # However, for efficiency we update the cached inverse.
        self.A_inv[arm_index] = np.linalg.inv(self.A[arm_index])
