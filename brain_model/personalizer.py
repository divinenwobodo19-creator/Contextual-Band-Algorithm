import numpy as np
from brain_model.core.lin_ucb import LinUCBDisjoint

class Personalizer:
    """
    High-level API for personalizing learning content.
    
    This class wraps the LinUCB algorithm to provide a user-friendly interface
    for LMS and grade tracker integrations.
    """
    def __init__(self, n_arms, n_features, arm_names=None, alpha=1.0):
        """
        Initialize the personalizer.
        
        Args:
            n_arms (int): Number of content types to choose from.
            n_features (int): Number of learner/context features.
            arm_names (list of str, optional): Human-readable names for each arm.
            alpha (float): Exploration-exploitation trade-off.
        """
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        
        # Human-readable mapping
        if arm_names:
            if len(arm_names) != n_arms:
                raise ValueError(f"arm_names length ({len(arm_names)}) must match n_arms ({n_arms}).")
            self.arm_names = arm_names
        else:
            self.arm_names = [f"arm_{i}" for i in range(n_arms)]
            
        self.name_to_idx = {name: i for i, name in enumerate(self.arm_names)}
        self.idx_to_name = {i: name for i, name in enumerate(self.arm_names)}
        
        # Core model
        self.model = LinUCBDisjoint(n_arms, n_features, alpha=alpha)

    def predict(self, context):
        """
        Get the recommended content for a given learner context.
        
        Args:
            context (array-like): Normalized features (e.g., [score, time, engagement]).
            
        Returns:
            str: Recommended content name.
        """
        if len(context) != self.n_features:
            raise ValueError(f"Context feature size must be {self.n_features}, got {len(context)}.")
            
        arm_idx = self.model.predict(context)
        return self.idx_to_name[arm_idx]

    def update(self, arm_name, context, reward):
        """
        Update the model based on learner's performance.
        
        Args:
            arm_name (str): Name of the content that was recommended.
            context (array-like): Context features used during prediction.
            reward (float): Observed reward (e.g., score between 0 and 1).
        """
        if arm_name not in self.name_to_idx:
            raise ValueError(f"Unknown arm name: {arm_name}")
            
        arm_idx = self.name_to_idx[arm_name]
        self.model.update(arm_idx, context, reward)

    def save(self, filepath):
        """
        Save the model state to a file.
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'A': self.model.A,
                'b': self.model.b,
                'arm_names': self.arm_names,
                'n_arms': self.n_arms,
                'n_features': self.n_features,
                'alpha': self.alpha
            }, f)

    @classmethod
    def load(cls, filepath):
        """
        Load the model state from a file.
        """
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        instance = cls(n_arms=data['n_arms'], 
                       n_features=data['n_features'], 
                       arm_names=data['arm_names'], 
                       alpha=data['alpha'])
        instance.model.A = data['A']
        instance.model.b = data['b']
        # Recalculate inverses
        instance.model.A_inv = [np.linalg.inv(A) for A in instance.model.A]
        return instance
