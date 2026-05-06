import numpy as np
from typing import Dict, List, Optional, Tuple

class LinUCBHybrid:
    """
    LinUCB Hybrid algorithm with Cluster-Specific support.
    Reference: Li et al., 2010 (Algorithm 2) extended for COBART-style clustering.
    
    Expected reward: r = z^T * beta + x^T * theta_a + c^T * gamma_k
    where z: shared features, x: arm-specific features, c: cluster-specific features.
    """
    def __init__(self, k_shared: int, d_arm: int, n_clusters: int = 5, alpha: float = 1.0, l2_lambda: float = 1.0):
        """
        Initialize the LinUCB Hybrid algorithm with clustering support.
        
        Args:
            k_shared (int): Dimension of shared features (z).
            d_arm (int): Dimension of arm-specific features (x).
            n_clusters (int): Number of student clusters (k).
            alpha (float): Exploration parameter.
            l2_lambda (float): L2 regularization parameter.
        """
        self.k = k_shared
        self.d = d_arm
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.l2_lambda = l2_lambda
        
        # 1. Shared parameters (A0, b0) - Global knowledge across all arms and clusters
        self.A0 = np.identity(self.k) * self.l2_lambda
        self.b0 = np.zeros((self.k, 1))
        self.A0_inv = np.identity(self.k) * (1.0 / self.l2_lambda)
        
        # 2. Cluster-specific parameters (Ak, bk) - Knowledge shared within student cohorts
        self.Ak = [np.identity(self.k) * self.l2_lambda for _ in range(n_clusters)]
        self.bk = [np.zeros((self.k, 1)) for _ in range(n_clusters)]
        self.Ak_inv = [np.identity(self.k) * (1.0 / self.l2_lambda) for _ in range(n_clusters)]
        
        # 3. Arm-specific parameters (Aa, Ba, ba)
        self.arms: Dict[str, Dict] = {}

    def _init_arm(self, arm_id: str):
        if arm_id not in self.arms:
            self.arms[arm_id] = {
                'A': np.identity(self.d) * self.l2_lambda,
                'A_inv': np.identity(self.d) * (1.0 / self.l2_lambda),
                'B': np.zeros((self.d, self.k)),
                'b': np.zeros((self.d, 1)),
                'last_cluster_id': None
            }

    def select(self, arm_ids: List[str], z_shared: np.ndarray, x_arms: Dict[str, np.ndarray], cluster_id: int = 0) -> str:
        """
        Select the best arm based on Hybrid LinUCB using vectorized operations across arms.
        Incorporates Cluster-Specific parameters (Ak_inv) for the student's cohort.
        """
        # Ensure inverses are up to date before selection
        self._ensure_inverses(cluster_id)
        
        z = z_shared.reshape(-1, 1)
        # For prediction, we use a combination of global and cluster-specific beta_hat
        # beta_hat_global = self.A0_inv @ self.b0
        # beta_hat_cluster = self.Ak_inv[cluster_id] @ self.bk[cluster_id]
        # Combined beta_hat = (beta_hat_global + beta_hat_cluster) / 2.0
        
        beta_hat_global = self.A0_inv @ self.b0
        beta_hat_cluster = self.Ak_inv[cluster_id] @ self.bk[cluster_id]
        beta_hat = (beta_hat_global + beta_hat_cluster) / 2.0
        
        # Pre-initialize any new arms
        for arm_id in arm_ids:
            self._init_arm(arm_id)
            
        # Collect arm parameters for vectorization
        n_arms = len(arm_ids)
        A_invs = np.array([self.arms[aid]['A_inv'] for aid in arm_ids]) # (N, d, d)
        Bs = np.array([self.arms[aid]['B'] for aid in arm_ids])          # (N, d, k)
        bs = np.array([self.arms[aid]['b'] for aid in arm_ids])          # (N, d, 1)
        xs = np.array([x_arms[aid].reshape(-1, 1) for aid in arm_ids])  # (N, d, 1)
        
        # 1. Compute theta_hat for all arms: theta_hat = A_inv @ (b - B @ beta_hat)
        B_beta = Bs @ beta_hat
        theta_hats = A_invs @ (bs - B_beta) # (N, d, 1)
        
        # 2. Compute Variance s_t,a for all arms
        # s = z.T @ A0_inv @ z - 2 * z.T @ A0_inv @ B.T @ A_inv @ x + x.T @ A_inv @ x + x.T @ A_inv @ B @ A0_inv @ B.T @ A_inv @ x
        # Use A0_inv or Ak_inv? For variance, we use the global A0_inv for conservative exploration.
        z_A0_inv = z.T @ self.A0_inv # (1, k)
        
        term1 = (z_A0_inv @ z).item()
        B_T_A_inv_x = np.transpose(Bs, (0, 2, 1)) @ A_invs @ xs
        term2 = -2 * (z_A0_inv @ B_T_A_inv_x)
        term3 = np.transpose(xs, (0, 2, 1)) @ A_invs @ xs
        term4 = np.transpose(B_T_A_inv_x, (0, 2, 1)) @ self.A0_inv @ B_T_A_inv_x
        
        variances = term1 + term2 + term3 + term4 # (N, 1, 1)
        
        # 3. Compute Predictions p = z.T @ beta_hat + x.T @ theta_hat + alpha * sqrt(variance)
        z_beta = (z.T @ beta_hat).item()
        x_theta = np.transpose(xs, (0, 2, 1)) @ theta_hats # (N, 1, 1)
        
        # OULAD FIX: Add a small penalty to frequently recommended arms to increase diversity
        # This helps Exploration Efficiency. Scale penalty by alpha to keep it relevant.
        recs = np.array([self.arms[aid].get('times_recommended', 0) for aid in arm_ids]).reshape(-1, 1, 1)
        rec_penalty = 2.0 * self.alpha * np.log1p(recs)
        
        # Add random noise to break ties and improve sensitivity
        noise = np.random.normal(0, 0.1, (n_arms, 1, 1))
        ps = z_beta + x_theta + self.alpha * np.sqrt(np.maximum(0, variances)) + noise - rec_penalty
        
        best_idx = np.argmax(ps.flatten())
        return arm_ids[best_idx]

    def update(self, arm_id: str, z_shared: np.ndarray, x_arm: np.ndarray, reward: float, cluster_id: int = 0):
        """Update matrices using vectorized numpy operations, including Cluster-Specific parameters."""
        self._init_arm(arm_id)
        arm = self.arms[arm_id]
        z = z_shared.reshape(-1, 1)
        x = x_arm.reshape(-1, 1)
        
        A_inv = arm['A_inv']
        B = arm['B']
        b = arm['b']
        
        # 1. Update A0 and b0 (shared) - Subtract old contribution
        B_T_A_inv = B.T @ A_inv
        old_contribution_A = B_T_A_inv @ B
        old_contribution_b = B_T_A_inv @ b
        
        self.A0 -= old_contribution_A
        self.b0 -= old_contribution_b
        
        # Track cluster-specific history to subtract correctly
        last_c = arm.get('last_cluster_id')
        if last_c is not None:
            self.Ak[last_c] -= old_contribution_A
            self.bk[last_c] -= old_contribution_b
        
        # 2. Update arm-specific A and A_inv using Sherman-Morrison (Fast)
        inv_x = A_inv @ x
        denom = 1.0 + (x.T @ inv_x).item()
        arm['A_inv'] -= (inv_x @ inv_x.T) / denom
        arm['A'] += x @ x.T
        
        # 3. Update B and b
        arm['B'] += x @ z.T
        arm['b'] += reward * x
        arm['last_cluster_id'] = cluster_id
        
        # 4. Update A0 and b0 - Add new contribution
        new_A_inv = arm['A_inv']
        new_B = arm['B']
        new_b = arm['b']
        
        new_contribution_A = (new_B.T @ new_A_inv @ new_B)
        new_contribution_b = (new_B.T @ new_A_inv @ new_b)
        
        zzT = z @ z.T
        rz = reward * z
        
        self.A0 += zzT + new_contribution_A
        self.b0 += rz + new_contribution_b
        self.Ak[cluster_id] += zzT + new_contribution_A
        self.bk[cluster_id] += rz + new_contribution_b
        
        # 5. DEFER Matrix Inversion for speed
        self._dirty_inverses = True

    def _ensure_inverses(self, cluster_id: Optional[int] = None):
        """Recompute inverses only if matrices have changed."""
        if getattr(self, '_dirty_inverses', True):
            self.A0_inv = np.linalg.inv(self.A0)
            if cluster_id is not None:
                self.Ak_inv[cluster_id] = np.linalg.inv(self.Ak[cluster_id])
            else:
                for k in range(self.n_clusters):
                    self.Ak_inv[k] = np.linalg.inv(self.Ak[k])
            self._dirty_inverses = False
