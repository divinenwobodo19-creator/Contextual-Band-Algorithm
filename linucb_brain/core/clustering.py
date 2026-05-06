import numpy as np
from typing import Dict, List, Optional
from sklearn.cluster import MiniBatchKMeans
class ClusteringEngine:
    """
    Handles dynamic clustering of students based on their context features.
    This allows sharing knowledge within student cohorts (e.g., 'Math-Visual' learners).
    """
    def __init__(self, n_clusters: int = 5, n_features: int = 17):
        self.n_clusters = n_clusters
        self.n_features = n_features
        # Using MiniBatchKMeans for online/incremental clustering
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
        self.initialized = False
        self.student_to_cluster: Dict[str, int] = {}

    def update_clusters(self, contexts: np.ndarray):
        """
        Update clusters based on a batch of student context vectors.
        """
        if contexts.shape[0] < self.n_clusters:
            return
            
        self.kmeans.partial_fit(contexts)
        self.initialized = True

    def get_cluster(self, student_id: str, context: np.ndarray) -> int:
        """
        Assign a student to a cluster based on their context.
        """
        if not self.initialized:
            # Fallback to a default cluster if not enough data yet
            return hash(student_id) % self.n_clusters
            
        cluster_id = self.kmeans.predict(context.reshape(1, -1))[0]
        self.student_to_cluster[student_id] = int(cluster_id)
        return int(cluster_id)

    def get_cluster_stats(self) -> Dict[int, int]:
        """Returns the distribution of students across clusters."""
        from collections import Counter
        return dict(Counter(self.student_to_cluster.values()))
