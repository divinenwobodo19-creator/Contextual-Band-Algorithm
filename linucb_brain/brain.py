import numpy as np
import uuid
import threading
from typing import Dict, List, Optional, Union
from .models.student import Student
from .models.content import Content
from .models.session import Session
from .core.linucb import LinUCBDisjoint
from .core.linucb_hybrid import LinUCBHybrid
from .core.lints import LinTSDisjoint
from .core.context import build_context, build_context_split, get_context_dimension, get_hybrid_dimensions
from .core.clustering import ClusteringEngine
from .diagnostics.neural_score import run_neural_diagnostics
from .diagnostics.report import render_neural_report
from .storage import save_brain, load_brain

class Brain:
    """
    The single public interface for the LinUCB Brain model.
    Now supports Clustering (COBART) and Multi-Objective Rewards.
    """
    def __init__(self, alpha: float = 1.0, auto_diagnose_every: int = 50, alpha_decay: float = 0.999, model_type: str = "disjoint", gamma: float = 1.0, n_clusters: int = 5, algorithm: Optional[str] = None, track_sessions: bool = True, max_sessions: int = 100000):
        """
        Initialize the Brain instance.
        
        Args:
            alpha (float): Exploration-exploitation tradeoff.
            auto_diagnose_every (int): Frequency of automatic diagnostics.
            alpha_decay (float): Multiplicative decay for alpha.
            model_type (str): "disjoint", "hybrid", or "ts".
            gamma (float): Discount factor.
            n_clusters (int): Number of student clusters for COBART.
            algorithm (str): Alias for model_type.
            track_sessions (bool): Whether to store session history in memory.
            max_sessions (int): Maximum number of sessions to keep in memory.
        """
        self.alpha = alpha
        self.initial_alpha = alpha
        self.alpha_decay = alpha_decay
        self.auto_diagnose_every = auto_diagnose_every
        self.model_type = algorithm if algorithm else model_type
        self.gamma = gamma
        self.n_clusters = n_clusters
        self.track_sessions = track_sessions
        self.max_sessions = max_sessions
        self.update_count = 0
        self.cumulative_regret = 0.0
        self.total_predicted_reward = 0.0
        
        # Multi-Objective Weights (can be tuned)
        self.reward_weights = {
            'improvement': 0.5,
            'completion': 0.3,
            'engagement': 0.2
        }
        
        # State
        self.students: Dict[str, Student] = {}
        self.contents: Dict[str, Content] = {}
        self.sessions: List[Session] = []
        
        # Core Model
        d = get_context_dimension()
        if self.model_type == "disjoint":
            self.model = LinUCBDisjoint(n_features=d, alpha=alpha, l2_lambda=1.0, gamma=gamma)
        elif self.model_type == "hybrid":
            k, d_h = get_hybrid_dimensions()
            self.model = LinUCBHybrid(k_shared=k, d_arm=d_h, n_clusters=n_clusters, alpha=alpha, l2_lambda=1.0)
        elif self.model_type == "ts":
            self.model = LinTSDisjoint(n_features=d, v=alpha, l2_lambda=1.0)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
            
        # Clustering Engine
        self.clustering = ClusteringEngine(n_clusters=n_clusters, n_features=d)
        
        # Last Neural Score
        self.last_neural_score: Optional[Dict[str, float]] = None

        # Thread safety lock (RLock allows re-entrancy from the same thread)
        self._lock = threading.RLock()

    def add_student(self, student_id: str, name: str, **kwargs) -> Student:
        """Add a new student profile."""
        with self._lock:
            if student_id in self.students:
                return self.students[student_id]
            student = Student(student_id=student_id, name=name, **kwargs)
            self.students[student_id] = student
            return student

    def add_agent(self, agent_id: str, features: Dict[str, float]):
        """Alias for add_student used in OULAD simulation."""
        with self._lock:
            student = self.add_student(student_id=agent_id, name=f"Student {agent_id}")
            for k, v in features.items():
                setattr(student, k, v)
            return student

    def add_content(self, content_id: str, title: str, topic: str, difficulty: int, content_type: str) -> Content:
        """Add a new content item."""
        with self._lock:
            if content_id in self.contents:
                return self.contents[content_id]
            content = Content(
                content_id=content_id, 
                title=title, 
                topic=topic, 
                difficulty=difficulty, 
                content_type=content_type
            )
            self.contents[content_id] = content
            return content

    def add_arm(self, arm_id: str, features: Dict[str, float]):
        """Alias for add_content used in OULAD simulation."""
        with self._lock:
            # Map difficulty from features if present, otherwise default to 3
            difficulty = int(features.get("difficulty", 0.5) * 4) + 1

            # Mapping activity types for better context build
            activity_type = "reading"
            if "activity_type" in features:
                atype = str(features["activity_type"]).lower()
                if "quiz" in atype: activity_type = "quiz"
                elif "video" in atype or "ouelluminate" in atype: activity_type = "video"
                elif "content" in atype or "resource" in atype: activity_type = "reading"
                elif "dataplus" in atype or "glossary" in atype: activity_type = "exercise"

            return self.add_content(
                content_id=arm_id,
                title=f"Arm {arm_id}",
                topic="OULAD",
                difficulty=difficulty,
                content_type=activity_type
            )

    def recommend(self, student_id: str, topic: Optional[str] = None, top_n: int = 1) -> Union[Content, List[Content]]:
        """
        Recommend content for a student.
        """
        with self._lock:
            if student_id not in self.students:
                raise KeyError(f"Student ID {student_id} not found.")

            student = self.students[student_id]
            available_contents = [c for c in self.contents.values() if topic is None or c.topic == topic]

            if not available_contents:
                raise ValueError("No matching content available.")

            content_ids = [c.content_id for c in available_contents]
            best_contents = []
            remaining_ids = list(content_ids)

            # 1. SPECIAL CASE: HYBRID top_n=1 (Vectorized)
            if self.model_type == "hybrid" and top_n == 1:
                z_shared, _ = build_context_split(student, self.contents[remaining_ids[0]])
                full_ctx = build_context(student, self.contents[remaining_ids[0]])
                cluster_id = self.clustering.get_cluster(student_id, full_ctx)

                x_arms = {cid: build_context_split(student, self.contents[cid])[1] for cid in remaining_ids}

                best_cid = self.model.select(remaining_ids, z_shared, x_arms, cluster_id=cluster_id)
                best_contents.append(self.contents[best_cid])
            else:
                # 2. GENERAL CASE (Looping for Disjoint, TS, or Hybrid top_n > 1)
                for _ in range(min(top_n, len(content_ids))):
                    ucbs = []
                    if self.model_type == "disjoint":
                        for cid in remaining_ids:
                            ctx = build_context(student, self.contents[cid])
                            arm = self.model.arms.get(cid)
                            if arm is None: self.model._init_arm(cid); arm = self.model.arms[cid]
                            A_inv = arm['A_inv']
                            theta = A_inv @ arm['b']
                            x = ctx.reshape(-1, 1)
                            p = theta.T @ x + self.alpha * np.sqrt(max(0, (x.T @ A_inv @ x).item()))
                            ucbs.append(p.item())
                    elif self.model_type == "ts":
                        for cid in remaining_ids:
                            self.model._init_arm(cid)
                            arm = self.model.arms[cid]
                            try:
                                theta_a = np.random.multivariate_normal(arm['mu_hat'].flatten(), (self.alpha ** 2) * arm['B_inv']).reshape(-1, 1)
                            except:
                                theta_a = arm['mu_hat']
                            ctx = build_context(student, self.contents[cid])
                            p = (theta_a.T @ ctx.reshape(-1, 1)).item()
                            ucbs.append(p)
                    elif self.model_type == "hybrid":
                        sample_ctx = build_context(student, self.contents[remaining_ids[0]])
                        cluster_id = self.clustering.get_cluster(student_id, sample_ctx)
                        z_shared, _ = build_context_split(student, self.contents[remaining_ids[0]])
                        beta_hat_global = self.model.A0_inv @ self.model.b0
                        beta_hat_cluster = self.model.Ak_inv[cluster_id] @ self.model.bk[cluster_id]
                        beta_hat = (beta_hat_global + beta_hat_cluster) / 2.0
                        for cid in remaining_ids:
                            _, x_arm = build_context_split(student, self.contents[cid])
                            self.model._init_arm(cid); arm = self.model.arms[cid]
                            A_inv = arm['A_inv']; theta_hat = A_inv @ (arm['b'] - arm['B'] @ beta_hat)
                            z = z_shared.reshape(-1, 1); x = x_arm.reshape(-1, 1)
                            z_A0_inv = z.T @ self.model.A0_inv; B_A_inv_x = arm['B'].T @ A_inv @ x
                            var = (z_A0_inv @ z - 2 * (z_A0_inv @ B_A_inv_x) + x.T @ A_inv @ x + B_A_inv_x.T @ self.model.A0_inv @ B_A_inv_x).item()
                            p = (z.T @ beta_hat + x.T @ theta_hat).item() + self.alpha * np.sqrt(max(0, var))
                            ucbs.append(p)

                    # Tie-breaking & Regret tracking
                    best_idx = np.argmax(np.array(ucbs) + np.random.normal(0, 1e-9, len(ucbs)))
                    if len(ucbs) > 1:
                        self.cumulative_regret += (ucbs[best_idx] - np.mean(ucbs))
                    best_cid = remaining_ids.pop(best_idx)
                    best_contents.append(self.contents[best_cid])

            for c in best_contents:
                c.times_recommended += 1
            return best_contents[0] if top_n == 1 else best_contents

    def calculate_multi_objective_reward(self, improvement: float, completed: bool, engaged: bool, churned: bool) -> float:
        """
        Combine multiple reward signals based on the Brain's current weights.
        """
        if churned:
            return -1.0
            
        weights = self.reward_weights
        
        # Normalize signals
        comp_signal = 1.0 if completed else -0.2
        eng_signal = 1.0 if engaged else -0.5
        imp_signal = np.clip(improvement * 5.0, -1.0, 1.0)
        
        weighted_reward = (
            weights['improvement'] * imp_signal +
            weights['completion'] * comp_signal +
            weights['engagement'] * eng_signal
        )
        
        return float(np.clip(weighted_reward, -1.0, 1.0))

    def update(self, student_id: Optional[str] = None, content_id: Optional[str] = None, reward: float = 0.0, agent_id: Optional[str] = None, arm_id: Optional[str] = None) -> None:
        """
        Update the model with a reward signal.
        Supports both (student_id, content_id) and (agent_id, arm_id) for flexibility.
        """
        with self._lock:
            s_id = student_id if student_id else agent_id
            c_id = content_id if content_id else arm_id

            if not s_id or not c_id:
                raise ValueError("Both student_id/agent_id and content_id/arm_id must be provided.")

            if s_id not in self.students:
                raise KeyError(f"Student ID {s_id} not found.")
            if c_id not in self.contents:
                raise KeyError(f"Content ID {c_id} not found.")

            student = self.students[s_id]
            content = self.contents[c_id]

            if self.model_type == "disjoint" or self.model_type == "ts":
                context = build_context(student, content)
                self.model.update(c_id, context, reward)
                context_vector = context.tolist()
            elif self.model_type == "hybrid":
                z_shared, x_arm = build_context_split(student, content)
                context = np.concatenate([z_shared, x_arm])

                cluster_id = self.clustering.get_cluster(s_id, context)

                self.model.update(c_id, z_shared, x_arm, reward, cluster_id=cluster_id)

                if hasattr(self.model, 'arms') and c_id in self.model.arms:
                    self.model.arms[c_id]['times_recommended'] = self.model.arms[c_id].get('times_recommended', 0) + 1

                context_vector = context.tolist()

                if self.update_count % 10000 == 0:
                    recent_contexts = np.array([np.array(s.context_vector) for s in self.sessions[-1000:]]) if len(self.sessions) >= 1000 else None
                    if recent_contexts is not None:
                        self.clustering.update_clusters(recent_contexts)

            if self.track_sessions:
                session = Session(
                    session_id=str(uuid.uuid4()),
                    student_id=s_id,
                    content_id=c_id,
                    context_vector=context_vector,
                    reward=reward,
                    topic=content.topic
                )
                self.sessions.append(session)

                if len(self.sessions) > self.max_sessions:
                    self.sessions = self.sessions[-self.max_sessions:]

            student.session_count += 1
            student.performance_score = (student.performance_score * (student.session_count - 1) + reward) / student.session_count

            content.times_rewarded += 1
            content.times_recommended = max(content.times_recommended, content.times_rewarded)
            n_r = content.times_rewarded
            content.avg_reward = (content.avg_reward * (n_r - 1) + reward) / n_r

            self.update_count += 1

            self.alpha = max(0.01, self.alpha * self.alpha_decay)
            self.model.alpha = self.alpha

            if self.update_count % self.auto_diagnose_every == 0:
                self.neural_score(verbose=False)

    def predict_grade(self, student_id: str, subject: str) -> float:
        """
        Predict a grade for a student in a subject.
        Uses the current performance score as a baseline for the grade.
        """
        with self._lock:
            if student_id not in self.students:
                raise KeyError(f"Student ID {student_id} not found.")

            student = self.students[student_id]

            student_sessions = [s for s in self.sessions if s.student_id == student_id and s.topic == subject]

            initial_grade = 0.5
            if subject in student.grade_history and student.grade_history[subject]:
                initial_grade = student.grade_history[subject][0]

            if not student_sessions:
                return initial_grade

            total_improvement = sum(s.reward for s in student_sessions if s.reward is not None)
            predicted = initial_grade + total_improvement

            from .utils import clip
            return clip(predicted, 0.0, 1.0)

    def neural_score(self, verbose: bool = True) -> dict:
        """Manually trigger full diagnostics."""
        with self._lock:
            scores = run_neural_diagnostics(self.students, self.contents, self.sessions, self)
            self.last_neural_score = scores

            if verbose:
                print(render_neural_report(scores))

            return scores

    def save(self, filepath: str) -> None:
        """Save brain state to file."""
        with self._lock:
            if self.model_type == "hybrid" and hasattr(self.model, '_ensure_inverses'):
                self.model._ensure_inverses()
            save_brain(self, filepath)

    @classmethod
    def load(cls, filepath: str):
        """Load brain state from file."""
        return load_brain(filepath, cls)

    def update_batch(self, updates: List[Dict]) -> None:
        """
        Perform batch updates for production scalability.
        
        Args:
            updates: List of dicts with student_id, content_id, reward.
        """
        for up in updates:
            self.update(up['student_id'], up['content_id'], up['reward'])

    def tune_parameters(self) -> Dict[str, float]:
        """
        Meta-Learner: Automatically adjust alpha and gamma based on neural score.
        """
        with self._lock:
            if not self.last_neural_score:
                return {'alpha': self.alpha, 'gamma': self.gamma}

            scores = self.last_neural_score

            if scores['convergence_score'] < 5.0:
                self.gamma = min(1.0, self.gamma + 0.01)
            if scores['exploration_score'] < 7.0:
                self.alpha = min(2.0, self.alpha + 0.1)
            elif scores['precision_score'] > 7.0 and scores['convergence_score'] < 6.0:
                self.alpha = max(0.1, self.alpha - 0.05)

            if hasattr(self.model, 'alpha'):
                self.model.alpha = self.alpha
            if hasattr(self.model, 'gamma'):
                self.model.gamma = self.gamma

            return {'alpha': self.alpha, 'gamma': self.gamma}

    def warm_start(self, sessions_data: List[Dict]) -> None:
        """
        Pre-train the brain with historical session data.
        
        Args:
            sessions_data (List[Dict]): List of session dicts containing 
                                        student_id, content_id, and reward.
        """
        with self._lock:
            for data in sessions_data:
                sid = data['student_id']
                cid = data['content_id']
                reward = data['reward']
                if sid in self.students and cid in self.contents:
                    self.update(sid, cid, reward)

    def summary(self) -> dict:
        """Return a summary of the brain state."""
        with self._lock:
            return {
                'student_count': len(self.students),
                'content_count': len(self.contents),
                'total_sessions': len(self.sessions),
                'model_type': self.model_type,
                'current_alpha': round(self.alpha, 4),
                'current_gamma': round(self.gamma, 4),
                'cumulative_regret': round(self.cumulative_regret, 2),
                'last_neural_score': self.last_neural_score.get('neural_score') if self.last_neural_score else None
            }
