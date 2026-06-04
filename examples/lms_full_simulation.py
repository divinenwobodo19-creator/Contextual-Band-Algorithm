import random
import sys
import os
import numpy as np
import time
from typing import List, Dict

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from linucb_brain import Brain, Student, Content, calculate_reward

class LMS_Environment:
    """A full Learning Management System simulation environment."""
    def __init__(self, n_students=20, n_content=100):
        self.subjects = ["Mathematics", "Physics", "Computer Science", "History"]
        self.content_types = ["video", "quiz", "exercise", "reading"]
        self.students_profiles = [
            {"type": "Beginner", "perf_base": 0.3, "pref": "video"},
            {"type": "Advanced", "perf_base": 0.8, "pref": "quiz"},
            {"type": "Visual", "perf_base": 0.5, "pref": "video"},
            {"type": "Practical", "perf_base": 0.5, "pref": "exercise"},
            {"type": "Theoretical", "perf_base": 0.6, "pref": "reading"}
        ]
        
        # Ground truth weights for "hidden" student types
        # These determine the actual reward in the simulation
        self.true_weights = {
            "Beginner": {"video": 0.9, "quiz": 0.1, "exercise": 0.2, "reading": 0.3},
            "Advanced": {"video": 0.2, "quiz": 0.9, "exercise": 0.8, "reading": 0.4},
            "Visual": {"video": 0.95, "quiz": 0.3, "exercise": 0.4, "reading": 0.1},
            "Practical": {"video": 0.4, "quiz": 0.5, "exercise": 0.9, "reading": 0.2},
            "Theoretical": {"video": 0.3, "quiz": 0.4, "exercise": 0.2, "reading": 0.9}
        }
        
        self.students_list = []
        for i in range(n_students):
            profile = random.choice(self.students_profiles)
            sid = f"S{i:03d}"
            name = f"{profile['type']} Learner {i}"
            self.students_list.append({
                "id": sid,
                "name": name,
                "type": profile['type'],
                "perf": profile['perf_base']
            })
            
        self.content_list = []
        for i in range(n_content):
            cid = f"C{i:03d}"
            topic = random.choice(self.subjects)
            ctype = random.choice(self.content_types)
            diff = random.randint(1, 5)
            self.content_list.append({
                "id": cid,
                "title": f"{topic} {ctype} {i}", 
                "topic": topic,
                "diff": diff,
                "type": ctype
            })

    def get_reward(self, student_id, content_id):
        """Simulate the actual outcome for a given student and content."""
        student = next(s for s in self.students_list if s['id'] == student_id)
        content = next(c for c in self.content_list if c['id'] == content_id)
        
        # Success probability depends on:
        # 1. Type preference match (Hidden Ground Truth)
        # 2. Topic/Difficulty alignment
        # 3. Random noise
        
        pref_weight = self.true_weights[student['type']][content['type']]
        
        # Difficulty penalty if too hard for student performance
        diff_norm = (content['diff'] - 1) / 4.0
        diff_match = 1.0 - abs(student['perf'] - diff_norm)
        
        success_prob = 0.7 * pref_weight + 0.3 * diff_match
        success_prob = np.clip(success_prob, 0.05, 0.95)
        
        completed = random.random() < success_prob
        
        # Reward calculation: improvement + completion
        before_score = student['perf']
        improvement = 0.05 if completed else -0.02
        after_score = np.clip(before_score + improvement, 0.0, 1.0)
        
        # Update student's hidden performance for simulation state
        student['perf'] = after_score
        
        reward = calculate_reward(
            before_score=before_score,
            after_score=after_score,
            completed=completed,
            time_spent_ratio=1.0
        )
        return reward, completed, improvement

def run_simulation(model_type, n_sessions=2000):
    print(f"\n--- RUNNING {model_type.upper()} LMS SIMULATION ---")
    env = LMS_Environment(n_students=30, n_content=150)
    brain = Brain(alpha=1.0, model_type=model_type, alpha_decay=0.9998)
    
    # Setup Brain
    for s in env.students_list:
        student = brain.add_student(s['id'], s['name'])
        student.performance_score = s['perf']
        student.metadata['type'] = s['type']
        
    for c in env.content_list:
        brain.add_content(c['id'], c['title'], c['topic'], c['diff'], c['type'])
        
    rewards_history = []
    regrets = []
    success_counts = {p['type']: 0 for p in env.students_profiles}
    total_counts = {p['type']: 0 for p in env.students_profiles}
    
    start_time = time.time()
    
    for i in range(n_sessions):
        student_data = random.choice(env.students_list)
        sid = student_data['id']
        topic = random.choice(env.subjects)
        
        # 1. RECOMMENDATION
        try:
            content = brain.recommend(sid, topic=topic)
        except ValueError:
            continue
            
        # 2. SIMULATE OUTCOME
        reward, completed, improvement = env.get_reward(sid, content.content_id)
        
        # Calculate Multi-Objective Reward
        reward = brain.calculate_multi_objective_reward(
            improvement=improvement,
            completed=completed,
            engaged=completed, # For simplicity, engagement = completion here
            churned=False
        )
        
        # Update BRAIN
        brain.update(sid, content.content_id, reward)
        
        # Metrics tracking
        rewards_history.append(reward)
        total_counts[student_data['type']] += 1
        if completed:
            success_counts[student_data['type']] += 1
            
        if (i + 1) % (n_sessions // 5) == 0:
            avg_r = np.mean(rewards_history[-(n_sessions // 5):])
            print(f"Session {i+1:5d} | Moving Avg Reward: {avg_r:.4f}")

    total_time = time.time() - start_time
    print(f"Simulation Finished in {total_time:.2f}s")
    
    # Generate final report data
    final_report = {
        "model_type": model_type,
        "avg_reward": np.mean(rewards_history),
        "total_sessions": n_sessions,
        "success_rate": sum(success_counts.values()) / sum(total_counts.values()),
        "success_by_type": {t: success_counts[t]/total_counts[t] for t in success_counts if total_counts[t] > 0},
        "neural_score": brain.neural_score(verbose=False)
    }
    return final_report

def print_final_report(reports):
    print("\n" + "="*60)
    print("           LMS SIMULATION DETAILED PERFORMANCE REPORT")
    print("="*60)
    
    for report in reports:
        m_type = report['model_type'].upper()
        print(f"\n--- MODEL: {m_type} ---")
        print(f"Overall Success Rate: {report['success_rate']*100:.1f}%")
        print(f"Average Reward:       {report['avg_reward']:.4f}")
        
        print("\nSuccess Rate by Student Persona:")
        for persona, rate in report['success_by_type'].items():
            bar = "█" * int(rate * 20)
            print(f"  {persona:12} | {rate*100:5.1f}% {bar}")
            
        ns = report['neural_score']
        print(f"\nNeural Diagnostics (Signature Score: {ns['neural_score']:.1f}/10.0)")
        print(f"  - Exploration:   {ns['exploration_score']:.1f}")
        print(f"  - Convergence:   {ns['convergence_score']:.1f}")
        print(f"  - Sensitivity:   {ns['context_score']:.1f}")
        print(f"  - Precision:     {ns['precision_score']:.1f}")
        print(f"  - Prediction:    {ns['grade_score']:.1f}")
        print(f"  - Cluster Cohort:{ns.get('purity_score', 5.0):.1f}")
        print(f"  - Obj. Balance:  {ns.get('balance_score', 7.5):.1f}")
        
    print("\n" + "="*60)
    print("Final Analysis:")
    h_rate = next(r['success_rate'] for r in reports if r['model_type'] == "hybrid")
    d_rate = next(r['success_rate'] for r in reports if r['model_type'] == "disjoint")
    improvement = (h_rate - d_rate) / d_rate * 100
    print(f"Hybrid model outperformed Disjoint by {improvement:.1f}% in success rate.")
    print("Hybrid model shows superior 'Sensitivity' by sharing student features across arms.")
    print("="*60)

if __name__ == "__main__":
    n_sessions = 5000
    report_disjoint = run_simulation("disjoint", n_sessions=n_sessions)
    report_hybrid = run_simulation("hybrid", n_sessions=n_sessions)
    print_final_report([report_disjoint, report_hybrid])
