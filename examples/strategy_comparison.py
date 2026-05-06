import random
import sys
import os
import numpy as np
import time
from typing import List, Dict

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from linucb_brain import Brain, Student, Content, calculate_reward

class NonStationaryLMS:
    """LMS where student preferences shift over time."""
    def __init__(self):
        self.subjects = ["Math"]
        self.content_types = ["video", "quiz", "exercise", "reading"]
        self.student_id = "S001"
        self.student_name = "Dynamic Learner"
        
        # Initial Preference: Videos are best
        self.pref_weights = {"video": 0.9, "quiz": 0.2, "exercise": 0.3, "reading": 0.4}
        
    def shift_preferences(self):
        """Shift preference: Exercises are now best."""
        print("\n[EVENT] Student preference shift: Exercises are now the top choice!")
        self.pref_weights = {"video": 0.2, "quiz": 0.3, "exercise": 0.9, "reading": 0.4}

    def get_outcome(self, content_type):
        prob = self.pref_weights[content_type]
        completed = random.random() < prob
        reward = 1.0 if completed else 0.0
        return reward

def run_strategy_test(model_type, gamma=1.0, n_steps=2000):
    env = NonStationaryLMS()
    brain = Brain(alpha=1.0, model_type=model_type, gamma=gamma, alpha_decay=1.0)
    
    brain.add_student(env.student_id, env.student_name)
    for i, ctype in enumerate(env.content_types):
        brain.add_content(f"C{i}", f"Content {ctype}", "Math", 3, ctype)
        
    rewards = []
    moving_avg = []
    
    for i in range(n_steps):
        if i == n_steps // 2:
            env.shift_preferences()
            
        # Recommend
        content = brain.recommend(env.student_id)
        
        # Outcome
        reward = env.get_outcome(content.content_type)
        
        # Update
        brain.update(env.student_id, content.content_id, reward)
        
        rewards.append(reward)
        moving_avg.append(np.mean(rewards[-100:]))
        
    return moving_avg

if __name__ == "__main__":
    print("Comparing Strategies in Non-Stationary Environment...")
    steps = 3000
    
    # 1. Standard LinUCB (No forgetting)
    print("\nRunning Standard LinUCB...")
    ma_linucb = run_strategy_test("disjoint", gamma=1.0, n_steps=steps)
    
    # 2. Discounted LinUCB (Forgetful)
    print("\nRunning Discounted LinUCB (gamma=0.95)...")
    ma_discounted = run_strategy_test("disjoint", gamma=0.95, n_steps=steps)
    
    # 3. Thompson Sampling
    print("\nRunning Thompson Sampling...")
    ma_ts = run_strategy_test("ts", n_steps=steps)
    
    print("\n" + "="*50)
    print("FINAL RESULTS (Avg Reward in last 500 steps):")
    print(f"Standard LinUCB:    {np.mean(ma_linucb[-500:]):.4f}")
    print(f"Discounted LinUCB:  {np.mean(ma_discounted[-500:]):.4f}")
    print(f"Thompson Sampling: {np.mean(ma_ts[-500:]):.4f}")
    print("="*50)
    
    if np.mean(ma_discounted[-500:]) > np.mean(ma_linucb[-500:]):
        print("Insight: Discounted LinUCB adapted better to the preference shift.")
    if np.mean(ma_ts[-500:]) > np.mean(ma_linucb[-500:]):
        print("Insight: Thompson Sampling showed robust adaptation.")
