import random
import sys
import os
import numpy as np
from typing import List, Dict

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from linucb_brain import Brain, calculate_reward

def run_churn_simulation():
    print("Running Churn-Aware LMS Simulation...")
    print("-" * 50)
    
    # Initialize Brain with Meta-Learning enabled (auto_diagnose will trigger tuning if we add it)
    # For this demo, we'll manually call tune_parameters
    brain = Brain(alpha=1.0, model_type="disjoint", auto_diagnose_every=100)
    
    # 1. Setup Environment
    student_id = "S_PICKY"
    brain.add_student(student_id, "Picky Learner")
    
    # Content: Math (Likes) vs History (Hates)
    brain.add_content("C_MATH", "Math Video", "Math", 3, "video")
    brain.add_content("C_HIST", "History Video", "History", 3, "video")
    
    history_streak = 0
    sessions = 500
    active = True
    
    for i in range(sessions):
        if not active:
            print(f"Session {i}: Student has CHURNED. Simulation ending.")
            break
            
        # Recommend
        content = brain.recommend(student_id)
        
        # Outcome logic
        churned = False
        engaged = True
        
        if content.topic == "History":
            history_streak += 1
            engaged = False
            # If student gets 3 history items in a row, they churn
            if history_streak >= 3:
                churned = True
                active = False
        else:
            history_streak = 0
            engaged = True
            
        # Calculate Advanced Reward
        reward = calculate_reward(
            before_score=0.5,
            after_score=0.6 if engaged else 0.4,
            completed=engaged,
            time_spent_ratio=1.0,
            engaged=engaged,
            churned=churned
        )
        
        # Update
        brain.update(student_id, content.content_id, reward)
        
        if (i+1) % 50 == 0:
            summary = brain.summary()
            print(f"Step {i+1} | Alpha: {summary['current_alpha']} | Regret: {summary['cumulative_regret']}")
            # Auto-tune parameters based on diagnostics
            brain.neural_score(verbose=False)
            brain.tune_parameters()

    print("-" * 50)
    if active:
        print("Success: Brain learned to avoid History and kept the student engaged!")
    else:
        print("Failure: Brain pushed too much History and lost the student.")
    
    print(f"Final Summary: {brain.summary()}")

if __name__ == "__main__":
    run_churn_simulation()
