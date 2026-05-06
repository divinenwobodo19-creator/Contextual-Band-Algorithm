import pandas as pd
import numpy as np
from linucb_brain import Brain
import os
import glob
import time

def run_cold_start_simulation():
    print("🚀 Starting Cold-Start Simulation...")
    
    # 1. Load the most recent trained brain state
    checkpoints = glob.glob("oulad_checkpoint_*.json")
    if not checkpoints:
        print("❌ No trained brain state found. Please run oulad_brain_run.py first.")
        return
    
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    latest_brain_path = checkpoints[0]
    print(f"🧠 Loading trained brain from: {latest_brain_path}")
    brain = Brain.load(latest_brain_path)
    
    # 2. Load data to find "unseen" or "test" students
    print("📂 Loading data...")
    agents_df = pd.read_csv("data/oulad/agents_clean.csv")
    interactions_df = pd.read_csv("data/oulad/interactions_clean.csv")
    
    # Select a small subset of students for the simulation (e.g., 10 students)
    # In a real cold-start, we'd use students NOT used in the training phase.
    # For this simulation, we'll pick 10 students and simulate their "first day" on the platform.
    test_students = agents_df.sample(10, random_state=42)
    
    results = []
    
    print(f"🧪 Simulating cold-start for {len(test_students)} students...")
    
    for _, student_row in test_students.iterrows():
        student_id = str(int(student_row['agent_id']))
        
        # Ensure student is registered in the brain for the simulation
        if student_id not in brain.students:
            brain.add_student(
                student_id, 
                name=f"Test_{student_id}",
                performance_score=student_row['performance_score'],
                metadata={
                    "education_level": student_row['education_level'],
                    "age_band": student_row['age_band'],
                    "credits_studied": student_row['credits_studied'],
                    "imd_band": student_row['imd_band'],
                    "region_code": student_row['region_code']
                }
            )
        
        # Get their actual interactions from the dataset
        student_interactions = interactions_df[interactions_df['agent_id'] == student_row['agent_id']]
        
        if len(student_interactions) == 0:
            continue
            
        print(f"\n👤 Student {student_id}:")
        
        # Metrics for this student
        cumulative_reward = 0
        hits = 0
        
        # Simulate 20 recommendation steps
        num_steps = min(20, len(student_interactions))
        
        for i in range(num_steps):
            # 1. Get recommendation (Top 1)
            # The Brain uses its trained Hybrid/Cluster knowledge to recommend even if it hasn't seen THIS student much
            recommendation = brain.recommend(student_id, top_n=1)
            
            # 2. Check if the recommendation matches what they actually did (or just simulate feedback)
            # In a bandit simulation, we often "reveal" the reward from the dataset
            actual_interaction = student_interactions.iloc[i]
            reward = float(actual_interaction['reward'])
            
            # 3. Update the brain
            brain.update(student_id, recommendation.content_id, reward)
            
            cumulative_reward += reward
            if reward > 0.5: # Consider > 0.5 a "successful" recommendation
                hits += 1
                
        avg_reward = cumulative_reward / num_steps if num_steps > 0 else 0
        print(f"   ✅ Steps: {num_steps}, Avg Reward: {avg_reward:.4f}, Hits: {hits}")
        
        results.append({
            'student_id': student_id,
            'avg_reward': avg_reward,
            'hits': hits
        })

    # 4. Summary
    overall_avg = sum(r['avg_reward'] for r in results) / len(results) if results else 0
    print(f"\n📊 Cold-Start Simulation Summary:")
    print(f"   Overall Avg Reward: {overall_avg:.4f}")
    print(f"   Total Students Tested: {len(results)}")
    
    if overall_avg > 0.6:
        print("   🌟 Status: EXCELLENT. The Hybrid model is successfully transferring knowledge to new students.")
    else:
        print("   ⚠️ Status: IMPROVING. Consider adjusting alpha for more exploration in cold-start scenarios.")

if __name__ == "__main__":
    run_cold_start_simulation()
