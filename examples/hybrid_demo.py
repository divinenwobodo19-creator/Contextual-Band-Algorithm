import random
import sys
import os
import numpy as np
import time

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from linucb_brain import Brain, Student, Content, calculate_reward

def run_hybrid_comparison():
    print("Hybrid vs Disjoint Performance Comparison")
    print("-" * 50)
    
    n_arms = 100
    n_steps = 1000
    
    # Define two brains
    brain_disjoint = Brain(alpha=1.0, model_type="disjoint")
    brain_hybrid = Brain(alpha=1.0, model_type="hybrid")
    
    # Add Students
    students_ids = ["S1", "S2", "S3"]
    for sid in students_ids:
        brain_disjoint.add_student(sid, sid)
        brain_hybrid.add_student(sid, sid)
        
    # Add 100 Content Items
    topics = ["Math", "Science"]
    content_types = ["video", "quiz", "exercise", "reading"]
    for i in range(n_arms):
        cid = f"C{i}"
        topic = topics[i % 2]
        ctype = content_types[i % 4]
        diff = random.randint(1, 5)
        brain_disjoint.add_content(cid, cid, topic, diff, ctype)
        brain_hybrid.add_content(cid, cid, topic, diff, ctype)

    def simulate(brain, label):
        print(f"\nSimulating {label}...")
        start = time.time()
        rewards = []
        for i in range(n_steps):
            sid = random.choice(students_ids)
            topic = random.choice(topics)
            
            # Recommendation
            content = brain.recommend(sid, topic=topic)
            
            # Simulated reward: all videos are good for everyone (shared pattern)
            # This is where Hybrid should excel
            if content.content_type == "video":
                base_prob = 0.7
            else:
                base_prob = 0.3
                
            obs_reward = 1.0 if random.random() < base_prob else 0.0
            brain.update(sid, content.content_id, obs_reward)
            rewards.append(obs_reward)
            
            if (i+1) % 200 == 0:
                print(f"Step {i+1}: Avg Reward = {np.mean(rewards):.4f}")
        
        end = time.time()
        print(f"Total Time: {end - start:.2f}s")
        print(f"Final Avg Reward: {np.mean(rewards):.4f}")
        return rewards

    rewards_d = simulate(brain_disjoint, "Disjoint (100 arms)")
    rewards_h = simulate(brain_hybrid, "Hybrid (100 arms)")
    
    print("\n" + "="*50)
    print(f"FINAL RESULT:")
    print(f"Disjoint Final Avg: {np.mean(rewards_d):.4f}")
    print(f"Hybrid Final Avg:   {np.mean(rewards_h):.4f}")
    print("="*50)

if __name__ == "__main__":
    run_hybrid_comparison()
