import numpy as np
import time
import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from linucb_brain.core.linucb import LinUCBDisjoint

def run_stress_test(name, n_arms, n_features, n_steps, scenario_fn):
    """Generic stress test runner."""
    print(f"\n--- STRESS TEST: {name} ---")
    print(f"Arms: {n_arms}, Features: {n_features}, Steps: {n_steps}")
    
    model = LinUCBDisjoint(n_features=n_features, alpha=1.0)
    arm_ids = [f"arm_{i}" for i in range(n_arms)]
    
    start_time = time.time()
    rewards = []
    
    # Pre-generate some data to avoid loop overhead if possible, but LinUCB is sequential
    for i in range(n_steps):
        context = np.random.randn(n_features)
        
        # Scenario logic for ground truth reward
        true_reward = scenario_fn(i, n_steps, context, n_arms)
        
        # Model Prediction
        selected_arm = model.select(arm_ids, context)
        
        # Observed reward (add noise)
        # Note: scenario_fn returns reward for SELECTED arm or something?
        # Let's make it simpler: scenario_fn(step, total_steps, context, arm_idx)
        arm_idx = int(selected_arm.split('_')[1])
        obs_reward = scenario_fn(i, n_steps, context, arm_idx)
        obs_reward = np.clip(obs_reward + np.random.normal(0, 0.1), -1.0, 1.0)
        
        # Update Model
        model.update(selected_arm, context, obs_reward)
        rewards.append(obs_reward)
        
        if (i + 1) % (n_steps // 5) == 0:
            avg_reward = np.mean(rewards[-(n_steps // 5):])
            print(f"Step {i+1}: Recent Avg Reward = {avg_reward:.4f}")

    end_time = time.time()
    print(f"Total Time: {end_time - start_time:.2f}s")
    print(f"Final Avg Reward: {np.mean(rewards):.4f}")
    return model, rewards

# SCENARIO 1: High Dimensionality & Large Arm Count
# Test if the Sherman-Morrison O(d^2) implementation holds up.
def scenario_high_dim(step, total_steps, context, arm_idx):
    # Reward is only based on first 5 features of arm 0 and arm 1
    if arm_idx == 0:
        return np.sum(context[:5])
    elif arm_idx == 1:
        return -np.sum(context[:5])
    return 0.0

# SCENARIO 2: Non-Stationary (Adversarial)
# True weights shift halfway.
def scenario_non_stationary(step, total_steps, context, arm_idx):
    # Arm 0 is good in first half, Arm 1 is good in second half
    if step < total_steps // 2:
        return 1.0 if arm_idx == 0 else 0.0
    else:
        return 1.0 if arm_idx == 1 else 0.0

# SCENARIO 3: Pure Noise (Zero Signal)
def scenario_pure_noise(step, total_steps, context, arm_idx):
    return np.random.normal(0, 1.0) # Completely random

# SCENARIO 4: Irrelevant Features (Needle in a Haystack)
def scenario_needle(step, total_steps, context, arm_idx):
    # Only the VERY LAST feature determines the reward
    # If it's positive, arm 0 is good. If negative, arm 1 is good.
    if context[-1] > 0:
        return 1.0 if arm_idx == 0 else -1.0
    else:
        return 1.0 if arm_idx == 1 else -1.0

if __name__ == "__main__":
    # Test 1: Max Limit for Variables (High Dim + Large Arms)
    # 100 features, 500 arms, 2000 steps
    # This tests O(d^2) and memory for 500 matrices of 100x100
    # 500 * 100 * 100 * 8 bytes (float64) = ~40MB (Safe)
    run_stress_test("High Dim & Large Arms", 500, 100, 2000, scenario_high_dim)
    
    # Test 2: Non-Stationary (Adversarial)
    # 10 features, 10 arms, 5000 steps
    run_stress_test("Non-Stationary (Shift)", 10, 10, 5000, scenario_non_stationary)
    
    # Test 3: Pure Noise
    # 10 features, 10 arms, 2000 steps
    run_stress_test("Pure Noise (Zero Signal)", 10, 10, 2000, scenario_pure_noise)
    
    # Test 4: Needle in Haystack
    # 50 features, 2 arms, 5000 steps
    run_stress_test("Needle in Haystack", 2, 50, 5000, scenario_needle)
