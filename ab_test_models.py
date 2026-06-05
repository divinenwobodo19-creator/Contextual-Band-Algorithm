
import numpy as np
import pandas as pd
from linucb_brain import Brain
from tqdm import tqdm
import time

def main():
    print("=" * 80)
    print("  HYBRID vs DISJOINT MODEL - A/B TEST".center(80))
    print("=" * 80)
    print()

    # Step 1: Load a small test slice of OULAD data
    print("Loading test data from OULAD dataset...")
    interactions_df = pd.read_csv("data/oulad/interactions_clean.csv")
    agents_df = pd.read_csv("data/oulad/agents_clean.csv")
    arms_df = pd.read_csv("data/oulad/arms_clean.csv")

    # Use a test slice of 50,000 interactions
    test_slice = interactions_df.head(50000)
    print(f"Loaded {len(test_slice):,} test interactions!\n")

    # Step 2: Initialize both models
    print("Initializing models...")
    hybrid_brain = Brain(algorithm="hybrid", alpha=2.0, n_clusters=5)
    disjoint_brain = Brain(algorithm="disjoint", alpha=2.0)
    print("Models initialized!\n")

    # Step 3: Register students and content for both models
    print("Registering students and content...")
    for row in tqdm(agents_df.itertuples(), total=len(agents_df), desc="Registering Students"):
        hybrid_brain.add_agent(str(row.agent_id), {
            "performance_score": row.performance_score,
            "education_level": row.education_level,
            "age_band": row.age_band,
            "credits_studied": row.credits_studied,
            "imd_band": row.imd_band,
            "region_code": row.region_code
        })
        disjoint_brain.add_agent(str(row.agent_id), {
            "performance_score": row.performance_score,
            "education_level": row.education_level,
            "age_band": row.age_band,
            "credits_studied": row.credits_studied,
            "imd_band": row.imd_band,
            "region_code": row.region_code
        })

    for row in tqdm(arms_df.itertuples(), total=len(arms_df), desc="Registering Content"):
        hybrid_brain.add_arm(str(row.arm_id), {
            "activity_code": row.activity_code,
            "difficulty": row.difficulty,
            "activity_type": row.activity_type
        })
        disjoint_brain.add_arm(str(row.arm_id), {
            "activity_code": row.activity_code,
            "difficulty": row.difficulty,
            "activity_type": row.activity_type
        })
    print("Students and content registered!\n")

    # Step 4: Run A/B test
    print("Running A/B test...")
    hybrid_rewards = []
    disjoint_rewards = []
    start_time = time.time()

    for idx, row in enumerate(tqdm(test_slice.itertuples(), total=len(test_slice), desc="Processing Interactions")):
        student_id = str(row.agent_id)
        arm_id = str(row.arm_id)
        reward = float(row.reward)

        # Update both models
        hybrid_brain.update(student_id, arm_id, reward)
        disjoint_brain.update(student_id, arm_id, reward)

        # Track rewards
        hybrid_rewards.append(reward)
        disjoint_rewards.append(reward)

    end_time = time.time()
    print(f"A/B test complete! Took {end_time - start_time:.2f} seconds!\n")

    # Step 5: Calculate and print results
    print("A/B TEST RESULTS:")
    print("-" * 80)
    print(f"{'Metric':<25} | {'Hybrid':<15} | {'Disjoint':<15}")
    print("-" * 80)

    hybrid_avg = np.mean(hybrid_rewards)
    disjoint_avg = np.mean(disjoint_rewards)
    print(f"{'Average Reward':<25} | {hybrid_avg:.4f}        | {disjoint_avg:.4f}")

    hybrid_cum = np.sum(hybrid_rewards)
    disjoint_cum = np.sum(disjoint_rewards)
    print(f"{'Cumulative Reward':<25} | {hybrid_cum:.2f}      | {disjoint_cum:.2f}")

    print("-" * 80)

    if hybrid_avg > disjoint_avg:
        print("\nWINNER: Hybrid Model!")
        improvement = ((hybrid_avg - disjoint_avg) / disjoint_avg) * 100
        print(f"   Performs {improvement:.1f}% better than Disjoint!")
    elif disjoint_avg > hybrid_avg:
        print("\nWINNER: Disjoint Model!")
        improvement = ((disjoint_avg - hybrid_avg) / hybrid_avg) * 100
        print(f"   Performs {improvement:.1f}% better than Hybrid!")
    else:
        print("\nTIE: Both models perform equally!")

    print()
    print("=" * 80)
    print("  A/B TEST COMPLETE!".center(80))
    print("=" * 80)

if __name__ == "__main__":
    main()
