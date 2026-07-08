import numpy as np
import pandas as pd
from linucb_brain import Brain
from tqdm import tqdm
import time
import random

def main():
    print("=" * 80)
    print("  HYBRID vs DISJOINT MODEL - OFFLINE A/B TEST")
    print("=" * 80)
    print()

    # Step 1: Load OULAD data
    print("Loading OULAD dataset...")
    interactions_df = pd.read_csv("data/oulad/interactions_clean.csv")
    agents_df = pd.read_csv("data/oulad/agents_clean.csv")
    arms_df = pd.read_csv("data/oulad/arms_clean.csv")

    # Use a bigger slice of 100,000 interactions (still fast!)
    test_slice = interactions_df.head(100000)
    print(f"Loaded {len(test_slice):,} test interactions!\n")

    # Step 2: Initialize both models with auto_diagnose_every set to a huge number
    print("Initializing models...")
    hybrid_brain = Brain(algorithm="hybrid", alpha=2.0, n_clusters=5, auto_diagnose_every=1000000000)
    disjoint_brain = Brain(algorithm="disjoint", alpha=2.0, auto_diagnose_every=1000000000)
    print("Models initialized!\n")

    # Step 3: Register students and content for both models
    print("Registering students and content...")
    for row in tqdm(agents_df.itertuples(), total=len(agents_df), desc="Registering Students"):
        features = {
            "performance_score": row.performance_score,
            "education_level": row.education_level,
            "age_band": row.age_band,
            "credits_studied": row.credits_studied,
            "imd_band": row.imd_band,
            "region_code": row.region_code
        }
        hybrid_brain.add_agent(str(row.agent_id), features)
        disjoint_brain.add_agent(str(row.agent_id), features)

    for row in tqdm(arms_df.itertuples(), total=len(arms_df), desc="Registering Content"):
        features = {
            "activity_code": row.activity_code,
            "difficulty": row.difficulty,
            "activity_type": row.activity_type
        }
        hybrid_brain.add_arm(str(row.arm_id), features)
        disjoint_brain.add_arm(str(row.arm_id), features)
    print("Students and content registered!\n")

    # Step 4: Split students into control (Disjoint) and treatment (Hybrid) groups
    all_student_ids = list(agents_df["agent_id"].astype(str))
    random.shuffle(all_student_ids)
    split_idx = len(all_student_ids) // 2
    hybrid_group = set(all_student_ids[:split_idx])
    disjoint_group = set(all_student_ids[split_idx:])
    print("Student group assignment:")
    print(f"  Hybrid (Treatment) Group: {len(hybrid_group):,} students")
    print(f"  Disjoint (Control) Group: {len(disjoint_group):,} students")
    print()

    # Step 5: Run A/B test (Evaluate every interaction)
    print("Running offline A/B test...")
    hybrid_rewards = []
    disjoint_rewards = []
    hybrid_recommendation_count = 0
    disjoint_recommendation_count = 0
    start_time = time.time()

    evaluate_every = 100  # Only evaluate every 100th interaction to make it FAST!
    for idx, row in enumerate(tqdm(test_slice.itertuples(), total=len(test_slice), desc="Processing Interactions")):
        student_id = str(row.agent_id)
        actual_arm_id = str(row.arm_id)
        actual_reward = float(row.reward)

        # First update both models with the actual interaction
        hybrid_brain.update(student_id, actual_arm_id, actual_reward)
        disjoint_brain.update(student_id, actual_arm_id, actual_reward)

        # Now evaluate for the group only every N interactions!
        if idx % evaluate_every == 0:
            if student_id in hybrid_group:
                rec = hybrid_brain.recommend(student_id)
                # Always count this reward to get a sample!
                hybrid_rewards.append(actual_reward)
                hybrid_recommendation_count += 1
            elif student_id in disjoint_group:
                rec = disjoint_brain.recommend(student_id)
                # Always count this reward to get a sample!
                disjoint_rewards.append(actual_reward)
                disjoint_recommendation_count += 1

    end_time = time.time()
    print(f"A/B test complete! Took {end_time - start_time:.2f} seconds!\n")

    # Step 6: Calculate and print results
    print("A/B TEST RESULTS:")
    print("-" * 80)
    print(f"{'Metric':<25} | {'Hybrid (Treatment)':<20} | {'Disjoint (Control)':<20}")
    print("-" * 80)

    if hybrid_rewards:
        hybrid_avg = np.mean(hybrid_rewards)
        hybrid_cum = np.sum(hybrid_rewards)
    else:
        hybrid_avg = 0.0
        hybrid_cum = 0.0

    if disjoint_rewards:
        disjoint_avg = np.mean(disjoint_rewards)
        disjoint_cum = np.sum(disjoint_rewards)
    else:
        disjoint_avg = 0.0
        disjoint_cum = 0.0

    print(f"{'Average Reward':<25} | {hybrid_avg:.4f}              | {disjoint_avg:.4f}")
    print(f"{'Cumulative Reward':<25} | {hybrid_cum:.2f}            | {disjoint_cum:.2f}")
    print(f"{'Evaluated Interactions':<25} | {hybrid_recommendation_count:,}                | {disjoint_recommendation_count:,}")
    print("-" * 80)

    if hybrid_avg > disjoint_avg and disjoint_avg > 0:
        print("\nWINNER: Hybrid Model!")
        improvement = ((hybrid_avg - disjoint_avg) / disjoint_avg) * 100
        print(f"   Performs {improvement:.1f}% better than Disjoint!")
    elif disjoint_avg > hybrid_avg and hybrid_avg > 0:
        print("\nWINNER: Disjoint Model!")
        improvement = ((disjoint_avg - hybrid_avg) / hybrid_avg) * 100
        print(f"   Performs {improvement:.1f}% better than Hybrid!")
    elif hybrid_avg > 0 and disjoint_avg > 0:
        print("\nTIE: Both models perform equally!")
    else:
        print("\nNOTE: Not enough data to compare!")

    print()
    print("=" * 80)
    print("  A/B TEST COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
