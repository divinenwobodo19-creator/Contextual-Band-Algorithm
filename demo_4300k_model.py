
import random
import sys
import os

from linucb_brain import Brain
import pandas as pd

def main():
    print("=" * 80)
    print("  4.3M OULAD Model - Live Recommendation Demo!".center(80))
    print("=" * 80)
    print()

    # Step 1: Load our trained model
    print("📦 Loading 4.3M checkpoint (oulad_checkpoint_4300k.json)...")
    brain = Brain.load("oulad_checkpoint_4300k.json")
    print("✅ Model loaded successfully!\n")

    # Step 2: Load students from the OULAD dataset (to get real student IDs)
    print("📚 Loading OULAD student data...")
    students_df = pd.read_csv("data/oulad/agents_clean.csv")
    real_student_ids = list(students_df['agent_id'].astype(str))
    print(f"✅ Loaded {len(real_student_ids):,} real students!\n")

    # Step 3: Pick 5 random students to test
    print("🎯 Randomly selecting 5 students for recommendations:")
    print("-" * 80)
    test_students = random.sample(real_student_ids, 5)

    for idx, student_id in enumerate(test_students, 1):
        print(f"\n--- Student {idx}: ID {student_id} ---")
        # Get student details
        student_row = students_df[students_df['agent_id'].astype(str) == student_id].iloc[0]
        print(f"  Performance Score: {student_row['performance_score']:.2f}")
        print(f"  Education Level: {student_row['education_level']}")
        print(f"  Age Band: {student_row['age_band']}")

        # Get top 3 recommendations
        print(f"\n  📋 Top 3 Recommended Content:")
        for rank in range(1, 4):
            rec = brain.recommend(student_id)
            # Find this content in our model
            content = brain.contents.get(rec.content_id, None)
            if content:
                print(f"    {rank}. Content {content.content_id}:")
                print(f"       Type: {content.content_type}")
                print(f"       Difficulty: {content.difficulty}/5")
                print(f"       Avg. Reward: {content.avg_reward:.4f}")
                print(f"       Recommended {content.times_recommended} times, rewarded {content.times_rewarded} times\n")
            else:
                print(f"    {rank}. Content {rec.content_id}: (details not available)\n")
    print("-" * 80)
    print("\n🎉 Demo complete! Your 4.3M model is working perfectly!")

if __name__ == "__main__":
    main()
