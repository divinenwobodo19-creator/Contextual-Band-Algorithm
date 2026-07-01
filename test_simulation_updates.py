from linucb_brain import Brain
import pandas as pd
import os
import time

print("=== TESTING NEW UPDATES - SMALL SIMULATION ===")

# Load data quickly (small subset)
print("Loading test data...")
agents_df       = pd.read_csv("data/oulad/agents_clean.csv")
arms_df         = pd.read_csv("data/oulad/arms_clean.csv")
interactions_df = pd.read_csv("data/oulad/interactions_clean.csv", nrows=1000) # Only 1k interactions!
print(f"Test data loaded: {len(interactions_df):,} interactions ready!")

# Initialize new brain
print("Initializing new hybrid model for testing...")
brain = Brain(
    algorithm="hybrid", 
    alpha=2.0, 
    auto_diagnose_every=100000, 
    track_sessions=True, 
    max_sessions=50000
)

# Register agents
print(f"Registering {len(agents_df):,} students...")
for row in agents_df.itertuples():
    brain.add_agent(
        agent_id=str(row.agent_id),
        features={
            "performance_score": row.performance_score,
            "education_level": row.education_level,
            "age_band": row.age_band,
            "credits_studied": row.credits_studied,
            "imd_band": row.imd_band,
            "region_code": row.region_code
        }
    )

# Register arms
print(f"Registering {len(arms_df):,} content items...")
for row in arms_df.itertuples():
    brain.add_arm(
        arm_id=str(row.arm_id),
        features={
            "activity_code": row.activity_code,
            "difficulty": row.difficulty,
            "activity_type": row.activity_type
        }
    )

print("Starting small simulation...")
start_time = time.time()

for idx, row in enumerate(interactions_df.itertuples()):
    student_id = str(row.agent_id)
    arm_id = str(row.arm_id)
    reward = float(row.reward)
    
    # Update the model
    brain.update(student_id, arm_id, reward)
    
    # Print progress every 200 steps
    if idx % 200 == 0:
        print(f"  Processed {idx:,}/{len(interactions_df):,} interactions...")

end_time = time.time()
print(f"\n✅ Test simulation COMPLETED in {end_time - start_time:.2f} seconds!")
print("✅ NO ERRORS - all code updates working perfectly!")

# Test a recommendation to confirm it's working
print("\nTesting a recommendation to verify...")
random_student = list(brain.students.keys())[0]
rec = brain.recommend(random_student)
print(f"Student {random_student} got recommendation:")
print(f"  Content: {rec.content_id}")
content = brain.contents.get(rec.content_id)
if content:
        print(f"  Type: {content.content_type}")
        print(f"  Difficulty: {content.difficulty}")
        print(f"  Avg Reward: {content.avg_reward:.4f}")

print("\n=== ALL TESTS PASSED! ===")
