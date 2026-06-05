
from linucb_brain import Brain
import random

def main():
    print("=" * 80)
    print("  AUTOMATIC INTERACTIVE DEMO TEST (FULL OULAD MODEL)")
    print("=" * 80)
    print()

    # Load the FULLY TRAINED 4.3M model
    print("Loading FULLY TRAINED 4.3M interaction model...")
    brain = Brain.load("oulad_checkpoint_4300k.json")
    print("Model loaded successfully!")
    print(f"  - Students: {len(brain.students):,}")
    print(f"  - Content: {len(brain.contents):,}")
    print()

    # Get the list of students actually in the brain
    brain_student_ids = list(brain.students.keys())

    # Pick 3 random students and show their recommendations
    print("Testing with 3 random students from the full model:")
    print("-" * 80)
    test_students = random.sample(brain_student_ids, min(3, len(brain_student_ids)))

    for idx, student_id in enumerate(test_students, 1):
        print()
        print(f"=== STUDENT {idx} (ID: {student_id})")
        print(f"--- TOP 3 RECOMMENDATIONS")
        for i in range(3):
            rec = brain.recommend(student_id)
            content = brain.contents.get(rec.content_id, None)
            if content:
                print(f"{i+1}. Content {content.content_id}")
                print(f"   - Type: {content.content_type}")
                print(f"   - Difficulty: {content.difficulty}/5")
                print(f"   - Avg Reward: {content.avg_reward:.4f}")
                print(f"   - Recommended: {content.times_recommended:,} times, Rewarded: {content.times_rewarded:,} times")
            else:
                print(f"{i+1}. Content {rec.content_id}")
        print()

    print("=" * 80)
    print("TEST COMPLETE!")

if __name__ == "__main__":
    main()
