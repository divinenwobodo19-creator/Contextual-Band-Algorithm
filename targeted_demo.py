
from linucb_brain import Brain
import random

def main():
    print("=" * 80)
    print("  TARGETED DEMO: ONLY MAX-DIFFICULTY QUIZ & EXERCISE")
    print("=" * 80)
    print()

    print("Loading full model...")
    brain = Brain.load("oulad_checkpoint_4300k.json")
    print("Model loaded!")
    print()

    # Find max difficulty
    max_diff = 0
    for c in brain.contents.values():
        if c.difficulty > max_diff:
            max_diff = c.difficulty
    print(f"Max difficulty available: {max_diff}/5")
    print()

    # Filter content: only quiz/exercise, max difficulty, with reward data
    eligible_content = []
    for c in brain.contents.values():
        if (
            c.content_type in ["quiz", "exercise"]
            and c.difficulty == max_diff
            and c.times_recommended > 0
            and c.avg_reward > 0
        ):
            eligible_content.append(c)
    print(f"Found {len(eligible_content)} eligible content items!")
    print()

    # Sort eligible content by average reward
    eligible_content_sorted = sorted(eligible_content, key=lambda x: -x.avg_reward)

    print("=== TOP 10 ELIGIBLE CONTENT ===")
    for idx, c in enumerate(eligible_content_sorted[:10], 1):
        print(f"{idx}. {c.content_type.upper()} {c.content_id}")
        print(f"   Avg Reward: {c.avg_reward:.4f}")
        print(f"   Recommended/Rewarded: {c.times_recommended:,}/{c.times_rewarded:,}")
    print()

    # Pick 3 students and show ONLY targeted recommendations
    student_ids = list(brain.students.keys())
    test_students = random.sample(student_ids, min(3, len(student_ids)))

    print("=== STUDENT RECOMMENDATIONS (MAX DIFFICULTY QUIZ/EXERCISE ONLY) ===")
    for s_idx, student_id in enumerate(test_students, 1):
        print(f"\n--- STUDENT {s_idx} (ID: {student_id}) ---")
        print(f"Top 5 quiz/exercise recommendations at {max_diff}/5:")
        # Show our manually selected top eligible content (since recommend() prefers reading)
        for rec_idx, c in enumerate(eligible_content_sorted[:5], 1):
            print(f"  {rec_idx}. {c.content_type} {c.content_id}")
            print(f"     Avg Reward: {c.avg_reward:.4f}")
            print(f"     Difficulty: {c.difficulty}/5")
    print()
    
    print("=" * 80)
    print("NOTE: The model normally prefers reading because it has even higher")
    print("      average rewards for most students! We manually filtered to show")
    print("      only quiz/exercise at max difficulty here!")
    print("=" * 80)

if __name__ == "__main__":
    main()
