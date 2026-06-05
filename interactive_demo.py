
from linucb_brain import Brain
import pandas as pd
import random

def main():
    print("=" * 80)
    print("  OULAD CONTEXTUAL BANDIT - INTERACTIVE DEMO".center(80))
    print("=" * 80)
    print()

    # Step 1: Load our trained model
    print("Loading trained OULAD model...")
    brain = Brain.load("brain_state.json")
    print("Model loaded successfully!")
    print(f"   - Students: {len(brain.students):,}")
    print(f"   - Content:  {len(brain.contents):,}")
    print()

    # Step 2: Load student data to pick real students
    print("Loading student database...")
    students_df = pd.read_csv("data/oulad/agents_clean.csv")
    real_student_ids = list(students_df['agent_id'].astype(str))
    print("Loaded student database!\n")

    while True:
        print("-" * 80)
        print("What would you like to do?")
        print("1. Get recommendations for a random student")
        print("2. Get recommendations for a specific student")
        print("3. Exit")
        choice = input("> ")

        if choice == "1":
            student_id = random.choice(real_student_ids)
            print(f"\nRandom student selected: {student_id}")
            show_recommendations(brain, student_id)
        elif choice == "2":
            student_id = input("\nEnter student ID: ").strip()
            if student_id not in brain.students:
                print(f"Student {student_id} not found!")
            else:
                show_recommendations(brain, student_id)
        elif choice == "3":
            print("\nGoodbye!")
            break
        else:
            print("\nInvalid choice!")


def show_recommendations(brain, student_id):
    print("\nTop 5 Recommendations:")
    for i in range(5):
        rec = brain.recommend(student_id)
        content = brain.contents.get(rec.content_id)
        if content:
            print(f"{i+1}. {content.title}")
            print(f"   Type: {content.content_type}, Difficulty: {content.difficulty}/5")
            print(f"   Avg Reward: {content.avg_reward:.4f}")
            print()
        else:
            print(f"{i+1}. Content {rec.content_id} (details missing)")


if __name__ == "__main__":
    main()
