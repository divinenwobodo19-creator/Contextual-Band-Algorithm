import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from linucb_brain import Brain

def run_grade_predictor():
    print("Grade Prediction Dashboard")
    print("-" * 30)
    
    if not os.path.exists("brain_state.json"):
        print("Error: brain_state.json not found. Run lms_demo.py first.")
        return
        
    # 1. Load Brain
    print("Loading brain state from brain_state.json...")
    brain = Brain.load("brain_state.json")
    
    # 2. Print Summary
    summary = brain.summary()
    print(f"Loaded {summary['student_count']} students, {summary['content_count']} content items.")
    print(f"Total sessions: {summary['total_sessions']}")
    
    # 3. Predict Grades
    print("\nPredicted Grades per Student:")
    print(f"{'Student Name':<15} | {'Subject':<10} | {'Predicted Grade':<15}")
    print("-" * 50)
    
    subjects = ["Math", "Science", "History"]
    for student_id, student in brain.students.items():
        for subject in subjects:
            predicted = brain.predict_grade(student_id, subject)
            print(f"{student.name:<15} | {subject:<10} | {predicted:.2f}")

if __name__ == "__main__":
    run_grade_predictor()
