import random
import sys
import os
import numpy as np

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from linucb_brain import Brain, Student, Content, calculate_reward

def run_lms_demo():
    print("Starting LMS Simulation Demo...")
    brain = Brain(alpha=1.0, auto_diagnose_every=500, alpha_decay=0.9995)

    # 1. Add Students
    students_data = [
        ("S1", "Alice", 0.8), # High performing
        ("S2", "Bob", 0.4),   # Low performing
        ("S3", "Charlie", 0.6), # Average
        ("S4", "Diana", 0.9), # Top student
        ("S5", "Evan", 0.3)    # Struggling
    ]
    for sid, name, perf in students_data:
        s = brain.add_student(sid, name)
        s.performance_score = perf
        s.grade_history = {"Math": [perf-0.1, perf], "Science": [perf]}

    # 2. Add Content
    topics = ["Math", "Science", "History"]
    content_types = ["video", "quiz", "exercise", "reading"]
    for i in range(10):
        topic = random.choice(topics)
        ctype = random.choice(content_types)
        difficulty = random.randint(1, 5)
        brain.add_content(f"C{i}", f"{topic} {ctype} {i}", topic, difficulty, ctype)

    # 3. Simulate 5000 Sessions
    print("Simulating 5000 learning sessions...")
    for i in range(5000):
        student_id = random.choice([s[0] for s in students_data])
        topic = random.choice(topics)
        
        # Periodically change student's current topic
        if i % 100 == 0:
            brain.students[student_id].current_topic = topic
        
        # Get recommendation
        try:
            content = brain.recommend(student_id, topic=topic)
        except ValueError:
            continue
            
        # Simulate interaction
        student = brain.students[student_id]
        
        # Highly context-sensitive reward logic
        # 1. Base prob from student performance
        success_prob = student.performance_score * 0.5
        
        # 2. Topic Match Bonus (if current student topic matches content topic)
        if student.current_topic == content.topic:
            success_prob += 0.2
            
        # 3. Content Type Affinity
        # Alice (High perf) likes Quizzes/Exercises
        if student_id == "S1" and content.content_type in ["quiz", "exercise"]:
            success_prob += 0.3
        # Evan (Struggling) likes Videos
        elif student_id == "S5" and content.content_type == "video":
            success_prob += 0.4
        # Diana (Top student) likes Exercises
        elif student_id == "S4" and content.content_type == "exercise":
            success_prob += 0.3
            
        # 4. Difficulty matching
        # If student perf is high, they handle high difficulty better
        if student.performance_score > 0.7 and content.difficulty >= 4:
            success_prob += 0.1
        elif student.performance_score < 0.4 and content.difficulty <= 2:
            success_prob += 0.1
            
        success_prob = np.clip(success_prob, 0.05, 0.95)
        completed = random.random() < success_prob
        
        # Update student performance based on success
        if completed:
            student.performance_score = min(1.0, student.performance_score + 0.02)
        else:
            student.performance_score = max(0.0, student.performance_score - 0.01)
            
        # Calculate reward
        reward = calculate_reward(
            before_score=student.performance_score - (0.02 if completed else -0.01),
            after_score=student.performance_score,
            completed=completed,
            time_spent_ratio=1.0
        )
        
        # Update brain
        brain.update(student_id, content.content_id, reward)

    # 4. Final Diagnostics
    print("\nSimulation Complete. Final Report:")
    brain.neural_score(verbose=True)

    # 5. Save Brain
    print("\nSaving brain state to brain_state.json...")
    brain.save("brain_state.json")
    print("Done.")

if __name__ == "__main__":
    run_lms_demo()
