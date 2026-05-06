import numpy as np
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from linucb_brain import Brain, Student, Content, calculate_reward, build_context

def debug_diagnostics():
    print("--- FIX 1 & 2: REWARD & CONTEXT DEBUG ---")
    brain = Brain(alpha=1.0, model_type="disjoint")
    
    # Setup Beginner and Advanced students
    s_beginner = brain.add_student("S_BEG", "Beginner Student")
    s_beginner.performance_score = 0.2
    s_beginner.session_count = 5
    
    s_advanced = brain.add_student("S_ADV", "Advanced Student")
    s_advanced.performance_score = 0.9
    s_advanced.session_count = 50
    
    # Setup Content
    c_math_video = brain.add_content("C_VID", "Math Video", "Math", 1, "video")
    c_math_quiz = brain.add_content("C_QUIZ", "Math Quiz", "Math", 5, "quiz")
    
    print("\n[CONTEXT DEBUG]")
    ctx_beg = build_context(s_beginner, c_math_video)
    ctx_adv = build_context(s_advanced, c_math_video)
    
    print(f"{'Feature':<25} | {'Beginner':<10} | {'Advanced':<10}")
    print("-" * 50)
    features = ["Perf Score", "Sessions", "Difficulty", "Topic Match", "Video", "Quiz", "Exercise", "Reading", "Grade Trend", "P*D", "P*Vid", "P*Quiz"]
    for i, feat in enumerate(features):
        print(f"{feat:<25} | {ctx_beg[i]:<10.4f} | {ctx_adv[i]:<10.4f}")
        
    from scipy.spatial.distance import cosine
    dist = cosine(ctx_beg, ctx_adv)
    print(f"\nCosine Distance between Beginner & Advanced: {dist:.6f}")

    print("\n[REWARD DEBUG - First 10 Sessions]")
    for i in range(10):
        # Simulate a session
        before = s_beginner.performance_score
        # Simulate some variance in after_score
        improvement = np.random.normal(0.1, 0.05)
        after = np.clip(before + improvement, 0.0, 1.0)
        completed = i % 2 == 0
        
        reward = calculate_reward(
            before_score=before,
            after_score=after,
            completed=completed,
            time_spent_ratio=1.0,
            engaged=True,
            churned=False
        )
        print(f"Session {i+1}: Before={before:.2f}, After={after:.2f}, Comp={completed}, Reward={reward:.4f}")
        brain.update("S_BEG", "C_VID", reward)

if __name__ == "__main__":
    debug_diagnostics()
