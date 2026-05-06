import numpy as np
import sys
import os

# Add parent directory to path so we can import brain_model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain_model import Personalizer

def simulate_learning(n_steps=1000):
    """
    Simulate a learner choosing between 3 types of content:
    - Video (arm 0)
    - Quiz (arm 1)
    - Reading (arm 2)
    
    The learner has 2 context features:
    - prior_score (0 to 1)
    - engagement_level (0 to 1)
    """
    n_arms = 3
    n_features = 2
    arm_names = ["Video", "Quiz", "Reading"]
    
    # Define "true" weights for each content type
    # For example, videos are good for low engagement, quizzes for high prior scores.
    # arm 0 (Video):   r = 0.5 * prior + 0.8 * engagement
    # arm 1 (Quiz):    r = 0.9 * prior + 0.2 * engagement
    # arm 2 (Reading): r = 0.3 * prior + 0.5 * engagement
    true_weights = np.array([
        [0.5, 0.8], # Video
        [0.9, 0.2], # Quiz
        [0.3, 0.5]  # Reading
    ])
    
    personalizer = Personalizer(n_arms=n_arms, n_features=n_features, arm_names=arm_names, alpha=1.0)
    
    cumulative_reward = 0
    rewards_history = []
    
    print(f"Starting simulation for {n_steps} steps...")
    
    for i in range(n_steps):
        # Generate random learner context
        context = np.random.rand(n_features)
        
        # Predict the best content
        recommended_arm = personalizer.predict(context)
        
        # Simulate reward with some noise
        arm_idx = personalizer.name_to_idx[recommended_arm]
        true_reward = np.dot(true_weights[arm_idx], context)
        observed_reward = true_reward + np.random.normal(0, 0.1)
        observed_reward = np.clip(observed_reward, 0, 1) # Ensure reward is between 0 and 1
        
        # Update model
        personalizer.update(recommended_arm, context, observed_reward)
        
        # Track progress
        cumulative_reward += observed_reward
        rewards_history.append(observed_reward)
        
        if (i + 1) % 100 == 0:
            print(f"Step {i+1}: Average Reward = {cumulative_reward / (i+1):.4f}")
            
    print("-" * 30)
    print(f"Simulation finished. Final Average Reward: {cumulative_reward / n_steps:.4f}")
    
    # Verification: Check if the model learned that quizzes are best for high prior scores
    # Example context: high prior (0.9), low engagement (0.1)
    # Video score: 0.5*0.9 + 0.8*0.1 = 0.53
    # Quiz score: 0.9*0.9 + 0.2*0.1 = 0.83
    # Reading score: 0.3*0.9 + 0.5*0.1 = 0.32
    # The model should choose "Quiz".
    test_context = [0.9, 0.1]
    best_recommendation = personalizer.predict(test_context)
    print(f"Verification: For context {test_context}, model recommends '{best_recommendation}'")

if __name__ == "__main__":
    simulate_learning()
