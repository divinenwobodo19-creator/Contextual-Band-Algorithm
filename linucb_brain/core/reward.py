import numpy as np

def calculate_reward(
    before_score: float, 
    after_score: float, 
    completed: bool, 
    time_spent_ratio: float,
    engaged: bool = True,
    churned: bool = False
) -> float:
    """
    Calculate an advanced reward signal focusing on continuous improvement delta.
    
    Args:
        before_score: Score before session (0-1).
        after_score: Score after session (0-1).
        completed: Whether content was finished.
        time_spent_ratio: actual / expected time.
        engaged: Qualitative engagement signal.
        churned: CRITICAL - If student left the platform after this.
        
    Returns:
        float: Reward in [-1.0, 1.0].
    """
    if churned:
        return -1.0
        
    # Switch to continuous signal based on performance delta
    # Scaling factor to make delta meaningful (e.g., 0.1 improvement -> 0.5 reward)
    delta = after_score - before_score
    delta_scaled = delta * 5.0 
    
    completion_bonus = 0.2 if completed else -0.1
    engagement_bonus = 0.1 if engaged else -0.2
    
    # Time penalty for extreme slowness
    time_penalty = -0.2 if time_spent_ratio > 2.0 else 0.0
    
    reward = delta_scaled + completion_bonus + engagement_bonus + time_penalty
    
    # Ensure raw values are being passed meaningfully
    # print(f"DEBUG REWARD: delta={delta:.4f}, total={reward:.4f}")
    
    return float(np.clip(reward, -1.0, 1.0))
