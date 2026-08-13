import numpy as np
from math import log1p
from typing import List, Tuple
from ..models.student import Student
from ..models.content import Content

def build_context(student: Student, content: Content) -> np.ndarray:
    """
    Construct a context vector from student and content features.
    """
    z, x = build_context_split(student, content)
    return np.concatenate([z, x])

# Pre-calculate constants for speed
LOG_101 = log1p(100)
CONTENT_TYPES = ["video", "quiz", "exercise", "reading"]

def build_context_split(student: Student, content: Content) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct shared and arm-specific context vectors.
    Optimized for performance in large-scale simulations.
    """
    # 1. Student Features (Shared)
    performance = getattr(student, 'performance_score', 0.0)
    # Using log1p(x) / log1p(100) assuming max_sessions=100
    sessions = log1p(getattr(student, 'session_count', 0)) / LOG_101
    
    edu_level = getattr(student, 'education_level', 0.0)
    age_band = getattr(student, 'age_band', 0.0)
    credits_studied = getattr(student, 'credits_studied', 0.0)
    imd_band = getattr(student, 'imd_band', 0.5)
    region_code = getattr(student, 'region_code', 0.0)
    
    # Grade trend slope (positive = improving, negative = declining)
    grade_history = getattr(student, 'grade_history', {})
    c_topic = content.topic
    history = grade_history.get(c_topic)
    
    if history and len(history) >= 2:
        recent = history[-5:]
        try:
            # Simple linear regression slope for small N
            n = len(recent)
            x_vals = np.arange(n)
            y_vals = np.array(recent)
            sum_x = n * (n - 1) / 2
            sum_y = np.sum(y_vals)
            sum_xx = n * (n - 1) * (2 * n - 1) / 6
            sum_xy = np.dot(x_vals, y_vals)
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x**2)
        except:
            slope = 0.0
    else:
        # Use initial performance as a proxy if no history exists for this topic
        slope = (performance - 0.5) * 0.1 # Slight bias based on overall perf
        
    z_shared = np.array([
        performance, 
        sessions, 
        slope,
        edu_level,
        age_band,
        credits_studied,
        imd_band,
        region_code
    ], dtype=float)
    
    # 2.Content Features (Arm-specific)
    difficulty = (content.difficulty - 1) / 4.0
    topic_match = 1.0 if student.current_topic == c_topic else 0.0
    
    c_type = content.content_type
    one_hot = [1.0 if c_type == ct else 0.0 for ct in CONTENT_TYPES]
    
    perf_diff = performance * difficulty
    perf_video = performance * one_hot[0]
    perf_quiz = performance * one_hot[1]
    
    x_arm = np.array([
        difficulty,
        topic_match,
        *one_hot,
        perf_diff,
        perf_video,
        perf_quiz
    ], dtype=float)
    
    return z_shared, x_arm

def get_context_dimension() -> int:
    """Return the fixed dimension of the context vector."""
    return 17 # 8 shared + 9 arm-specific

def get_hybrid_dimensions() -> Tuple[int, int]:
    """Return (k_shared, d_arm) dimensions."""
    return 8, 9 # z: perf, sess, slope, edu, age, credits, imd, region; x: diff, match, 4*type, 3*interact
