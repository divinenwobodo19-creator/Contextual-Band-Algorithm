# LinUCB Brain

A Contextual Bandit-powered "brain model" for learning personalization, designed to plug into any LMS, grade tracker, or learning personalization system.

## Features
- **Contextual Personalization**: Supports both **Disjoint** and **Hybrid** LinUCB algorithms.
- **Hybrid Learning**: Shared knowledge across arms for faster convergence in large arm scenarios.
- **Neural Score Engine**: A built-in diagnostic tool to measure exploration efficiency, reward convergence, context sensitivity, and prediction precision.
- **Modular Design**: Separates data models (Student, Content, Session) from core logic.
- **Efficient Updates**: Uses the Sherman-Morrison rank-1 update formula for $O(d^2)$ matrix inversion updates.
- **JSON Storage**: Easy to save and load brain states for persistence.

## Installation
```bash
pip install .
```

## Quick Start
```python
from linucb_brain import Brain

# Initialize (Disjoint or Hybrid)
brain = Brain(alpha=1.0, model_type="hybrid")

# Add Students
brain.add_student("S1", "Alice")

# Add Content
brain.add_content("C1", "Math Video", "Math", 3, "video")
brain.add_content("C2", "Quiz", "Math", 5, "quiz")

# Get Recommendation
recommendation = brain.recommend("S1", topic="Math")

# Update with Reward
# reward = calculate_reward(...) or a score between -1 and 1
brain.update("S1", "C1", 0.8)

# Run Diagnostics
brain.neural_score()
```

## Project Structure
- `linucb_brain/`: Core library code.
- `examples/`: Walkthroughs for LMS and Grade Prediction.
- `tests/`: Comprehensive test suite.
