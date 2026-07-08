
# Contextual Bandit Brain Model (OULAD Trained)
A Contextual Bandit-powered "brain model" for learning personalization, trained on the full Open University Learning Analytics Dataset (OULAD).

## Project Highlights

- 100% Trained!: Processed 10.6M+ interactions from the OULAD dataset!
- Hybrid LinUCB Algorithm: For better personalization and faster convergence!
- Live API & Dashboard: Ready for production use!
- Neural Score Diagnostics: Built-in performance metrics!

## Features

1. Contextual Personalization: Both Disjoint and Hybrid LinUCB algorithm support!
2. Hybrid Learning: Shared knowledge across arms/students!
3. Neural Score Engine: Diagnostics for exploration efficiency, context sensitivity, and more!
4. OULAD Integration: Pre-trained on the full OULAD dataset!
5. FastAPI Backend: Production-ready API!
6. Streamlit Dashboard: Live monitoring and analytics!

## Dataset
This project uses the Open University Learning Analytics Dataset (OULAD) - 10.6M+ interactions from 32k+ students and 6k+ learning materials!

## Quick Start

### Run the API
```bash
PYTHONPATH=. uvicorn linucb_brain.api.app:app --host 0.0.0.0 --port 8000
```

### Run the Dashboard
```bash
streamlit run dashboard.py
```

### Run the Simulation
```bash
PYTHONPATH=. python3 oulad_brain_run.py
```

### Get a Recommendation (API)
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"student_id": "608041", "top_n": 3}'
```

## Project Structure
```
├── linucb_brain/       # Core library code
├── examples/           # Example scripts
├── tests/              # Test suite
├── data/oulad/         # OULAD dataset
├── dashboard.py        # Streamlit dashboard
├── oulad_brain_run.py  # OULAD simulation
└── monitor_training.py # Training monitor
```

## Results
- Trained on: 10,657,981 interactions
- Exploration Efficiency: 9.4/10
- Average Reward: 0.70-0.90 for content recommendations
- Context Sensitivity Improvements: Boosted feature scaling by 5-10x!
- A/B Test Winner: Hybrid Model, 1.4% better than Disjoint!

## Model Parameters
The `Brain` class has several configurable parameters to tune performance:

### Core Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float | 1.0 | Exploration-exploitation tradeoff. Higher values → more exploration. |
| `alpha_decay` | float | 0.999 | Multiplicative decay for alpha over time. |
| `model_type` | str | "disjoint" | Algorithm to use: "disjoint", "hybrid", or "ts". Use `algorithm` as an alias. |
| `algorithm` | str | None | Alias for `model_type`. |
| `gamma` | float | 1.0 | Discount factor for LinUCB updates. |
| `n_clusters` | int | 5 | Number of student clusters for COBART hybrid model. |
| `auto_diagnose_every` | int | 50 | Frequency of automatic neural score diagnostics (in thousands of interactions). |
| `track_sessions` | bool | True | Whether to store session history in memory. |
| `max_sessions` | int | 100000 | Maximum number of sessions to keep in memory (prevents memory issues). |

### Multi-Objective Reward Weights
You can tune `brain.reward_weights` to prioritize different goals:
```python
brain.reward_weights = {
    'improvement': 0.5,   # Priority on score improvement
    'completion': 0.3,    # Priority on content completion
    'engagement': 0.2     # Priority on student engagement
}
```

## Docker Support
```bash
docker-compose up --build
```

## License
MIT
