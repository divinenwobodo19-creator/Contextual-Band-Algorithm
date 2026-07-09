# Contextual Bandit Brain

A self-learning personalization engine for education, powered by Contextual Bandit algorithms. Trained on **10.6 million** real student interactions from the Open University.

## What It Does

Every student is different. Most LMS platforms serve the same content to everyone. The Brain learns **what works for whom** — adapting in real-time as each student interacts with the platform.

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│ Student      │────>│ Brain recommends │────>│ Student      │
│ (context)    │     │ (explore/exploit)│     │ engages with │
└─────────────┘     └──────────────────┘     │ content      │
                                             └──────┬───────┘
                                                    │
┌─────────────┐     ┌──────────────────┐            │
│ Brain       │<────│ Reward signal    │<───────────┘
│ updates     │     │ (score, complete)│
│ matrices    │     └──────────────────┘
└─────────────┘
```

## Key Features

| Capability | What It Means |
|---|---|
| **3 Algorithms** | Disjoint LinUCB, Hybrid LinUCB (+ clustering), Linear Thompson Sampling |
| **Real-time Learning** | Updates after every interaction — no batch retraining |
| **Cold-start Ready** | Explores when it has no data; exploits when it does |
| **Self-Diagnostics** | Neural Score measures 7 dimensions of model health |
| **Online Clustering** | Groups similar students to share knowledge (COBART-style) |
| **FastAPI + Docker** | Production-ready REST API, one-command deploy |

## Neural Score — Built-in Diagnostics

The Brain scores itself on 7 dimensions (each 0–10):

```
Exploration Efficiency  →  8.3/10  (Are we trying diverse content?)
Reward Convergence      →  7.1/10  (Is average reward improving?)
Context Sensitivity     →  6.8/10  (Do different students get different recs?)
Recommendation Precision→  7.5/10  (Are recommended items getting good rewards?)
Grade Prediction        →  6.2/10  (Can we forecast student performance?)
Cohort Purity           →  7.0/10  (Are student clusters well-separated?)
Objective Balance       →  7.5/10  (Are all reward signals being used?)
─────────────────────────────────
NEURAL SCORE            →  7.2/10
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the investor demo (8 students, 8 content items, 500 interactions)
PYTHONPATH=. python3 demo_investor.py

# 3. Start the API (auto-seeds demo data if no checkpoint found)
PYTHONPATH=. uvicorn linucb_brain.api.app:app --host 0.0.0.0 --port 8000

# 4. Get a recommendation
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"student_id": "608041", "top_n": 3}'

# 5. Open the dashboard
streamlit run dashboard.py
```

> **Windows?** Use `set PYTHONPATH=. && uvicorn ...` (CMD) or `$env:PYTHONPATH='.'; uvicorn ...` (PowerShell) instead of `PYTHONPATH=.`.

## Full Training (OULAD Dataset)

To reproduce the 10.6M-interaction training:

```bash
# 1. Download OULAD from Kaggle into data/oulad/
# 2. Preprocess
PYTHONPATH=. python3 oulad_preprocessor.py
# 3. Train (may take hours)
PYTHONPATH=. python3 oulad_brain_run.py
```

## Project Structure

```
├── linucb_brain/       # Core engine
│   ├── core/           # Algorithms (linucb, hybrid, thompson, clustering)
│   ├── models/         # Data models (student, content, session)
│   ├── diagnostics/    # Neural Score engine
│   └── api/            # FastAPI application
├── examples/           # Integration examples
├── tests/              # Test suite (pytest)
├── figures/            # Training visualizations
├── dashboard.py        # Streamlit real-time dashboard
├── demo_investor.py    # Guided walkthrough demo
└── docker-compose.yml  # One-command deploy
```

## Docker

```bash
docker-compose up --build
```

## Results (10.6M Interactions)

- **Exploration Efficiency:** 9.4/10
- **Average Reward:** 0.70–0.90
- **Students:** 32,000+
- **Content Items:** 6,000+

## License

MIT
