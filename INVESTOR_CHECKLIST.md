# Investor Demo Checklist

## Prerequisites

```bash
python3 --version          # 3.12+
pip3 list | grep -iE "numpy|pandas|scikit|scipy|fastapi|uvicorn|streamlit|matplotlib|plotly"
docker --version           # optional
```

---

## Step 1 — Unit Tests

```bash
cd "/home/jazzman/Documents/trae_projects/Contextual Band brain model"
PYTHONPATH=. python3 -m pytest tests/ -v
```

Expected: **24 passed** in ~30s

---

## Step 2 — Investor Demo

```bash
cd "/home/jazzman/Documents/trae_projects/Contextual Band brain model"
PYTHONPATH=. python3 demo_investor.py
```

What happens:
1. Brain initializes with Hybrid LinUCB (α=1.5, 3 clusters)
2. 8 students & 8 content items registered (4 topics)
3. Cold-start recommendations (before training) — shows topic match
4. 2,000 training interactions — model learns in real-time
5. Post-training recommendations — shows improvement
6. Multi-objective reward examples
7. Neural Score self-diagnostics
8. Save/load persistence test

Expected (6–8s):
```
Step   500:  topic match = ~70%
Step  1000:  topic match = ~78%
Step  1500:  topic match = ~83%
Step  2000:  topic match = ~88%
Neural Score: ~7.0/10
```

---

## Step 3 — REST API Demo

Terminal 1:

```bash
cd "/home/jazzman/Documents/trae_projects/Contextual Band brain model"
PYTHONPATH=. uvicorn linucb_brain.api.app:app --host 0.0.0.0 --port 8000
```

Terminal 2:

```bash
# Add students
curl -X POST http://localhost:8000/students \
  -H "Content-Type: application/json" \
  -d '{"student_id":"S001","name":"Alice","performance_score":0.72,"current_topic":"Math"}'

curl -X POST http://localhost:8000/students \
  -H "Content-Type: application/json" \
  -d '{"student_id":"S002","name":"Bob","performance_score":0.45,"current_topic":"Science"}'

# Add content
curl -X POST http://localhost:8000/content \
  -H "Content-Type: application/json" \
  -d '{"content_id":"C001","title":"Algebra","topic":"Math","difficulty":2,"content_type":"video"}'

curl -X POST http://localhost:8000/content \
  -H "Content-Type: application/json" \
  -d '{"content_id":"C002","title":"Biology 101","topic":"Science","difficulty":3,"content_type":"reading"}'

# Recommend
curl http://localhost:8000/recommend/S001

# Submit feedback
curl -X POST http://localhost:8000/update \
  -H "Content-Type: application/json" \
  -d '{"student_id":"S001","content_id":"C001","reward":0.85}'

# Model health
curl http://localhost:8000/summary | python3 -m json.tool
curl http://localhost:8000/neural-score | python3 -m json.tool
```

---

## Step 4 — Streamlit Dashboard

```bash
cd "/home/jazzman/Documents/trae_projects/Contextual Band brain model"
PYTHONPATH=. streamlit run teacher_portal.py
```

Opens at `http://localhost:8501`

---

## Step 5 — Docker

```bash
cd "/home/jazzman/Documents/trae_projects/Contextual Band brain model"
docker compose up -d
```

---

## Step 6 — OULAD Large-Scale Simulation (optional)

```bash
cd "/home/jazzman/Documents/trae_projects/Contextual Band brain model"
PYTHONPATH=. python3 oulad_brain_run.py
```

Trains on 32,593 students across 10M+ interactions. Checkpoints saved every 100k.

---

## Step 7 — Load Trained Model

```bash
cd "/home/jazzman/Documents/trae_projects/Contextual Band brain model"
PYTHONPATH=. python3 -c "
from linucb_brain import Brain
brain = Brain.load('oulad_brain_state.json')
s = brain.summary()
print(f'Students: {s[\"student_count\"]}')
print(f'Sessions: {s[\"session_count\"]}')
print(f'Content:  {s[\"content_count\"]}')
print(f'Alpha:    {s[\"alpha\"]}')
"
```

---

## Key Metrics

| Metric | Demo Value |
|--------|-----------|
| Speed | ~250-300 interactions/sec |
| Topic match (after training) | ~87% |
| Neural Score | ~7.1/10 |
| Context sensitivity | ~8.6/10 (GREEN) |
| Tests | 24/24 pass |
| Cold-start | Works from first student |
