# Contextual Band Brain Model - Project Index

This file serves as a central hub for your project when opened as an **Obsidian Vault**. You can click the links below to navigate your codebase.

## Core Brain Modules
- [[linucb_brain/brain.py]] - Main entry point and interface
- [[linucb_brain/storage.py]] - Model persistence (Save/Load)
- [[linucb_brain/utils.py]] - Helper functions

## Algorithms (Core)
- [[linucb_brain/core/linucb.py]] - Disjoint LinUCB
- [[linucb_brain/core/linucb_hybrid.py]] - Hybrid LinUCB (with Cluster tracking)
- [[linucb_brain/core/lints.py]] - Linear Thompson Sampling
- [[linucb_brain/core/clustering.py]] - Dynamic Cohort Clustering
- [[linucb_brain/core/reward.py]] - Multi-objective Reward Logic
- [[linucb_brain/core/context.py]] - Feature Engineering & Vectors

## Models (Data Structures)
- [[linucb_brain/models/student.py]] - Student profiles
- [[linucb_brain/models/content.py]] - Content/Arm metadata
- [[linucb_brain/models/session.py]] - Interaction history

## API & Services
- [[linucb_brain/api/app.py]] - FastAPI application
- [[linucb_brain/api/schemas.py]] - Pydantic data models

## Diagnostics
- [[linucb_brain/diagnostics/neural_score.py]] - Model performance metrics
- [[linucb_brain/diagnostics/report.py]] - Detailed system reports

## Tests
- [[tests/test_brain.py]]
- [[tests/test_hybrid.py]]
- [[tests/test_linucb.py]]
- [[tests/test_neural_score.py]]
- [[tests/test_personalizer.py]]

## Execution & Simulation
- [[oulad_brain_run.py]] - Main OULAD dataset simulation
- [[oulad_preprocessor.py]] - Data cleaning pipeline
- [[dashboard.py]] - Real-time monitoring dashboard

---
