from fastapi import FastAPI, HTTPException, Depends
from typing import List, Dict, Optional
import os

from ..brain import Brain
from ..core.reward import calculate_reward
from .schemas import (
    StudentSchema, 
    ContentSchema, 
    RecommendationRequest, 
    UpdateRequest, 
    RewardRequest, 
    NeuralScoreResponse, 
    BrainSummary
)

app = FastAPI(title="LinUCB Brain API", version="0.1.0")

import glob

# Global brain instance
BRAIN_STATE_PATH = os.getenv("BRAIN_STATE_PATH", "oulad_brain_state.json")

def get_brain():
    # Check for full brain state first
    if os.path.exists(BRAIN_STATE_PATH):
        return Brain.load(BRAIN_STATE_PATH)
    
    # Otherwise check for checkpoints (just like dashboard)
    checkpoints = glob.glob("oulad_checkpoint_*.json")
    if checkpoints:
        # Sort by modification time (most recent first)
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        latest = checkpoints[0]
        print(f"Loading Brain from checkpoint: {latest}")
        return Brain.load(latest)
        
    print("Initializing new Hybrid Brain (No state found)")
    return Brain(model_type="hybrid")

brain_instance = get_brain()

@app.get("/")
def read_root():
    return {"status": "alive", "engine": "LinUCB Brain", "model": brain_instance.model_type}

@app.post("/students", response_model=StudentSchema)
def add_student(student: StudentSchema):
    try:
        brain_instance.add_student(
            student.student_id, 
            student.name, 
            grade_history=student.grade_history,
            performance_score=student.performance_score,
            current_topic=student.current_topic,
            metadata=student.metadata
        )
        return student
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/content", response_model=ContentSchema)
def add_content(content: ContentSchema):
    try:
        brain_instance.add_content(
            content.content_id, 
            content.title, 
            content.topic, 
            content.difficulty, 
            content.content_type
        )
        return content
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/recommend")
def recommend(request: RecommendationRequest):
    try:
        recommendations = brain_instance.recommend(
            request.student_id, 
            topic=request.topic, 
            top_n=request.top_n
        )
        if request.top_n == 1:
            return recommendations.__dict__ if recommendations else {}
        return [r.__dict__ for r in recommendations]
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/update")
def update(request: UpdateRequest):
    try:
        brain_instance.update(request.student_id, request.content_id, request.reward)
        return {"status": "success"}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/calculate-reward")
def get_reward(request: RewardRequest):
    reward = calculate_reward(
        request.before_score,
        request.after_score,
        request.completed,
        request.time_spent_ratio,
        request.engaged,
        request.churned
    )
    return {"reward": reward}

@app.get("/summary", response_model=BrainSummary)
def get_summary():
    score = brain_instance.last_neural_score
    return {
        "student_count": len(brain_instance.students),
        "content_count": len(brain_instance.contents),
        "total_sessions": brain_instance.update_count,
        "model_type": brain_instance.model_type,
        "current_alpha": brain_instance.alpha,
        "current_gamma": brain_instance.gamma,
        "cumulative_regret": brain_instance.cumulative_regret,
        "last_neural_score": score.get("neural_score") if score else 0.0
    }

@app.post("/tune")
def tune_parameters():
    new_params = brain_instance.tune_parameters()
    return {"status": "optimized", "params": new_params}

@app.post("/save")
def save_brain_state():
    brain_instance.save(BRAIN_STATE_PATH)
    return {"status": "saved", "path": BRAIN_STATE_PATH}
