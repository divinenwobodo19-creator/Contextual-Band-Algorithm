from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime

class StudentSchema(BaseModel):
    student_id: str
    name: str
    grade_history: Dict[str, List[float]] = {}
    performance_score: float = 0.5
    current_topic: str = ""
    metadata: Dict[str, Any] = {}

class ContentSchema(BaseModel):
    content_id: str
    title: str
    topic: str
    difficulty: int = Field(..., ge=1, le=5)
    content_type: str

class RecommendationRequest(BaseModel):
    student_id: str
    topic: Optional[str] = None
    top_n: int = 1

class UpdateRequest(BaseModel):
    student_id: str
    content_id: str
    reward: float = Field(..., ge=-1.0, le=1.0)

class RewardRequest(BaseModel):
    before_score: float
    after_score: float
    completed: bool
    time_spent_ratio: float = 1.0
    engaged: bool = True
    churned: bool = False

class NeuralScoreResponse(BaseModel):
    exploration_score: float
    convergence_score: float
    context_score: float
    precision_score: float
    grade_score: float
    purity_score: float = 5.0
    balance_score: float = 7.5
    neural_score: float

class BrainSummary(BaseModel):
    student_count: int
    content_count: int
    total_sessions: int
    model_type: str
    current_alpha: float
    current_gamma: float
    cumulative_regret: float
    last_neural_score: Optional[float]
