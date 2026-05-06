from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class Student:
    """Student profile data model."""
    student_id: str
    name: str
    grade_history: Dict[str, List[float]] = field(default_factory=dict) # e.g {"math": [0.6, 0.75, 0.8]}
    session_count: int = 0
    performance_score: float = 0.0 # rolling average
    current_topic: str = ""
    metadata: Dict = field(default_factory=dict)
    
    # OULAD specific features
    education_level: float = 0.0
    age_band: float = 0.0
    credits_studied: float = 0.0
    imd_band: float = 0.5
    region_code: float = 0.0
