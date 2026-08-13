from dataclasses import dataclass, field
from typing import Dict, List

def normalize_label(raw: str) -> str:
    """Canonical form for human-facing labels: trim, collapse spaces, uppercase."""
    if not raw:
        return ""
    return " ".join(raw.split()).upper()


@dataclass
class Student:
    """Student profile data model."""
    student_id: str
    name: str
    class_id: str = ""  # SchoolClass.class_id (UUID) scoped to school; label moved to class record
    school_id: str = ""  # School.school_id (UUID) — the tenant scope
    grade_history: Dict[str, List[float]] = field(default_factory=dict) # e.g {"math": [0.6, 0.75, 0.8]}
    session_count: int = 0
    performance_score: float = 0.0 # rolling average
    current_topic: str = ""
    metadata: Dict = field(default_factory=dict)

    # Provenance
    created_at: str = ""
    updated_at: str = ""

    # OULAD specific features
    education_level: float = 0.0
    age_band: float = 0.0
    credits_studied: float = 0.0
    imd_band: float = 0.5
    region_code: float = 0.0

    def __post_init__(self):
        if not self.created_at:
            from datetime import datetime
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at
        self.name = " ".join(self.name.split()).strip()

    def touch(self):
        from datetime import datetime
        self.updated_at = datetime.now().isoformat()