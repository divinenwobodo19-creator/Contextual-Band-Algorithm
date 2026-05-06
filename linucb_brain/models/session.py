from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

@dataclass
class Session:
    """Session tracker per student."""
    session_id: str
    student_id: str
    content_id: str
    context_vector: List[float]
    reward: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    topic: str = ""
