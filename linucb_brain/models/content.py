from dataclasses import dataclass

@dataclass
class Content:
    """Learning content/arm data model."""
    content_id: str
    title: str
    topic: str
    difficulty: int # 1–5
    content_type: str # "video", "quiz", "exercise", "reading"
    times_recommended: int = 0
    times_rewarded: int = 0
    avg_reward: float = 0.0 # running average reward when this content is used
