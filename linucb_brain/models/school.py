from dataclasses import dataclass, field
from typing import Dict, Optional


GRADE_LEVELS = ["JSS1", "JSS2", "JSS3", "SSS1", "SSS2", "SSS3"]


def normalize_label(raw: str) -> str:
    """Canonical form for human-facing labels: trim, collapse spaces, uppercase."""
    if not raw:
        return ""
    return " ".join(raw.split()).upper()


@dataclass
class School:
    """An educational institution (tenant). Owns classes and students."""
    name: str
    school_id: str = ""
    created_at: str = ""
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.school_id:
            import uuid
            self.school_id = str(uuid.uuid4())[:8]
        if not self.created_at:
            from datetime import datetime
            self.created_at = datetime.now().isoformat()
        self.name = normalize_label(self.name)


@dataclass
class SchoolClass:
    """A class arm within a school. Label is local to the school."""
    label: str
    school_id: str
    class_id: str = ""
    grade_level: str = ""
    created_at: str = ""
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.class_id:
            import uuid
            self.class_id = str(uuid.uuid4())[:8]
        if not self.created_at:
            from datetime import datetime
            self.created_at = datetime.now().isoformat()
        self.label = normalize_label(self.label)

    @staticmethod
    def split_label(label: str) -> tuple:
        """Split 'JSS1A' into (grade_level, arm). 'Gold' -> ('', 'Gold')."""
        label = normalize_label(label)
        for lvl in GRADE_LEVELS:
            if label.startswith(lvl):
                return lvl, label[len(lvl):]
        return "", label