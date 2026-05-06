import pytest
import os
import numpy as np
from linucb_brain import Brain, Student, Content

def test_add_student_and_add_content():
    brain = Brain()
    student = brain.add_student("S1", "Alice")
    assert student.student_id == "S1"
    assert student.name == "Alice"
    assert len(brain.students) == 1
    
    content = brain.add_content("C1", "Math Video", "Math", 3, "video")
    assert content.content_id == "C1"
    assert content.title == "Math Video"
    assert len(brain.contents) == 1

def test_recommend_returns_valid_content():
    brain = Brain()
    brain.add_student("S1", "Alice")
    brain.add_content("C1", "Math Video", "Math", 3, "video")
    brain.add_content("C2", "History Reading", "History", 2, "reading")
    
    # Recommend from all
    rec = brain.recommend("S1")
    assert isinstance(rec, Content)
    assert rec.content_id in ["C1", "C2"]
    
    # Recommend from topic
    rec_math = brain.recommend("S1", topic="Math")
    assert rec_math.content_id == "C1"

def test_update_increments_session_count():
    brain = Brain()
    brain.add_student("S1", "Alice")
    brain.add_content("C1", "Math Video", "Math", 3, "video")
    
    initial_sessions = len(brain.sessions)
    initial_student_sessions = brain.students["S1"].session_count
    
    # Standard flow: recommend then update
    brain.recommend("S1")
    brain.update("S1", "C1", 0.8)
    
    assert len(brain.sessions) == initial_sessions + 1
    assert brain.students["S1"].session_count == initial_student_sessions + 1
    assert brain.contents["C1"].times_recommended == 1

def test_save_and_load_preserves_state(tmp_path):
    brain = Brain()
    brain.add_student("S1", "Alice")
    brain.add_content("C1", "Math Video", "Math", 3, "video")
    brain.update("S1", "C1", 0.9)
    
    save_path = tmp_path / "test_brain.json"
    brain.save(save_path)
    
    loaded_brain = Brain.load(save_path)
    
    assert loaded_brain.alpha == brain.alpha
    assert len(loaded_brain.students) == 1
    assert len(loaded_brain.contents) == 1
    assert len(loaded_brain.sessions) == 1
    assert loaded_brain.students["S1"].name == "Alice"
    assert loaded_brain.contents["C1"].title == "Math Video"
    assert np.allclose(loaded_brain.model.arms["C1"]['A'], brain.model.arms["C1"]['A'])
