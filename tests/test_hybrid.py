import pytest
import numpy as np
from linucb_brain import Brain, Student, Content

def test_hybrid_brain_initialization():
    brain = Brain(model_type="hybrid")
    assert brain.model_type == "hybrid"
    assert hasattr(brain.model, 'A0')
    assert brain.model.k == 8
    assert brain.model.d == 9

def test_hybrid_recommend_and_update():
    brain = Brain(model_type="hybrid", alpha=1.0)
    brain.add_student("S1", "Alice")
    brain.add_content("C1", "Math Video", "Math", 3, "video")
    brain.add_content("C2", "Math Quiz", "Math", 5, "quiz")
    
    # Recommend
    rec = brain.recommend("S1", topic="Math")
    assert rec.content_id in ["C1", "C2"]
    
    # Update
    brain.update("S1", rec.content_id, 1.0)
    assert len(brain.sessions) == 1
    assert brain.model.arms[rec.content_id]['b'].sum() != 0

def test_hybrid_save_load(tmp_path):
    brain = Brain(model_type="hybrid")
    brain.add_student("S1", "Alice")
    brain.add_content("C1", "Math Video", "Math", 3, "video")
    brain.update("S1", "C1", 0.9)
    
    save_path = tmp_path / "hybrid_brain.json"
    brain.save(save_path)
    
    loaded = Brain.load(save_path)
    assert loaded.model_type == "hybrid"
    assert np.allclose(loaded.model.A0, brain.model.A0)
    assert len(loaded.model.arms) == 1
    assert "C1" in loaded.model.arms
