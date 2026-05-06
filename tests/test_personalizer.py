import pytest
import numpy as np
import os
from brain_model import Personalizer

def test_personalizer_initialization():
    p = Personalizer(n_arms=3, n_features=2, arm_names=["A", "B", "C"])
    assert p.n_arms == 3
    assert p.n_features == 2
    assert p.arm_names == ["A", "B", "C"]
    assert p.name_to_idx["A"] == 0

def test_personalizer_predict_and_update():
    p = Personalizer(n_arms=2, n_features=2)
    context = [0.5, 0.5]
    
    # Predict
    arm = p.predict(context)
    assert arm in ["arm_0", "arm_1"]
    
    # Update
    p.update(arm, context, reward=1.0)
    
    # After a high reward for one arm, it should be more likely to be picked for similar context
    # (though with alpha=1.0 exploration is still high)
    for _ in range(10):
        p.update("arm_0", [1.0, 0.0], reward=1.0)
        p.update("arm_1", [1.0, 0.0], reward=0.0)
        
    # Lower alpha for more deterministic choice in test
    p.model.alpha = 0.1
    recommended = p.predict([1.0, 0.0])
    assert recommended == "arm_0"

def test_save_load(tmp_path):
    p = Personalizer(n_arms=2, n_features=2, arm_names=["X", "Y"])
    p.update("X", [1.0, 1.0], 1.0)
    
    model_path = tmp_path / "model.pkl"
    p.save(model_path)
    
    p2 = Personalizer.load(model_path)
    assert p2.n_arms == 2
    assert p2.arm_names == ["X", "Y"]
    assert np.allclose(p.model.A[0], p2.model.A[0])
    assert np.allclose(p.model.b[0], p2.model.b[0])

def test_invalid_inputs():
    p = Personalizer(n_arms=2, n_features=2)
    
    with pytest.raises(ValueError):
        p.predict([1.0]) # Wrong feature size
        
    with pytest.raises(ValueError):
        p.update("arm_invalid", [1.0, 1.0], 1.0) # Invalid arm name
