import pytest
import numpy as np
from linucb_brain.core.linucb import LinUCBDisjoint

def test_arm_selection_returns_valid_arm_id():
    n_features = 3
    model = LinUCBDisjoint(n_features=n_features, alpha=1.0)
    arm_ids = ["arm1", "arm2", "arm3"]
    context = np.random.rand(n_features)
    
    selected_arm = model.select(arm_ids, context)
    assert selected_arm in arm_ids

def test_update_modifies_A_and_b_matrices_correctly():
    n_features = 2
    model = LinUCBDisjoint(n_features=n_features, alpha=1.0)
    arm_id = "arm1"
    context = np.array([1.0, 0.2])
    reward = 1.0
    
    # Before update
    model._init_arm(arm_id)
    A_before = model.arms[arm_id]['A'].copy()
    b_before = model.arms[arm_id]['b'].copy()
    
    model.update(arm_id, context, reward)
    
    # After update
    A_after = model.arms[arm_id]['A']
    A_inv_after = model.arms[arm_id]['A_inv']
    b_after = model.arms[arm_id]['b']
    
    # A += context @ context.T
    expected_A = A_before + np.outer(context, context)
    assert np.allclose(A_after, expected_A)
    
    # Check A_inv is correct
    assert np.allclose(A_inv_after, np.linalg.inv(A_after))
    
    # b += reward * context
    expected_b = b_before + reward * context.reshape(-1, 1)
    assert np.allclose(b_after, expected_b)

def test_alpha_zero_always_picks_highest_mean_reward_arm():
    # With alpha=0, it's pure exploitation
    n_features = 1
    model = LinUCBDisjoint(n_features=n_features, alpha=0.0)
    arm_ids = ["arm1", "arm2"]
    context = np.array([1.0])
    
    # Arm 1 has higher reward history
    model.update("arm1", context, 1.0)
    model.update("arm1", context, 1.0)
    
    # Arm 2 has lower reward history
    model.update("arm2", context, 0.1)
    
    # Select many times
    for _ in range(10):
        selected = model.select(arm_ids, context)
        assert selected == "arm1"
