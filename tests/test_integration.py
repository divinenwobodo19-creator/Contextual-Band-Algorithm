"""
Integration test for the entire Brain pipeline:
1. Initialize Brain
2. Add students/agents
3. Add content/arms
4. Recommend
5. Update
6. Save
7. Load
8. Verify
"""
import os
import tempfile
from linucb_brain import Brain

def test_full_pipeline_integration():
    print("=== Starting full integration test ===\n")
    
    # 1. Initialize Brain
    brain = Brain(
        model_type="hybrid",
        alpha=2.0,
        alpha_decay=0.99,
        n_clusters=5
    )
    print("✅ Brain initialized successfully")
    
    # 2. Add students (using add_agent)
    students = [
        ("S1", {"performance_score": 0.7, "education_level": 2, "age_band": 3}),
        ("S2", {"performance_score": 0.4, "education_level": 1, "age_band": 2}),
        ("S3", {"performance_score": 0.9, "education_level": 3, "age_band": 4}),
    ]
    
    for sid, features in students:
        brain.add_agent(sid, features)
    assert len(brain.students) == 3
    print(f"✅ Added {len(brain.students)} students")
    
    # 3. Add content (using add_arm)
    contents = [
        ("C1", {"difficulty": 0.25, "activity_type": "reading"}),
        ("C2", {"difficulty": 0.5, "activity_type": "video"}),
        ("C3", {"difficulty": 0.75, "activity_type": "quiz"}),
        ("C4", {"difficulty": 0.5, "activity_type": "exercise"}),
    ]
    
    for cid, features in contents:
        brain.add_arm(cid, features)
    assert len(brain.contents) == 4
    print(f"✅ Added {len(brain.contents)} content items")
    
    # 4. Get recommendations
    rec_s1 = brain.recommend("S1")
    rec_s2 = brain.recommend("S2")
    rec_s3 = brain.recommend("S3", top_n=2)
    assert rec_s1 is not None
    assert len(rec_s3) == 2
    print(f"✅ Recommendations received for all students")
    
    # 5. Update with rewards
    update_pairs = [
        ("S1", rec_s1.content_id, 0.8),
        ("S2", rec_s2.content_id, 0.4),
        ("S3", rec_s3[0].content_id, 0.9),
    ]
    for sid, cid, reward in update_pairs:
        brain.update(agent_id=sid, arm_id=cid, reward=reward)
    assert brain.update_count == len(update_pairs)
    print(f"✅ Updated model with {brain.update_count} rewards")
    
    # 6. Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "brain_integration_test.json")
        brain.save(save_path)
        assert os.path.exists(save_path)
        print("✅ Brain saved successfully")
        
        # Load back
        loaded_brain = Brain.load(save_path)
        print("✅ Brain loaded successfully")
        
        # 7. Verify state preserved
        assert loaded_brain.alpha == brain.alpha
        assert loaded_brain.model_type == brain.model_type
        assert loaded_brain.n_clusters == brain.n_clusters
        assert len(loaded_brain.students) == len(brain.students)
        assert len(loaded_brain.contents) == len(brain.contents)
        assert len(loaded_brain.sessions) == len(brain.sessions)
        
        # 8. Verify we can still get recommendations
        loaded_rec = loaded_brain.recommend("S1")
        assert loaded_rec is not None
        print("✅ Loaded brain still recommends content correctly")
    
    print("\n=== Integration test PASSED! ✅ ===")

if __name__ == "__main__":
    test_full_pipeline_integration()
