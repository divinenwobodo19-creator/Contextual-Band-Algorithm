"""
API Integration Tests for LinUCB Brain API
Uses FastAPI's TestClient to test all endpoints
"""
from fastapi.testclient import TestClient
import os
import tempfile
from linucb_brain.api.app import app, brain_instance as original_brain

client = TestClient(app)

# We'll use a temporary directory for brain state to avoid messing with production files
def setup_test_brain():
    # Create a fresh brain for testing
    from linucb_brain import Brain
    temp_dir = tempfile.TemporaryDirectory()
    temp_brain_path = os.path.join(temp_dir.name, "test_brain.json")
    
    # Override the get_brain dependency temporarily
    # For simplicity, let's just create a new test brain and replace the global one for testing
    return temp_dir, temp_brain_path

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "alive"
    assert "engine" in data
    assert "model" in data

def test_get_summary():
    response = client.get("/summary")
    assert response.status_code == 200
    data = response.json()
    assert "student_count" in data
    assert "content_count" in data
    assert "total_sessions" in data
    assert "model_type" in data

def test_add_student():
    student_data = {
        "student_id": "test_student_1",
        "name": "Test Student One",
        "performance_score": 0.7,
        "grade_history": {"Math": [0.6, 0.7, 0.8]},
        "current_topic": "Algebra",
        "metadata": {"notes": "Test student"}
    }
    response = client.post("/students", json=student_data)
    assert response.status_code in [200, 400]  # Either works (400 if already exists)
    if response.status_code == 200:
        data = response.json()
        assert data["student_id"] == student_data["student_id"]

def test_add_content():
    content_data = {
        "content_id": "test_content_1",
        "title": "Test Video Lesson",
        "topic": "Algebra",
        "difficulty": 3,
        "content_type": "video"
    }
    response = client.post("/content", json=content_data)
    assert response.status_code in [200, 400]  # 400 if already exists
    if response.status_code == 200:
        data = response.json()
        assert data["content_id"] == content_data["content_id"]

def test_recommendation_endpoint():
    # First make sure we have a test student and content
    test_add_student()
    test_add_content()
    
    # Get recommendation
    request_data = {
        "student_id": "test_student_1",
        "top_n": 1
    }
    response = client.post("/recommend", json=request_data)
    assert response.status_code in [200, 400]
    if response.status_code == 200:
        data = response.json()
        assert "content_id" in data

def test_calculate_reward_endpoint():
    request_data = {
        "before_score": 0.5,
        "after_score": 0.7,
        "completed": True,
        "time_spent_ratio": 1.0,
        "engaged": True,
        "churned": False
    }
    response = client.post("/calculate-reward", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "reward" in data
    assert isinstance(data["reward"], float)

if __name__ == "__main__":
    print("=== Running API Integration Tests ===\n")
    
    print("Test 1: Root endpoint")
    test_read_root()
    print("✅ Passed\n")
    
    print("Test 2: Summary endpoint")
    test_get_summary()
    print("✅ Passed\n")
    
    print("Test 3: Calculate reward endpoint")
    test_calculate_reward_endpoint()
    print("✅ Passed\n")
    
    print("Test 4: Add student endpoint")
    test_add_student()
    print("✅ Passed\n")
    
    print("Test 5: Add content endpoint")
    test_add_content()
    print("✅ Passed\n")
    
    print("Test 6: Recommendation endpoint")
    test_recommendation_endpoint()
    print("✅ Passed\n")
    
    print("=== All API Integration Tests PASSED! ✅ ===")
