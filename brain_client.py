import requests
from typing import List, Dict, Optional, Any

class BrainClient:
    """
    A lightweight SDK client for interacting with the LinUCB Brain API.
    """
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def add_student(self, student_id: str, name: str, **kwargs) -> Dict:
        """Register a new student in the brain."""
        payload = {
            "student_id": student_id,
            "name": name,
            **kwargs
        }
        response = requests.post(f"{self.base_url}/students", json=payload)
        response.raise_for_status()
        return response.json()

    def add_content(self, content_id: str, title: str, topic: str, difficulty: int, content_type: str) -> Dict:
        """Register a new content item in the brain."""
        payload = {
            "content_id": content_id,
            "title": title,
            "topic": topic,
            "difficulty": difficulty,
            "content_type": content_type
        }
        response = requests.post(f"{self.base_url}/content", json=payload)
        response.raise_for_status()
        return response.json()

    def recommend(self, student_id: str, topic: Optional[str] = None, top_n: int = 1) -> List[Dict]:
        """Get content recommendations for a student."""
        payload = {
            "student_id": student_id,
            "topic": topic,
            "top_n": top_n
        }
        response = requests.post(f"{self.base_url}/recommend", json=payload)
        response.raise_for_status()
        res = response.json()
        return [res] if isinstance(res, dict) else res

    def update(self, student_id: str, content_id: str, reward: float) -> Dict:
        """Provide feedback (reward) to the brain after a recommendation."""
        payload = {
            "student_id": student_id,
            "content_id": content_id,
            "reward": reward
        }
        response = requests.post(f"{self.base_url}/update", json=payload)
        response.raise_for_status()
        return response.json()

    def get_diagnostics(self) -> Dict:
        """Get the latest Neural Score and diagnostics."""
        response = requests.get(f"{self.base_url}/diagnostics")
        response.raise_for_status()
        return response.json()

    def get_summary(self) -> Dict:
        """Get a summary of the brain's current state."""
        response = requests.get(f"{self.base_url}/summary")
        response.raise_for_status()
        return response.json()

# Example Usage
if __name__ == "__main__":
    client = BrainClient()
    try:
        print("Brain Status:", client.get_summary())
    except Exception as e:
        print("Could not connect to Brain API. Is it running?")
