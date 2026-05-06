import time
import random
from brain_client import BrainClient

def simulate_lms_workflow():
    print("🎓 REAL-WORLD LMS INTEGRATION DEMO")
    print("="*50)
    
    # 1. Initialize the SDK Client
    # In a real setup, this URL would be your production server
    client = BrainClient(base_url="http://localhost:8000")
    
    try:
        # Check if API is alive
        summary = client.get_summary()
        print(f"✅ Connected to Brain API. Model: {summary['model_type']}")
        print(f"📊 Current Brain Knowledge: {summary['total_sessions']:,} interactions")
    except Exception as e:
        print("❌ Error: Brain API is not running. Please start it with 'uvicorn linucb_brain.api.app:app'")
        return

    # 2. Simulate a Student Logging In
    # We'll use one of the students from the OULAD dataset or a new one
    student_id = "ST_DEMO_001"
    print(f"\n👤 Student {student_id} logged into the LMS.")
    
    # Check if student exists, if not, add them
    try:
        client.add_student(
            student_id=student_id,
            name="John Doe",
            performance_score=0.65,
            current_topic="Mathematics",
            metadata={"education": "A Level", "age_band": "0-35"}
        )
        print("📝 Registered new student in the Brain.")
    except:
        print("ℹ️ Student already exists in the Brain.")

    # 3. Get Personalized Recommendations
    print(f"\n🔍 Requesting personalized content for {student_id} in topic 'Mathematics'...")
    try:
        recommendations = client.recommend(student_id=student_id, topic="Mathematics", top_n=3)
        
        print("\n--- RECOMMENDED CONTENT ---")
        for i, rec in enumerate(recommendations):
            print(f"{i+1}. [{rec['content_id']}] {rec['title']} (Diff: {rec['difficulty']}, Type: {rec['content_type']})")
    except Exception as e:
        print(f"⚠️ Could not get recommendations: {e}")
        return

    # 4. Simulate Student Engagement
    # Let's assume the student picked the first recommendation and spent 5 minutes on it
    chosen_content = recommendations[0]
    print(f"\n🖱️ Student clicked on: {chosen_content['title']}")
    print("⏳ Simulating learning session (5 seconds)...")
    time.sleep(5)
    
    # 5. Calculate Reward and Update Brain
    # In a real LMS, you'd calculate this based on quiz scores, time spent, etc.
    # For this demo, let's say they passed a quick quiz after the video.
    reward = 0.85 # High reward for good engagement/score
    
    print(f"✅ Session Complete. Reporting reward {reward} to the Brain...")
    client.update(student_id=student_id, content_id=chosen_content['content_id'], reward=reward)
    
    # 6. Verify Diagnostics
    print("\n📈 Refreshing Brain Diagnostics...")
    diag = client.get_diagnostics()
    print(f"⭐ New Neural Score: {diag['neural_score']:.2f}/10.0")
    print(f"🎯 Exploration: {diag['exploration_score']:.1f} | Sensitivity: {diag['context_score']:.1f}")

    print("\n" + "="*50)
    print("🚀 DEMO COMPLETE: The Brain is now smarter!")

if __name__ == "__main__":
    simulate_lms_workflow()
