import requests
import time
import os
import sys

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def render_dashboard(summary, diagnostics):
    clear_screen()
    print("=" * 60)
    print("           LINUCB BRAIN — LIVE MONITOR")
    print("=" * 60)
    
    # Summary
    print(f"Status: ONLINE | Model: {summary.get('model_type', 'N/A').upper()}")
    print(f"Total Sessions: {summary.get('total_sessions', 0):<10} | Students: {summary.get('student_count', 0)}")
    print(f"Current Alpha:  {summary.get('current_alpha', 0):.4f}     | Gamma:    {summary.get('current_gamma', 0):.4f}")
    print(f"Total Regret:   {summary.get('cumulative_regret', 0):.2f}")
    print("-" * 60)
    
    # Neural Score ASCII Report
    final = diagnostics.get('neural_score', 0.0)
    if final >= 8.5: status = "EXCELLENT"
    elif final >= 7.0: status = "GOOD"
    elif final >= 5.0: status = "FAIR"
    else: status = "WEAK"
    
    print(f"NEURAL SCORE: {final:.1f}/10.0 ({status})")
    
    metrics = [
        ("Exploration", "exploration_score"),
        ("Convergence", "convergence_score"),
        ("Sensitivity", "context_score"),
        ("Precision", "precision_score"),
        ("Grade Pred.", "grade_score")
    ]
    
    for label, key in metrics:
        score = diagnostics.get(key, 0.0)
        bar_len = int(score * 3)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"{label:15} | {score:4.1f} | {bar}")
        
    print("-" * 60)
    print("Polling every 2 seconds... (Ctrl+C to stop)")

def main():
    api_url = os.getenv("BRAIN_API_URL", "http://localhost:8000")
    
    while True:
        try:
            summary = requests.get(f"{api_url}/summary").json()
            diagnostics = requests.get(f"{api_url}/diagnostics").json()
            render_dashboard(summary, diagnostics)
        except Exception as e:
            print(f"Error connecting to Brain API at {api_url}: {e}")
            
        time.sleep(2)

if __name__ == "__main__":
    main()
