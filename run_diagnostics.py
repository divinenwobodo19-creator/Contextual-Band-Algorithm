import os
import glob
from linucb_brain import Brain
import json

def run_diagnostics():
    print("🩺 Running Neural Score Diagnostics...")
    
    # 1. Load the most recent checkpoint
    checkpoints = glob.glob("oulad_checkpoint_*.json")
    if not checkpoints:
        print("❌ No checkpoints found to analyze.")
        return
        
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    latest_cp = checkpoints[0]
    print(f"📂 Loading latest checkpoint: {latest_cp}")
    
    try:
        brain = Brain.load(latest_cp)
        
        # 2. Run Neural Score
        print("\n📊 Calculating Neural Scores...")
        scores = brain.neural_score(verbose=True)
        
        # 3. Print a formatted health report
        print("\n" + "="*40)
        print("       BRAIN HEALTH REPORT")
        print("="*40)
        
        # Mapping scores to qualitative status
        def get_status(val):
            if val >= 8.0: return "🟢 EXCELLENT"
            if val >= 6.0: return "🟡 GOOD"
            if val >= 4.0: return "🟠 FAIR"
            return "🔴 POOR"

        print(f"Overall Neural Score: {scores['neural_score']:.2f} / 10.0")
        print(f"Status: {get_status(scores['neural_score'])}")
        print("-" * 20)
        print(f"1. Exploration:  {scores['exploration_score']:.2f} - {get_status(scores['exploration_score'])}")
        print(f"2. Convergence:  {scores['convergence_score']:.2f} - {get_status(scores['convergence_score'])}")
        print(f"3. Contextual:   {scores['context_score']:.2f} - {get_status(scores['context_score'])}")
        print(f"4. Precision:    {scores['precision_score']:.2f} - {get_status(scores['precision_score'])}")
        print(f"5. Cluster Purity: {scores['purity_score']:.2f} - {get_status(scores['purity_score'])}")
        print(f"6. Obj. Balance: {scores['balance_score']:.2f} - {get_status(scores['balance_score'])}")
        print("="*40)
        
        # Final Recommendation
        if scores['neural_score'] > 7.0:
            print("\n🚀 RECOMMENDATION: The model is highly stable and ready for production deployment.")
        else:
            print("\n🛠️ RECOMMENDATION: Continue training or refine rewards to improve precision.")

    except Exception as e:
        print(f"❌ Error during diagnostics: {e}")

if __name__ == "__main__":
    run_diagnostics()
