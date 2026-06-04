from linucb_brain import Brain
import pandas as pd
import os

# Create data directory if it doesn't exist (though it should)
os.makedirs("data/oulad", exist_ok=True)

# Load preprocessed data
print("Loading preprocessed data (this can take 30-60s for 10M rows)...", flush=True)
agents_df       = pd.read_csv("data/oulad/agents_clean.csv")
arms_df         = pd.read_csv("data/oulad/arms_clean.csv")
interactions_df = pd.read_csv("data/oulad/interactions_clean.csv")
print(f"Data loaded: {len(interactions_df):,} interactions ready.", flush=True)

import glob
import signal
import sys

# Global flag for graceful shutdown
interrupted = False

def signal_handler(sig, frame):
    global interrupted
    print("\n🛑 Shutdown signal received. Saving state before exiting...", flush=True)
    interrupted = True

# Register the signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

# Initialize brain 
BRAIN_STATE_PATH = "oulad_brain_state.json"

def get_checkpoint_index(filename):
     try:
         # Extract numbers from filename (e.g., oulad_checkpoint_1300k_emergency.json -> 1300)
         # Using regex for robust extraction of the index
         import re
         match = re.search(r'checkpoint_(\d+)k', filename)
         if match:
             return int(match.group(1))
         return 0
     except Exception:
         return 0

def get_latest_checkpoint():
    checkpoints = glob.glob("oulad_checkpoint_*.json")
    if not checkpoints:
        return None
    
    # Separate regular and emergency checkpoints
    regular_cps = [cp for cp in checkpoints if "emergency" not in cp]
    emergency_cps = [cp for cp in checkpoints if "emergency" in cp]
    
    # Prioritize regular checkpoints first
    if regular_cps:
        regular_cps.sort(key=get_checkpoint_index)
        return regular_cps[-1]
    elif emergency_cps:
        emergency_cps.sort(key=get_checkpoint_index)
        return emergency_cps[-1]
    else:
        return None

latest_cp = get_latest_checkpoint()

if latest_cp:
    print(f"Resuming from checkpoint: {latest_cp}", flush=True)
    brain = Brain.load(latest_cp)
    start_idx = get_checkpoint_index(latest_cp) * 1000
else:
    print ("NO checkpoint found,Starting fresh...", flush=True)
    brain = Brain(algorithm="hybrid", alpha=2.0, auto_diagnose_every=100000, track_sessions=True, max_sessions=50000) 
    start_idx = 0

# Register all agents 
if not latest_cp:
    print(f"Loading {len(agents_df):,} agents...", flush=True) 
    for row in agents_df.itertuples(): 
        brain.add_agent( 
            agent_id=str(row.agent_id), 
            features={ 
                "performance_score": row.performance_score, 
                "education_level":   row.education_level, 
                "age_band":          row.age_band, 
                "credits_studied":   row.credits_studied,
                "imd_band":          row.imd_band,
                "region_code":       row.region_code
            } 
        ) 

    # Register all arms 
    print(f"Loading {len(arms_df):,} arms...", flush=True) 
    for row in arms_df.itertuples(): 
        brain.add_arm( 
            arm_id=str(row.arm_id), 
            features={ 
                "activity_code": row.activity_code, 
                "difficulty":    row.difficulty,
                "activity_type": row.activity_type
            } 
        ) 
    print(f"Successfully registered {len(brain.contents):,} arms in the brain.", flush=True)

# Stream interactions through the brain 
# NOW RUNNING FULL 10.6M DATASET
print(f"Running FULL {len(interactions_df):,} interactions simulation from index {start_idx:,}...\n", flush=True) 
import time
start_time = time.time()

for i, row in enumerate(interactions_df.iloc[start_idx:].itertuples(), start=start_idx): 
    if interrupted:
        # Save emergency checkpoint
        emergency_cp = f"oulad_checkpoint_{i//1000}k_emergency.json"
        brain.save(emergency_cp)
        print(f"💾 Emergency checkpoint saved at {i:,} to {emergency_cp}", flush=True)
        sys.exit(0)

    brain.update( 
        agent_id=str(row.agent_id), 
        arm_id=str(row.arm_id), 
        reward=float(row.reward) 
    ) 

    if i % 10000 == 0 and i > start_idx: 
        elapsed = time.time() - start_time
        processed_in_run = i - start_idx
        speed = processed_in_run / elapsed
        print(f"  Processed {i:,} interactions... ({speed:.1f} rows/sec)", flush=True) 
        if i % 100000 == 0:
            # Checkpoint every 100k for live dashboard visibility
            brain.save(f"oulad_checkpoint_{i//1000}k.json")
            print(f"  Checkpoint saved at {i:,}", flush=True)

# Final Neural Score 
print("\nRunning final Neural Score diagnostics...") 
brain.neural_score(verbose=True) 

# Save trained brain 
brain.save("oulad_brain_state.json") 
print("\nBrain state saved to oulad_brain_state.json") 
