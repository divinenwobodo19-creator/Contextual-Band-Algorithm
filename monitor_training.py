import json
import glob
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import re

def get_checkpoint_index(filename):
    try:
        match = re.search(r'checkpoint_(\d+)k', filename)
        if match:
            return int(match.group(1))
        return 0
    except Exception:
        return 0

def monitor_checkpoints():
    print("Starting Training Performance Monitor...")
    print("Watching for 'oulad_checkpoint_*.json' files...")
    
    plt.ion() # Interactive mode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    last_processed = set()
    history = []

    try:
        while True:
            # Find all checkpoints
            checkpoints = glob.glob("oulad_checkpoint_*.json")
            # Sort using robust index extraction
            checkpoints.sort(key=get_checkpoint_index)
            
            new_data = False
            for cp in checkpoints:
                if cp not in last_processed:
                    try:
                        with open(cp, 'r') as f:
                            data = json.load(f)
                            
                        # Extract metrics
                        step = get_checkpoint_index(cp)
                        update_count = data.get('update_count', 0)
                        
                        # Calculate average reward from contents if sessions aren't available
                        contents = data.get('contents', {})
                        avg_reward = 0
                        if contents:
                            rewards = [c.get('avg_reward', 0) for c in contents.values() if c.get('times_rewarded', 0) > 0]
                            avg_reward = sum(rewards) / len(rewards) if rewards else 0
                        
                        history.append({
                            'step_k': step,
                            'updates': update_count,
                            'avg_reward': avg_reward,
                            'timestamp': datetime.now()
                        })
                        last_processed.add(cp)
                        new_data = True
                        print(f"Processed {cp}: Avg Reward = {avg_reward:.4f}")
                    except Exception as e:
                        print(f"Error reading {cp}: {e}")

            if new_data and history:
                df = pd.DataFrame(history)
                
                # Plot 1: Average Reward Over Time
                ax1.clear()
                ax1.plot(df['step_k'], df['avg_reward'], marker='o', color='b')
                ax1.set_title('Model Performance (Average Reward)')
                ax1.set_ylabel('Avg Reward')
                ax1.grid(True)

                # Plot 2: Learning Progress (Updates)
                ax2.clear()
                ax2.bar(df['step_k'], df['updates'], color='g', alpha=0.6)
                ax2.set_title('Training Progress (Total Updates)')
                ax2.set_xlabel('Checkpoint (k interactions)')
                ax2.set_ylabel('Total Updates')
                ax2.grid(True)

                plt.tight_layout()
                plt.draw()
                plt.pause(0.1)
                
            time.sleep(10) # Wait 10 seconds before checking again
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    monitor_checkpoints()
