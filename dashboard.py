import streamlit as st
import pandas as pd
import numpy as np
# import plotly.express as px  # Bypassing plotly due to environment issues
# import plotly.graph_objects as go
import json
import os
import glob
import time
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="LinUCB Brain - Real-time Dashboard",
    page_icon="🧠",
    layout="wide"
)

# Title
st.title("🧠 LinUCB Brain: OULAD Live Simulation")
st.markdown("---")

# Sidebar for controls
st.sidebar.header("Simulation Controls")
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 60, 10)
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)

import re

def get_checkpoint_index(filename):
    try:
        match = re.search(r'checkpoint_(\d+)k', filename)
        if match:
            return int(match.group(1))
        return 0
    except Exception:
        return 0

# Helper to load the latest checkpoint or brain state
def get_latest_state():
    # Get all potential state files
    files = glob.glob("oulad_checkpoint_*.json")
    if os.path.exists("oulad_brain_state.json"):
        files.append("oulad_brain_state.json")
    
    if not files:
        return None, None
        
    # Sort by modification time (most recent first)
    files.sort(key=os.path.getmtime, reverse=True)
    latest_file = files[0]
    
    try:
        with open(latest_file, "r") as f:
            state = json.load(f)
            
        if latest_file == "oulad_brain_state.json":
            label = "Final State"
        else:
            # Extract k number from oulad_checkpoint_100k.json or emergency files
            idx = get_checkpoint_index(latest_file)
            is_emergency = "emergency" in latest_file
            label = f"Checkpoint {idx}k" + (" (Emergency)" if is_emergency else "")
            
        return state, label
    except Exception as e:
        st.error(f"Error loading {latest_file}: {str(e)}")
        return None, None

# Helper to load all checkpoints for trend analysis
def get_all_checkpoint_metrics():
    checkpoints = glob.glob("oulad_checkpoint_*.json")
    if not checkpoints:
        return pd.DataFrame()
    
    all_data = []
    for f in checkpoints:
        try:
            with open(f, "r") as json_file:
                d = json.load(json_file)
                score = d.get('last_neural_score', {})
                if score:
                    all_data.append({
                        'sessions': d.get('total_sessions', 0),
                        'neural_score': score.get('neural_score', 0),
                        'exploration': score.get('exploration_score', 0),
                        'sensitivity': score.get('context_score', 0)
                    })
        except:
            continue
            
    df = pd.DataFrame(all_data)
    if not df.empty:
        df = df.sort_values('sessions')
    return df

state, source = get_latest_state()

if state:
    # 1. Top Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.metric("Source", source)
    with m2:
        st.metric("Total Students", f"{state.get('student_count', 0):,}")
    with m3:
        st.metric("Total Arms", f"{state.get('content_count', 0):,}")
    with m4:
        st.metric("Sessions Processed", f"{state.get('total_sessions', 0):,}")

    # 2. Neural Score Display
    st.subheader("Latest Neural Score")
    
    last_score = state.get('last_neural_score')
    if last_score:
        s1, s2 = st.columns([1, 2])
        
        with s1:
            # Display overall score clearly
            st.metric("Neural Score", f"{last_score.get('neural_score', 0):.2f}/10.0")
            
            # Dimension breakdown
            dimensions = {
                'Exploration': last_score.get('exploration_score', 0),
                'Convergence': last_score.get('convergence_score', 0),
                'Sensitivity': last_score.get('context_score', 0),
                'Precision': last_score.get('precision_score', 0),
                'Grade Pred.': last_score.get('grade_score', 0),
                'Cohort Purity': last_score.get('purity_score', 5.0),
                'Obj. Balance': last_score.get('balance_score', 7.5)
            }
            df_dim = pd.DataFrame({'Dimension': list(dimensions.keys()), 'Score': list(dimensions.values())})
            st.bar_chart(df_dim.set_index('Dimension'))
            
        with s2:
            # 2.5 Trend Analysis (New)
            st.write("**Performance Trend**")
            df_trend = get_all_checkpoint_metrics()
            if not df_trend.empty:
                st.line_chart(df_trend.set_index('sessions')[['neural_score', 'exploration', 'sensitivity']])
            else:
                st.info("Trend data will appear after multiple checkpoints are saved.")

    # 3. Model Parameters
    st.markdown("---")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.write(f"**Model Type:** `{state.get('model_type', 'N/A')}`")
    with p2:
        st.write(f"**Alpha (Exploration):** `{state.get('current_alpha', 0):.4f}`")
    with p3:
        st.write(f"**Gamma (Discounting):** `{state.get('current_gamma', 0):.4f}`")

    # 4. Progress Placeholder (If running)
    if source != "Final State":
        progress = state.get('total_sessions', 0) / 10655280
        st.progress(min(progress, 1.0), text=f"Simulation Progress: {progress*100:.1f}%")
        st.info("The simulation is currently running. Refresh to see updated scores.")
    else:
        st.success("Simulation Complete! All 10.6M interactions processed.")

else:
    st.warning("No brain state or checkpoints found. Start the simulation first!")
    if st.button("Check Again"):
        st.rerun()

# Auto-refresh logic
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
