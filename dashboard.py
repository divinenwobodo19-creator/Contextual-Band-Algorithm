import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import glob
from datetime import datetime
import re

st.set_page_config(
    page_title="LinUCB Brain - Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 LinUCB Brain: Live Dashboard")
st.markdown("---")

st.sidebar.header("Controls")
data_source = st.sidebar.radio("Data Source", ["Auto-detect", "Demo (brain_state.json)", "OULAD Checkpoints"])
refresh = st.sidebar.button("🔄 Refresh Now")
if refresh:
    st.rerun()


def get_checkpoint_index(filename):
    try:
        match = re.search(r'checkpoint_(\d+)k', filename)
        return int(match.group(1)) if match else 0
    except:
        return 0


def load_brain_state():
    """Load live Brain object from brain_state.json (saved by demo)."""
    try:
        from linucb_brain import Brain
        brain = Brain.load("brain_state.json")
        return brain
    except Exception as e:
        return None


def get_oulad_checkpoints():
    files = glob.glob("oulad_checkpoint_*.json")
    if os.path.exists("oulad_brain_state.json"):
        files.append("oulad_brain_state.json")
    if not files:
        return None, None
    files.sort(key=os.path.getmtime, reverse=True)
    latest = files[0]
    try:
        with open(latest) as f:
            state = json.load(f)
        if latest == "oulad_brain_state.json":
            label = "Final State"
        else:
            idx = get_checkpoint_index(latest)
            is_emergency = "emergency" in latest
            label = f"Checkpoint {idx}k" + (" (Emergency)" if is_emergency else "")
        return state, label
    except:
        return None, None


def get_checkpoint_trend():
    checkpoints = glob.glob("oulad_checkpoint_*.json")
    if not checkpoints:
        return pd.DataFrame()
    all_data = []
    for f in checkpoints:
        try:
            with open(f) as jf:
                d = json.load(jf)
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
    return df.sort_values('sessions') if not df.empty else df


# ---------- LOAD DATA ----------
brain = None
oulad_state = None
oulad_label = None

if data_source == "Auto-detect":
    brain = load_brain_state()
    if brain is None:
        oulad_state, oulad_label = get_oulad_checkpoints()
elif data_source == "Demo (brain_state.json)":
    brain = load_brain_state()
elif data_source == "OULAD Checkpoints":
    oulad_state, oulad_label = get_oulad_checkpoints()


# ---------- RENDER ----------
if brain is not None:
    summary = brain.summary()
    scores = brain.neural_score(verbose=False)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Source", "Demo (brain_state.json)")
    with m2:
        st.metric("Students", f"{summary['student_count']}")
    with m3:
        st.metric("Content Items", f"{summary['content_count']}")
    with m4:
        st.metric("Sessions", f"{summary['total_sessions']:,}")

    st.subheader("Neural Score")
    s1, s2 = st.columns([1, 2])

    with s1:
        ns = scores.get('neural_score', 0)
        st.metric("Neural Score", f"{ns:.2f}/10.0")

        dims = {
            'Exploration': scores.get('exploration_score', 0),
            'Convergence': scores.get('convergence_score', 0),
            'Context': scores.get('context_score', 0),
            'Precision': scores.get('precision_score', 0),
            'Grade': scores.get('grade_score', 0),
            'Purity': scores.get('purity_score', 5.0),
            'Balance': scores.get('balance_score', 7.5),
        }
        df_dim = pd.DataFrame({'Dimension': list(dims.keys()), 'Score': list(dims.values())})
        st.bar_chart(df_dim.set_index('Dimension'))

    with s2:
        st.write("**Recommendation Log (last 10)**")
        sessions = brain.sessions[-10:] if brain.sessions else []
        if sessions:
            rows = []
            for s in sessions:
                sid = s.student_id
                student = brain.students.get(sid)
                topic = getattr(student, 'current_topic', '?') if student else '?'
                rows.append({
                    'Student': sid,
                    'Topic': topic,
                    'Content': s.content_id,
                    'Reward': f"{s.reward:.3f}" if s.reward is not None else "-",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No sessions recorded yet.")

    st.markdown("---")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.write(f"**Model:** `{summary['model_type']}`")
    with p2:
        st.write(f"**Alpha:** `{summary['alpha']:.4f}`")
    with p3:
        st.write(f"**Students:** `{summary['student_count']}` / **Content:** `{summary['content_count']}`")

elif oulad_state is not None:
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Source", oulad_label)
    with m2:
        st.metric("Students", f"{oulad_state.get('student_count', 0):,}")
    with m3:
        st.metric("Arms", f"{oulad_state.get('content_count', 0):,}")
    with m4:
        st.metric("Sessions", f"{oulad_state.get('total_sessions', 0):,}")

    st.subheader("Neural Score")
    last_score = oulad_state.get('last_neural_score')
    if last_score:
        s1, s2 = st.columns([1, 2])
        with s1:
            st.metric("Neural Score", f"{last_score.get('neural_score', 0):.2f}/10.0")
            dims = {
                'Exploration': last_score.get('exploration_score', 0),
                'Convergence': last_score.get('convergence_score', 0),
                'Sensitivity': last_score.get('context_score', 0),
                'Precision': last_score.get('precision_score', 0),
                'Grade': last_score.get('grade_score', 0),
                'Purity': last_score.get('purity_score', 5.0),
                'Balance': last_score.get('balance_score', 7.5),
            }
            df_dim = pd.DataFrame({'Dimension': list(dims.keys()), 'Score': list(dims.values())})
            st.bar_chart(df_dim.set_index('Dimension'))
        with s2:
            st.write("**Performance Trend**")
            df_trend = get_checkpoint_trend()
            if not df_trend.empty:
                st.line_chart(df_trend.set_index('sessions')[['neural_score', 'exploration', 'sensitivity']])
            else:
                st.info("Trend data appears after multiple checkpoints.")

    st.markdown("---")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.write(f"**Model:** `{oulad_state.get('model_type', 'N/A')}`")
    with p2:
        st.write(f"**Alpha:** `{oulad_state.get('current_alpha', 0):.4f}`")
    with p3:
        st.write(f"**Gamma:** `{oulad_state.get('current_gamma', 0):.4f}`")

    if oulad_label != "Final State":
        progress = oulad_state.get('total_sessions', 0) / 10655280
        st.progress(min(progress, 1.0), f"Progress: {progress*100:.1f}%")
    else:
        st.success("OULAD simulation complete! All 10.6M interactions.")

else:
    st.warning("No data found. Run the demo first:")
    st.code("cd /path/to/project && PYTHONPATH=. python3 demo_investor.py", language="bash")
    st.code("PYTHONPATH=. streamlit run dashboard.py", language="bash")


