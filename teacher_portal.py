"""
Teacher Portal — Streamlit App
================================
For traditional schools: input weekly scores, get group recommendations.

Run: PYTHONPATH=. streamlit run teacher_portal.py
"""

import json, os, sys
import streamlit as st
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from linucb_brain import Brain

st.set_page_config(page_title="Teacher Portal", layout="wide")

CONFIG_FILE = "class_config.json"

# ── Helpers ──────────────────────────────────────────────────────────────────

def nigerian_grade(score: float) -> str:
    p = score * 100
    if p >= 75:  return "A1"
    if p >= 70:  return "B2"
    if p >= 65:  return "B3"
    if p >= 60:  return "C4"
    if p >= 55:  return "C5"
    if p >= 50:  return "C6"
    if p >= 45:  return "D7"
    if p >= 40:  return "E8"
    return "F9"


def nigerian_grade_description(score: float) -> str:
    p = score * 100
    if p >= 75:  return "Excellent"
    if p >= 70:  return "Very Good"
    if p >= 65:  return "Good"
    if p >= 60:  return "Credit"
    if p >= 55:  return "Credit"
    if p >= 50:  return "Credit"
    if p >= 45:  return "Pass"
    if p >= 40:  return "Pass"
    return "Fail"


def load_class_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {"term_subjects": [], "term_name": ""}


def save_class_config(config: dict) -> None:
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)


TIER_LABELS = {
    "remediation": {"label": "Needs Extra Help", "icon": "🔴",
        "note": "Scored below 40% (F9). Give them remedial exercises before moving on.",
        "color": "#E74C3C"},
    "on_track": {"label": "At Expected Level", "icon": "🟡",
        "note": "Scored 40–74% (E8 to B2). Continue with standard curriculum materials.",
        "color": "#F1C40F"},
    "ahead": {"label": "Ahead of Class", "icon": "🟢",
        "note": "Scored 75% or above (A1). Give them advanced or challenge materials.",
        "color": "#27AE60"},
}

TIER_ORDER = ["remediation", "on_track", "ahead"]

# ── Config & Brain ───────────────────────────────────────────────────────────

class_config = load_class_config()

@st.cache_resource
def get_brain():
    if os.path.exists("brain_state.json"):
        return Brain.load("brain_state.json")
    elif os.path.exists("school_brain_state.json"):
        return Brain.load("school_brain_state.json")
    else:
        b = Brain(model_type="hybrid", alpha=1.5, n_clusters=3)
        st.info("No saved brain found. Starting fresh.")
        return b

brain = get_brain()

# Seed term subjects in session state
if "term_subjects" not in st.session_state:
    st.session_state.term_subjects = class_config.get("term_subjects", [])

# Derived lists
student_list = sorted(brain.students.values(), key=lambda s: s.name)
subjects_with_scores = sorted(set(
    subj for s in brain.students.values()
    for subj in s.grade_history.keys()
))

# ── Layout ───────────────────────────────────────────────────────────────────

st.title("Teacher Portal")
st.markdown("Enter weekly test scores — see which students need extra help and what to give them.")

tab1, tab2, tab3, tab4 = st.tabs(["Enter Scores", "Class Groups", "Student Progress", "Print Report"])

# ============================================================================
# TAB 1 — Enter Scores
# ============================================================================
with tab1:
    st.subheader("Step 1: What subject are you teaching?")

    # ── Term subjects chips ─────────────────────────────────────────────────
    if st.session_state.term_subjects:
        subject_counts = {}
        for subj in st.session_state.term_subjects:
            count = 0
            for student in brain.students.values():
                g = student.grade_history.get(subj, [])
                if len(g) > count:
                    count = len(g)
            subject_counts[subj] = count

        st.markdown("**This term's subjects:**")
        chip_cols = st.columns(6)
        for i, subj in enumerate(st.session_state.term_subjects):
            cnt = subject_counts[subj]
            label = f"{subj} ×{cnt}" if cnt else subj
            with chip_cols[i % 6]:
                if st.button(label, key=f"chip_{i}", use_container_width=True):
                    st.session_state["subject_input"] = subj
                    st.rerun()

        with st.expander("Manage subjects"):
            st.caption("Remove subjects added by mistake, or add new ones.")
            for i, subj in enumerate(st.session_state.term_subjects):
                c1, c2 = st.columns([5, 1])
                cnt = subject_counts[subj]
                label = f"{subj}  ({cnt} entries)" if cnt else subj
                c1.markdown(f"**{label}**")
                if c2.button("Delete", key=f"del_subj_{i}", help=f"Remove {subj}"):
                    st.session_state.term_subjects.pop(i)
                    save_class_config({"term_subjects": st.session_state.term_subjects})
                    st.rerun()
            st.text_input("Add a subject", key="add_subj_input",
                          placeholder="Subject name", label_visibility="collapsed")
            if st.button("Add"):
                new_s = st.session_state.get("add_subj_input", "").strip()
                if new_s and new_s not in st.session_state.term_subjects:
                    st.session_state.term_subjects.append(new_s)
                    save_class_config({"term_subjects": st.session_state.term_subjects})
                    st.rerun()

    subject = st.text_input("Subject name (e.g. Algebra - Week 4, Geometry - Week 1)",
                            placeholder="Type the subject or topic name", key="subject_input")

    # ── Score entry table ────────────────────────────────────────────────────
    if not student_list:
        st.warning("No students yet. Add them in the Student Progress tab or load demo data below.")
    elif not subject.strip():
        st.info("Type a subject name above to begin.")
    else:
        st.subheader(f"Step 2: Enter scores for \"{subject}\" (0–100)")

        rows = []
        for student in student_list:
            grades = student.grade_history.get(subject, [])
            last_score = int(grades[-1] * 100) if grades else int(student.performance_score * 100)
            rows.append({
                "Student": student.name,
                "Score (0-100)": last_score,
            })

        df = pd.DataFrame(rows)
        edited_df = st.data_editor(
            df,
            column_config={
                "Student": st.column_config.TextColumn(
                    "Student", help="Edit student name if needed",
                ),
                "Score (0-100)": st.column_config.NumberColumn(
                    "Score (0-100)", min_value=0, max_value=100, step=1,
                    help="Enter score from 0 to 100",
                ),
            },
            hide_index=True,
            use_container_width=True,
            key=f"score_editor_{subject}",
        )

        # ── Quick preview bar ────────────────────────────────────────────────
        scores_pct = edited_df["Score (0-100)"].values
        rem = int((scores_pct < 40).sum())
        on_track = int(((scores_pct >= 40) & (scores_pct < 75)).sum())
        ahead = int((scores_pct >= 75).sum())
        st.markdown(
            f"<div style='padding:8px 12px;background:#f0f2f6;border-radius:6px;font-size:14px;'>"
            f"<b>Preview:</b> 🔴 Needs Extra Help: {rem} &nbsp;&nbsp; "
            f"🟡 At Expected Level: {on_track} &nbsp;&nbsp; "
            f"🟢 Ahead of Class: {ahead}"
            f"</div>",
            unsafe_allow_html=True,
        )

        if st.button("Submit Scores & Update Model", type="primary", use_container_width=True):
            entries = []
            name_changed = False
            for i, student in enumerate(student_list):
                new_name = edited_df.iloc[i]["Student"]
                if new_name != student.name:
                    brain.students[student.student_id].name = new_name
                    name_changed = True
                score = edited_df.iloc[i]["Score (0-100)"]
                entries.append({"student_id": student.student_id, "subject": subject, "score": score / 100.0})

            # Auto-add to term subjects on submit
            if subject not in st.session_state.term_subjects:
                st.session_state.term_subjects.append(subject)

            with st.spinner("Saving scores to model..."):
                result = brain.bulk_update(entries)
                brain.save("brain_state.json")
                save_class_config({"term_subjects": st.session_state.term_subjects})
            st.success(f"Scores saved for {result['processed']} students in \"{subject}\"."
                       + (" Student names also updated." if name_changed else ""))
            st.rerun()

    with st.expander("Load demo students (Nigerian names)"):
        demo_data = [
            ("S01", "Chinwe", 0.82, "Mathematics"),
            ("S02", "Emeka", 0.55, "Mathematics"),
            ("S03", "Amina", 0.91, "Mathematics"),
            ("S04", "Tunde", 0.63, "Mathematics"),
            ("S05", "Ngozi", 0.76, "English"),
            ("S06", "Chidi", 0.44, "English"),
            ("S07", "Zainab", 0.88, "English"),
            ("S08", "Efe", 0.59, "English"),
            ("S09", "Adaeze", 0.71, "Science"),
            ("S10", "Kelechi", 0.48, "Science"),
            ("S11", "Fatima", 0.93, "Science"),
            ("S12", "Obinna", 0.37, "Science"),
            ("S13", "Chioma", 0.79, "History"),
            ("S14", "Segun", 0.52, "History"),
            ("S15", "Yetunde", 0.85, "History"),
            ("S16", "Musa", 0.41, "History"),
        ]
        if st.button("Load 16 demo students"):
            for sid, name, grade, topic in demo_data:
                brain.add_student(sid, name, performance_score=grade, current_topic=topic)
            brain.save("brain_state.json")
            st.success("Demo students loaded! Type a subject above and enter their scores.")
            st.rerun()

# ============================================================================
# TAB 2 — Class Groups (Nigerian grading scale)
# ============================================================================
with tab2:
    st.subheader("Class Groups by Subject")

    if not subjects_with_scores:
        st.info("No scores entered yet. Go to 'Enter Scores' tab first.")
    else:
        subj = st.selectbox("Select subject to view groups",
                            subjects_with_scores, key="triage_subject",
                            on_change=lambda: st.session_state.pop("triage_trigger", None))

        # Auto-run — just reading the selectbox triggers the display below
        result = brain.triage(subj)
        st.metric("Total Students Assessed", result["total_students"])

        t1, t2, t3 = st.columns(3)
        for col, tier_name in zip([t1, t2, t3], TIER_ORDER):
            tier = result["tiers"][tier_name]
            info = TIER_LABELS[tier_name]
            with col:
                st.markdown(f"### {info['icon']} {info['label']}")
                st.markdown(f"**{tier['count']} students**")
                st.caption(info["note"])
                names = [(s["name"], s["predicted_score"]) for s in tier["students"]]
                if names:
                    for name, pscore in names:
                        grade = nigerian_grade(pscore)
                        st.markdown(f"— {name} ({grade})")
                else:
                    st.markdown("_No students in this group._")

# ============================================================================
# TAB 3 — Student Progress
# ============================================================================
with tab3:
    st.subheader("Student Progress by Subject")

    if not subjects_with_scores:
        st.info("No scores recorded yet. Go to 'Enter Scores' tab first.")
    else:
        selected_subj = st.selectbox("Select subject to view",
                                     subjects_with_scores, key="progress_subj")

        rows = []
        for student in student_list:
            grades = student.grade_history.get(selected_subj, [])
            if grades:
                latest = grades[-1]
                avg = sum(grades) / len(grades)
                numbered = "  ".join(
                    f"{i+1}: {g*100:.0f}" for i, g in enumerate(grades)
                )
                rows.append({
                    "Student": student.name,
                    "Latest": f"{latest*100:.0f}",
                    "Average": f"{avg*100:.0f}",
                    "Grade": nigerian_grade(avg),
                    "Tests Taken": len(grades),
                    "Scores Over Time": numbered,
                })
            else:
                rows.append({
                    "Student": student.name,
                    "Latest": "—",
                    "Average": "—",
                    "Grade": "—",
                    "Tests Taken": 0,
                    "Scores Over Time": "No scores yet",
                })

        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    with st.expander("Register a new student"):
        new_id = st.text_input("Student ID", placeholder="e.g. S17")
        new_name = st.text_input("Full Name", placeholder="e.g. John Doe")
        new_subject = st.text_input("Main Subject", placeholder="e.g. Math, Science")
        if st.button("Add Student"):
            if new_id and new_name:
                brain.add_student(new_id, new_name, performance_score=0.5, current_topic=new_subject or "")
                brain.save("brain_state.json")
                st.success(f"Added {new_name}!")
                st.rerun()

# ============================================================================
# TAB 4 — Printable Report
# ============================================================================
with tab4:
    st.subheader("Printable Class Report")

    if not subjects_with_scores:
        st.info("No scores entered yet.")
    else:
        report_subj = st.selectbox("Subject for report", subjects_with_scores, key="report_subj")

        if st.button("Generate Report", type="primary"):
            result = brain.triage(report_subj)

            waec_legend = """
            <h3 style="margin-top:30px;">WAEC Grading Scale Reference</h3>
            <table style="width:auto; border-collapse:collapse; font-size:11pt;">
              <tr style="background:#2C3E50;color:white;">
                <th style="padding:6px 12px;">Grade</th>
                <th style="padding:6px 12px;">Score Range</th>
                <th style="padding:6px 12px;">Meaning</th>
              </tr>
              <tr style="background:#E8F8F5;"><td>A1</td><td>75–100%</td><td>Excellent</td></tr>
              <tr><td>B2</td><td>70–74%</td><td>Very Good</td></tr>
              <tr style="background:#FEF9E7;"><td>B3</td><td>65–69%</td><td>Good</td></tr>
              <tr><td>C4</td><td>60–64%</td><td>Credit</td></tr>
              <tr style="background:#FEF9E7;"><td>C5</td><td>55–59%</td><td>Credit</td></tr>
              <tr><td>C6</td><td>50–54%</td><td>Credit</td></tr>
              <tr style="background:#FEF9E7;"><td>D7</td><td>45–49%</td><td>Pass</td></tr>
              <tr><td>E8</td><td>40–44%</td><td>Pass</td></tr>
              <tr style="background:#FDEDEC;"><td>F9</td><td>0–39%</td><td>Fail</td></tr>
            </table>"""

            html = f"""
            <html><head><meta charset="utf-8">
            <style>
              body {{ font-family: Arial, sans-serif; font-size: 12pt; padding: 20px; }}
              h1 {{ color: #2C3E50; border-bottom: 3px solid #3498DB; padding-bottom: 8px; }}
              h2 {{ margin-top: 24px; }}
              .students {{ font-size: 12pt; line-height: 1.8; }}
              .box {{ padding: 12px 16px; border-radius: 6px; margin: 12px 0; }}
              .box-red {{ background: #FDEDEC; border-left: 5px solid #E74C3C; }}
              .box-yellow {{ background: #FEF9E7; border-left: 5px solid #F1C40F; }}
              .box-green {{ background: #E8F8F5; border-left: 5px solid #27AE60; }}
              .note {{ font-size: 11pt; color: #555; font-style: italic; }}
              .footer {{ margin-top: 30px; font-size: 10pt; color: #999; border-top: 1px solid #ccc; padding-top: 10px; }}
              .grade {{ font-weight: bold; }}
            </style></head>
            <body>
            <h1>Class Report — {report_subj}</h1>
            <p><strong>Total students assessed:</strong> {result['total_students']}</p>
            """

            tier_css = {"remediation": "box-red", "on_track": "box-yellow", "ahead": "box-green"}
            for tier_name in TIER_ORDER:
                tier = result["tiers"][tier_name]
                info = TIER_LABELS[tier_name]
                names = [(s["name"], s["predicted_score"]) for s in tier["students"]]
                html += f"<div class='box {tier_css[tier_name]}'>"
                html += f"<h2>{info['icon']} {info['label']} — {tier['count']} students</h2>"
                html += f"<p class='note'>{info['note']}</p>"
                if names:
                    for name, pscore in names:
                        grade = nigerian_grade(pscore)
                        desc = nigerian_grade_description(pscore)
                        html += f"<div class='students'>- {name} <span class='grade'>({grade} — {desc})</span></div>"
                else:
                    html += "<p>_No students in this group._</p>"
                html += "</div>"

            html += waec_legend
            html += f'<div class="footer">Generated by Smart Content Recommender — {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}</div>'
            html += "</body></html>"

            st.download_button(
                "Download HTML Report (print from browser)",
                html,
                file_name=f"class_report_{report_subj}.html",
                mime="text/html",
                use_container_width=True,
            )

            st.divider()
            st.markdown("### Preview")
            st.components.v1.html(html, height=600, scrolling=True)
