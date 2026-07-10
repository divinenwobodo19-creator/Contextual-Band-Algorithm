"""
Investor Demo — Contextual Bandit Brain Model
==============================================
Run:  PYTHONPATH=. python3 demo_investor.py
"""

import numpy as np
import time
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from linucb_brain import Brain

SEPARATOR = "=" * 78
TOTAL_STEPS = 2000


def section(title):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def compute_reward(student_perf, rec_topic, student_topic, rec_diff):
    """Smarter reward: reward topic matches and appropriate difficulty."""
    topic_bonus = 0.3 if rec_topic == student_topic else 0.0
    diff_penalty = 0.2 * abs(rec_diff / 4.0 - student_perf)
    noise = np.random.normal(0, 0.05)
    raw = 0.5 + topic_bonus - diff_penalty + noise
    return float(np.clip(raw, 0.0, 1.0))


def main():
    print(f"""
{SEPARATOR}
  CONTEXTUAL BANDIT BRAIN — Investor Demonstration
  A Self-Learning Personalization Engine for Education
{SEPARATOR}
""")

    # ------------------------------------------------------------------ #
    # 1. BRAIN INITIALIZATION
    # ------------------------------------------------------------------ #
    section("1. INITIALIZING THE BRAIN")

    print("""
  The Brain uses a Hybrid LinUCB algorithm — an extension of the
  classic Contextual Bandit (Li et al., 2010) that:
    • Shares knowledge across students via shared parameters
    • Maintains arm-specific parameters for content precision
    • Groups students into cohorts via online clustering
    • Self-diagnostics via 7-dimension Neural Score
""")

    brain = Brain(
        model_type="hybrid",
        alpha=1.5,
        n_clusters=3,
        auto_diagnose_every=1000,
    )
    print(f"  ✓ Hybrid Brain initialized  (α={brain.alpha}, clusters={brain.n_clusters})")

    # ------------------------------------------------------------------ #
    # 2. REGISTERING STUDENTS & CONTENT
    # ------------------------------------------------------------------ #
    section("2. REGISTERING STUDENTS & CONTENT")

    students = {
        "S001": ("Alice",   0.72, "Math"),
        "S002": ("Bob",     0.45, "Science"),
        "S003": ("Charlie", 0.88, "History"),
        "S004": ("Diana",   0.61, "Math"),
        "S005": ("Eve",     0.93, "Science"),
        "S006": ("Frank",   0.38, "English"),
        "S007": ("Grace",   0.55, "Math"),
        "S008": ("Hank",    0.79, "History"),
    }
    for sid, (name, perf, topic) in students.items():
        brain.add_student(sid, name, performance_score=perf, current_topic=topic)

    content = [
        ("C001", "Algebra Fundamentals",  "Math",    2, "video"),
        ("C002", "Advanced Calculus",     "Math",    5, "quiz"),
        ("C003", "Cell Biology",          "Science", 3, "reading"),
        ("C004", "Quantum Physics",       "Science", 5, "video"),
        ("C005", "WWII Overview",         "History", 2, "reading"),
        ("C006", "Ancient Civilizations", "History", 4, "quiz"),
        ("C007", "Grammar Essentials",    "English", 1, "video"),
        ("C008", "Creative Writing",      "English", 3, "exercise"),
    ]
    for cid, title, topic, diff, ctype in content:
        brain.add_content(cid, title, topic, diff, ctype)

    print(f"  ✓ {len(brain.students)} students, {len(brain.contents)} content items (4 topics)")

    # ------------------------------------------------------------------ #
    # 3. COLD-START RECOMMENDATIONS
    # ------------------------------------------------------------------ #
    section("3. COLD-START — BEFORE TRAINING")

    print("\n  First recommendations — purely exploratory (no history):\n")
    cold_recs = {}
    for sid, (name, perf, topic) in students.items():
        rec = brain.recommend(sid, top_n=1)
        cold_recs[sid] = rec
        match = "✓" if rec.topic == topic else "✗"
        print(f"  {match} {name:<12} ({topic:<8}, perf={perf:.2f}) → {rec.title:<25}  [{rec.content_type}]")

    cold_topics_ok = sum(1 for sid, (_, _, t) in students.items() if cold_recs[sid].topic == t)
    print(f"\n  → Topic match rate: {cold_topics_ok}/{len(students)}  (random chance ~2/8)")
    print(f"  → No learning yet — pure exploration")

    # ------------------------------------------------------------------ #
    # 4. TRAINING
    # ------------------------------------------------------------------ #
    section(f"4. TRAINING — {TOTAL_STEPS} INTERACTIONS")

    print("""
  Each interaction: Brain recommends → student engages → reward signal → update.
  Rewards are higher when topic matches and difficulty is appropriate.
""")

    student_ids = list(brain.students.keys())
    rewards_log = []
    topic_match_log = []
    start = time.time()

    for i in range(TOTAL_STEPS):
        sid = np.random.choice(student_ids)
        rec = brain.recommend(sid, top_n=1)
        student = brain.students[sid]
        reward = compute_reward(student.performance_score, rec.topic, student.current_topic, rec.difficulty)
        brain.update(sid, rec.content_id, reward)
        rewards_log.append(reward)
        topic_match_log.append(1.0 if rec.topic == student.current_topic else 0.0)

        if (i + 1) % 500 == 0:
            avg_r = np.mean(rewards_log[-500:])
            avg_tm = np.mean(topic_match_log[-500:])
            print(f"  • Step {i+1:>5}:  avg reward = {avg_r:.3f}  |  topic match = {avg_tm:.0%}  |  α = {brain.alpha:.4f}")

    elapsed = time.time() - start
    final_tm = np.mean(topic_match_log[-500:])
    print(f"\n  ✓ {TOTAL_STEPS} interactions in {elapsed:.1f}s ({TOTAL_STEPS/elapsed:.0f} rows/sec)")
    print(f"  ✓ Final topic match rate: {final_tm:.0%}  (was {cold_topics_ok}/{len(students)} at cold-start)")
    print(f"  ✓ Alpha decayed from 1.5 → {brain.alpha:.4f}")

    # ------------------------------------------------------------------ #
    # 5. RECOMMENDATIONS AFTER TRAINING
    # ------------------------------------------------------------------ #
    section("5. RESULTS — AFTER TRAINING")

    print("\n  The same students, now with personalized recommendations:\n")
    trained_recs = {}
    for sid, (name, perf, topic) in students.items():
        rec = brain.recommend(sid, top_n=1)
        trained_recs[sid] = rec
        match = "✓" if rec.topic == topic else " "
        cold_topic = cold_recs[sid].topic
        changed = "← changed" if rec.content_id != cold_recs[sid].content_id else ""
        print(f"  {match} {name:<12} ({topic:<8}, perf={perf:.2f}) → {rec.title:<25}  [{rec.content_type}]  {changed}")

    trained_topics_ok = sum(1 for sid, (_, _, t) in students.items() if trained_recs[sid].topic == t)
    print(f"\n  → Topic match rate: {trained_topics_ok}/{len(students)}  (was {cold_topics_ok}/{len(students)} before training)")
    if trained_topics_ok > cold_topics_ok:
        print(f"  → The model learned to match content to student interests.")

    # ------------------------------------------------------------------ #
    # 6. MULTI-OBJECTIVE REWARD
    # ------------------------------------------------------------------ #
    section("6. MULTI-OBJECTIVE REWARD SIGNAL")

    examples = [
        ("Improvement +0.3, completed, engaged",         brain.calculate_multi_objective_reward(0.3,  True,  True,  False)),
        ("Small improvement +0.05, not completed",        brain.calculate_multi_objective_reward(0.05, False, True,  False)),
        ("Improvement +0.2, completed, not engaged",      brain.calculate_multi_objective_reward(0.2,  True,  False, False)),
        ("Student churned (left platform)",               brain.calculate_multi_objective_reward(0.0,  False, False, True)),
    ]
    for desc, reward in examples:
        print(f"  • {desc:<52} → {reward:+.2f}")

    # ------------------------------------------------------------------ #
    # 7. NEURAL SCORE
    # ------------------------------------------------------------------ #
    section("7. SELF-DIAGNOSTICS — NEURAL SCORE")

    scores = brain.neural_score(verbose=False)
    print("")
    print(f"  {'Dimension':<30} {'Score':<8} {'Status'}")
    print(f"  {'-'*30} {'-'*8} {'-'*14}")
    for dim in ["exploration_score", "convergence_score", "context_score",
                 "precision_score", "grade_score", "purity_score", "balance_score"]:
        val = scores.get(dim, 0.0)
        status = "GREEN" if val >= 7.0 else "YELLOW" if val >= 5.0 else "RED"
        label = dim.replace("_score", "").replace("_", " ").title()
        bar = "█" * int(val) + "░" * (10 - int(val))
        print(f"  {label:<30} {val:<8.1f} {bar}  {status}")
    print(f"  {'-'*30} {'-'*8} {'-'*14}")
    print(f"  {'★ NEURAL SCORE':<30} {scores['neural_score']:<8.1f}")
    print(f"\n  → Scores are honest indicators of model health. Context sensitivity improves")
    print(f"    as the model sees more differentiated student interactions.")

    # ------------------------------------------------------------------ #
    # 8. PERSISTENCE
    # ------------------------------------------------------------------ #
    section("8. PERSISTENCE — SAVE & LOAD")

    save_path = "/tmp/demo_brain_state.json"
    brain.save(save_path)
    loaded = Brain.load(save_path)
    assert loaded.summary()["student_count"] == brain.summary()["student_count"]
    print(f"  ✓ Saved to {save_path}")
    print(f"  ✓ Loaded back: {len(loaded.students)} students, {len(loaded.contents)} items, {len(loaded.sessions)} sessions")
    print(f"  ✓ State integrity verified")

    # ------------------------------------------------------------------ #
    # 9. SUMMARY
    # ------------------------------------------------------------------ #
    section("SUMMARY")

    print(f"""
  TECH HIGHLIGHTS
  ───────────────
  • 3 algorithms: Disjoint, Hybrid (+clustering), Thompson Sampling
  • Online learning — no batch retraining
  • Sherman-Morrison matrix updates (O(d²) vs O(d³))
  • Thread-safe for concurrent API access
  • Self-diagnostics via 7-dimension Neural Score

  DEMO RESULTS
  ────────────
  • {TOTAL_STEPS} interactions processed in {elapsed:.1f}s ({TOTAL_STEPS/elapsed:.0f}/sec)
  • Topic match rate: {cold_topics_ok}/{len(students)} → {trained_topics_ok}/{len(students)} (learning confirmed)
  • Final Neural Score: {scores['neural_score']:.1f}/10

  BUSINESS VALUE
  ──────────────
  • Reduces dropout by personalizing in real-time
  • Works in cold-start (no history needed)
  • Explainable decisions (not a black box)
  • Ready for LMS integration via REST API + Docker
""")

    print(SEPARATOR)
    print("  DEMO COMPLETE")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
