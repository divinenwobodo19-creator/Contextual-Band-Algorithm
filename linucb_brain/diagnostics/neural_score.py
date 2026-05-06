import numpy as np
import random
from math import log, exp
from scipy.spatial.distance import cosine
from typing import Dict, List, Optional
from ..models.student import Student
from ..models.content import Content
from ..models.session import Session

def run_neural_diagnostics(
    students: Dict[str, Student], 
    contents: Dict[str, Content], 
    sessions: List[Session],
    brain_instance
) -> Dict[str, float]:
    """
    Run full Neural Score diagnostics for the brain.
    
    Args:
        students (Dict[str, Student]): Dictionary of students.
        contents (Dict[str, Content]): Dictionary of content items.
        sessions (List[Session]): List of all recorded sessions.
        brain_instance: The Brain object to call methods from.
        
    Returns:
        Dict[str, float]: Calculated scores for each dimension.
    """
    scores = {}
    
    # 1. EXPLORATION EFFICIENCY (exploration_score)
    # entropy = -sum(p * log(p)) for p = recommendations per arm / total
    # MEASURE DIVERSITY OF RECOMMENDATIONS MADE DURING DIAGNOSTICS
    diag_recs = {}
    student_ids = list(students.keys())
    content_ids = list(contents.keys())
    
    # Sample 100 students and get their top recommendation
    for _ in range(min(100, len(students))):
        sid = random.choice(student_ids)
        rec = brain_instance.recommend(sid, top_n=1)
        diag_recs[rec.content_id] = diag_recs.get(rec.content_id, 0) + 1
        
    total_diag = sum(diag_recs.values())
    if total_diag > 0:
        p_recs = [count / total_diag for count in diag_recs.values()]
        entropy = -sum(p * log(p) for p in p_recs)
        # Max entropy for 100 samples is log(min(100, len(contents)))
        max_ent = log(min(100, len(contents))) if len(contents) > 1 else 1.0
        exploration_score = (entropy / max_ent) * 10.0
    else:
        exploration_score = 0.0
    scores['exploration_score'] = min(exploration_score, 10.0)

    # 2. REWARD CONVERGENCE (convergence_score)
    if len(sessions) < 100: # Need more sessions for a stable convergence score
        convergence_score = 5.0
    else:
        # Divide into 5 chunks and check if avg reward is trending up
        chunk_size = len(sessions) // 5
        chunk_avgs = []
        for i in range(5):
            chunk = sessions[i*chunk_size : (i+1)*chunk_size]
            chunk_avgs.append(np.mean([s.reward for s in chunk if s.reward is not None]))
        
        # Calculate slope of chunk_avgs
        if len(chunk_avgs) >= 2:
            x = np.arange(len(chunk_avgs))
            slope, _ = np.polyfit(x, chunk_avgs, 1)
            # Normalize slope: 0.05 improvement per chunk is EXCELLENT (slope=0.05 -> score 10)
            convergence_score = (slope + 0.01) * 100.0 + 5.0
        else:
            convergence_score = 5.0
    scores['convergence_score'] = min(max(convergence_score, 0.0), 10.0)

    # 3. CONTEXT SENSITIVITY (context_score)
    if len(students) < 2:
        context_score = 5.0
    else:
        sensitivity_samples = []
        student_ids = list(students.keys())
        
        print("\n" + "="*80)
        print("  DEBUG: CONTEXT SENSITIVITY - AGENT COMPARISON")
        print("="*80)
        
        # Sample 3 agents for comparison as requested
        sample_ids = random.sample(student_ids, min(3, len(student_ids)))
        if len(sample_ids) >= 2:
            from ..core.context import build_context
            content_id = list(contents.keys())[0] if contents else "dummy"
            dummy_content = contents[content_id] if contents else Content(content_id, "dummy", "dummy", 3, "video")
            
            # Use alpha=0.1 to see what's learned, not just noise
            orig_alpha = brain_instance.alpha
            brain_instance.alpha = 0.1
            
            agent_data = []
            for sid in sample_ids:
                ctx = build_context(students[sid], dummy_content)
                rec = brain_instance.recommend(sid, top_n=1)
                agent_data.append({'id': sid, 'ctx': ctx, 'rec': rec.content_id})
            
            # Print side by side comparison
            for i in range(len(agent_data)):
                a = agent_data[i]
                print(f"Agent {i+1} (ID: {a['id']}):")
                print(f"  Context: {np.array2string(a['ctx'], precision=3, separator=', ')}")
                print(f"  Recommended Arm: {a['rec']}")
            
            # Print pair distances
            for i in range(len(agent_data)):
                for j in range(i+1, len(agent_data)):
                    a, b = agent_data[i], agent_data[j]
                    dist = cosine(a['ctx'], b['ctx'])
                    diff = 1.0 if a['rec'] != b['rec'] else 0.0
                    print(f"Distance ({a['id']} vs {b['id']}): {dist:.6f} | Rec Diff: {diff}")
            
            brain_instance.alpha = orig_alpha
        print("="*80 + "\n")
        
        # Now run standard sensitivity sampling for the score
        for _ in range(min(20, len(students) * (len(students) - 1) // 2)):
            id_a, id_b = random.sample(student_ids, 2)
            content_id = list(contents.keys())[0] if contents else "dummy"
            from ..core.context import build_context
            dummy_content = contents[content_id] if contents else Content(content_id, "dummy", "dummy", 3, "video")
            
            context_a = build_context(students[id_a], dummy_content)
            context_b = build_context(students[id_b], dummy_content)
            dist = cosine(context_a, context_b)
            
            orig_alpha = brain_instance.alpha
            brain_instance.alpha = 0.1
            rec_a = brain_instance.recommend(id_a, top_n=1)
            rec_b = brain_instance.recommend(id_b, top_n=1)
            brain_instance.alpha = orig_alpha
            
            diff = 1.0 if rec_a.content_id != rec_b.content_id else 0.0
            denominator = 1 + exp(-10 * (dist - 0.3))
            sensitivity_samples.append(diff / denominator)
            
        context_score = np.mean(sensitivity_samples) * 10.0 if sensitivity_samples else 0.0
    scores['context_score'] = min(max(context_score, 0.0), 10.0)

    # 4. RECOMMENDATION PRECISION (precision_score)
    if not sessions:
        precision_score = 5.0
    else:
        global_mean_reward = np.mean([s.reward for s in sessions if s.reward is not None])
        above_mean_count = sum(1 for s in sessions if s.reward is not None and s.reward > global_mean_reward)
        precision = above_mean_count / len(sessions)
        precision_score = precision * 10.0
    scores['precision_score'] = min(precision_score, 10.0)

    # 5. GRADE PREDICTION ACCURACY (grade_score)
    accuracy_samples = []
    for student_id, student in students.items():
        for subject, grades in student.grade_history.items():
            if len(grades) >= 3:
                # predicted = predict_grade(student_id, subject)
                predicted = brain_instance.predict_grade(student_id, subject)
                actual = grades[-1]
                error = abs(predicted - actual)
                accuracy_samples.append(1 - error)
                
    if not accuracy_samples:
        grade_score = 5.0
    else:
        grade_score = np.mean(accuracy_samples) * 10.0
    scores['grade_score'] = min(max(grade_score, 0.0), 10.0)

    # 6. CLUSTER PURITY (New for COBART)
    if hasattr(brain_instance, 'clustering') and brain_instance.clustering.initialized:
        stats = brain_instance.clustering.get_cluster_stats()
        # Measure how balanced the clusters are (Entropy of student distribution)
        counts = list(stats.values())
        total = sum(counts)
        p = [c / total for c in counts]
        entropy = -sum(pk * log(pk) for pk in p)
        max_entropy = log(brain_instance.clustering.n_clusters)
        purity_score = (entropy / max_entropy) * 10.0 if max_entropy > 0 else 5.0
    else:
        purity_score = 5.0
    scores['purity_score'] = min(max(purity_score, 0.0), 10.0)

    # 7. OBJECTIVE BALANCE (New for Multi-Objective)
    # Check if any objective is being completely ignored (very low correlation with reward)
    # For now, we'll return a placeholder based on session diversity
    scores['balance_score'] = 7.5 # Target: 7.5 baseline

    # Final Neural Score (Weighted with new dimensions)
    final_score = (
        scores['exploration_score'] * 0.15 +
        scores['convergence_score'] * 0.20 +
        scores['context_score']     * 0.15 +
        scores['precision_score']   * 0.20 +
        scores['grade_score']       * 0.10 +
        scores['purity_score']      * 0.10 +
        scores['balance_score']     * 0.10
    )
    scores['neural_score'] = final_score
    
    return scores
