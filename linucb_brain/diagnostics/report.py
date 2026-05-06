from typing import Dict

def render_neural_report(scores: Dict[str, float]) -> str:
    """
    Render the Neural Score report as ASCII.
    
    Args:
        scores (Dict[str, float]): Scores from diagnostics.
        
    Returns:
        str: Formatted ASCII report.
    """
    final = scores.get('neural_score', 0.0)
    
    # Status thresholds:
    # - 8.5–10.0 → EXCELLENT
    # - 7.0–8.4  → GOOD
    # - 5.0–6.9  → FAIR
    # - below 5  → WEAK
    if final >= 8.5:
        status = "EXCELLENT"
    elif final >= 7.0:
        status = "GOOD"
    elif final >= 5.0:
        status = "FAIR"
    else:
        status = "WEAK"
        
    report = f"""
╔══════════════════════════════════════════════╗
║           LINUCB BRAIN — NEURAL SCORE        ║
╠══════════════════════════════════════════════╣
║  Exploration Efficiency   →   {scores.get('exploration_score', 0.0):.1f}/10     ║
║  Reward Convergence       →   {scores.get('convergence_score', 0.0):.1f}/10     ║
║  Context Sensitivity      →   {scores.get('context_score', 0.0):.1f}/10     ║
║  Recommendation Precision →   {scores.get('precision_score', 0.0):.1f}/10     ║
║  Grade Prediction         →   {scores.get('grade_score', 0.0):.1f}/10     ║
║  Cluster Cohort Purity    →   {scores.get('purity_score', 5.0):.1f}/10     ║
║  Objective Balance        →   {scores.get('balance_score', 7.5):.1f}/10     ║
╠══════════════════════════════════════════════╣
║  ★  NEURAL SCORE :  {final:.1f} / 10.0   ★      ║
║  Status: {status}                               ║
╚══════════════════════════════════════════════╝
"""
    return report.strip()
