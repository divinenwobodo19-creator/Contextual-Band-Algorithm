
from linucb_brain import Brain

def main():
    print("=" * 80)
    print("  EXPLORING AVAILABLE CONTENT IN FULL MODEL")
    print("=" * 80)
    print()

    print("Loading model...")
    brain = Brain.load("oulad_checkpoint_4300k.json")
    print("Model loaded!")
    print(f"  Total content items: {len(brain.contents):,}")
    print()

    # Analyze content
    types = {}
    difficulties = {}
    all_contents = []
    for content in brain.contents.values():
        all_contents.append(content)
        t = content.content_type
        d = content.difficulty
        types[t] = types.get(t, 0) + 1
        difficulties[d] = difficulties.get(d, 0) + 1

    print("=== Content Types ===")
    for t, count in types.items():
        print(f"  - {t}: {count:,} items")
    print()

    print("=== Difficulty Levels ===")
    max_diff = max(difficulties.keys())
    for d in sorted(difficulties.keys()):
        print(f"  - {d}/5: {difficulties[d]:,} items")
    print(f"  Highest difficulty: {max_diff}/5")
    print()

    # Find the highest difficulty content of each type
    print("=== HIGHEST DIFFICULTY CONTENT ===")
    high_diff_contents = [c for c in all_contents if c.difficulty == max_diff]
    print(f"  Found {len(high_diff_contents):,} items with difficulty {max_diff}/5!")
    print()

    # Separate by type
    type_groups = {}
    for c in high_diff_contents:
        t = c.content_type
        if t not in type_groups:
            type_groups[t] = []
        type_groups[t].append(c)
    
    for t in sorted(type_groups.keys()):
        print(f"  --- {t.upper()} ---")
        # Show top 5 by average reward
        sorted_contents = sorted(type_groups[t], key=lambda x: -x.avg_reward)[:5]
        for i, c in enumerate(sorted_contents, 1):
            print(f"    {i}. Content {c.content_id}")
            print(f"       Difficulty: {c.difficulty}/5")
            print(f"       Avg Reward: {c.avg_reward:.4f}")
            print(f"       Recommended/Rewarded: {c.times_recommended:,}/{c.times_rewarded:,}")
        print()
    
    print("=" * 80)
    print()
    
    # Now get recommendations for a student, filtering to quiz/video and max difficulty
    print("=== GETTING TARGETED RECOMMENDATIONS ===")
    student_id = list(brain.students.keys())[0]
    print(f"For student: {student_id}")
    print(f"Looking for: QUIZ/VIDEO, DIFFICULTY {max_diff}/5")
    print()
    
    # Let's manually check what's available for quiz/video at max diff
    if 'quiz' in type_groups:
        top_quiz = sorted(type_groups['quiz'], key=lambda x: -x.avg_reward)[0]
        print(f"  TOP MAX-DIFFICULTY QUIZ:")
        print(f"    Content: {top_quiz.content_id}")
        print(f"    Avg Reward: {top_quiz.avg_reward:.4f}")
        print(f"    Success Rate: {top_quiz.times_rewarded}/{top_quiz.times_recommended}")
    
    if 'video' in type_groups:
        top_video = sorted(type_groups['video'], key=lambda x: -x.avg_reward)[0]
        print(f"  TOP MAX-DIFFICULTY VIDEO:")
        print(f"    Content: {top_video.content_id}")
        print(f"    Avg Reward: {top_video.avg_reward:.4f}")
        print(f"    Success Rate: {top_video.times_rewarded}/{top_video.times_recommended}")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
