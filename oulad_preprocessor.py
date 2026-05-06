import pandas as pd
import numpy as np
from oulad_loader import load_raw_data

def build_agents(students: pd.DataFrame) -> pd.DataFrame:
    """Convert studentInfo into Brain-ready agent profiles."""

    # Encode categorical fields numerically
    edu_map = {
        "No Formal quals": 0, "Lower Than A Level": 1,
        "A Level or Equivalent": 2, "HE Qualification": 3,
        "Post Graduate Qualification": 4
    }
    result_map = {
        "Withdrawn": 0, "Fail": 0, "Pass": 1, "Distinction": 1
    }
    age_map = {
        "0-35": 0, "35-55": 1, "55<=": 2
    }
    imd_map = {
        "0-10%": 0.05, "10-20%": 0.15, "20-30%": 0.25, "30-40%": 0.35, "40-50%": 0.45,
        "50-60%": 0.55, "60-70%": 0.65, "70-80%": 0.75, "80-90%": 0.85, "90-100%": 0.95
    }
    region_list = sorted(students["region"].unique().tolist())
    region_map = {r: i for i, r in enumerate(region_list)}

    agents = pd.DataFrame()
    agents["agent_id"]           = students["id_student"]
    agents["performance_score"]  = students["num_of_prev_attempts"].clip(0, 5) / 5.0
    agents["education_level"]    = students["highest_education"].map(edu_map).fillna(0) / 4.0
    agents["age_band"]           = students["age_band"].map(age_map).fillna(0) / 2.0
    agents["credits_studied"]    = students["studied_credits"] / 600.0  # normalize
    agents["imd_band"]           = students["imd_band"].map(imd_map).fillna(0.5) # mean imputation
    agents["region_code"]        = students["region"].map(region_map).fillna(0) / float(len(region_list)-1)
    agents["final_result"]       = students["final_result"].map(result_map).fillna(0)

    return agents.dropna()


def build_arms(vle: pd.DataFrame) -> pd.DataFrame:
    """Convert VLE activity types into Brain-ready arm profiles."""

    activity_map = {
        "forumng": 0, "resource": 1, "quiz": 2,
        "oucontent": 3, "homepage": 4, "subpage": 5,
        "ouelluminate": 6, "dataplus": 7, "glossary": 8
    }

    arms = vle[["id_site", "activity_type"]].drop_duplicates()
    arms = arms.copy()
    arms["arm_id"]        = arms["id_site"]
    arms["activity_code"] = arms["activity_type"].map(activity_map).fillna(0) / 8.0
    arms["difficulty"]    = np.random.uniform(0.2, 0.8, len(arms))  # proxy

    return arms[["arm_id", "activity_type", "activity_code", "difficulty"]]


def build_interactions(
    student_vle: pd.DataFrame,
    student_assessment: pd.DataFrame,
    agents: pd.DataFrame,
    arms: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge VLE interactions with assessment scores to produce
    (agent_id, arm_id, reward) triples for Brain.update().
    Optimized to aggregate rewards per student to avoid row explosion.
    """

    # Aggregate assessment scores per student to 0.0–1.0 as reward signal
    # Use the average score for each student
    print("Aggregating student rewards...")
    student_rewards = student_assessment.groupby("id_student")["score"].mean().fillna(0) / 100.0
    student_rewards = student_rewards.reset_index().rename(columns={"score": "reward"})

    # Merge VLE clicks with agent and arm ids
    print("Merging VLE interactions with rewards (this may take a minute)...")
    interactions = student_vle.merge(
        student_rewards,
        on="id_student", how="left"
    )
    interactions["reward"] = interactions["reward"].fillna(0.0)

    # Keep only agents and arms that exist in our built profiles
    valid_agents = set(agents["agent_id"])
    valid_arms   = set(arms["arm_id"])

    interactions = interactions[
        interactions["id_student"].isin(valid_agents) &
        interactions["id_site"].isin(valid_arms)
    ]

    interactions = interactions.rename(columns={
        "id_student": "agent_id",
        "id_site":    "arm_id"
    })

    return interactions[["agent_id", "arm_id", "date", "sum_click", "reward"]]


if __name__ == "__main__":
    data         = load_raw_data()
    agents       = build_agents(data["students"])
    arms         = build_arms(data["vle"])
    interactions = build_interactions(
        data["student_vle"],
        data["student_assessment"],
        agents,
        arms
    )

    print(f"Agents built:       {len(agents):,}")
    print(f"Arms built:         {len(arms):,}")
    print(f"Interactions built: {len(interactions):,}")
    print("\nSample interaction:")
    print(interactions.head(5))

    # Save preprocessed data
    agents.to_csv("data/oulad/agents_clean.csv", index=False)
    arms.to_csv("data/oulad/arms_clean.csv", index=False)
    interactions.to_csv("data/oulad/interactions_clean.csv", index=False)
    print("\nClean data saved to data/oulad/")
