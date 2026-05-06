import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/oulad/")

def load_raw_data():
    student_info        = pd.read_csv(DATA_PATH / "studentInfo.csv")
    student_assessment  = pd.read_csv(DATA_PATH / "studentAssessment.csv")
    student_vle         = pd.read_csv(DATA_PATH / "studentVle.csv")
    assessments         = pd.read_csv(DATA_PATH / "assessments.csv")
    vle                 = pd.read_csv(DATA_PATH / "vle.csv")
    courses             = pd.read_csv(DATA_PATH / "courses.csv")

    return {
        "students":           student_info,
        "student_assessment": student_assessment,
        "student_vle":        student_vle,
        "assessments":        assessments,
        "vle":                vle,
        "courses":            courses
    }

def preview_data(data: dict):
    for name, df in data.items():
        print(f"\n{'='*50}")
        print(f"  {name.upper()} — {df.shape[0]:,} rows, {df.shape[1]} cols")
        print(f"{'='*50}")
        print(df.head(3))
        print(f"Columns: {list(df.columns)}")

if __name__ == "__main__":
    data = load_raw_data()
    preview_data(data)
