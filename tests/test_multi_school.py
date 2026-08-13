"""
Regression tests for the multi-school data model and triage semantics:
- WAEC boundary tiers (0.39 / 0.40 / 0.74 / 0.75)
- assessed-only triage (performance_score fallback must NOT tier students)
- school/class label collisions stay isolated across schools
- legacy flat registry auto-migration is idempotent
"""
import json
import pytest
from linucb_brain import Brain, Student
from linucb_brain.registry import (
    load_registry, save_registry, _migrate, ensure_school, add_class,
    get_schools, get_classes, get_class_by_id,
)

# ── Triage boundaries ────────────────────────────────────────────────────────

def test_triage_threshold_boundaries():
    brain = Brain()
    for sid, score in [("R", 0.39), ("O", 0.40), ("O2", 0.74), ("A", 0.75)]:
        st = brain.add_student(sid, sid)
        st.grade_history["Maths"] = [score]

    result = brain.triage("Maths")
    assert result["total_students"] == 4
    tier_of = {}
    for name in TIER_ORDER:
        for s in result["tiers"][name]["students"]:
            tier_of[s["student_id"]] = name

    assert tier_of["R"] == "remediation"
    assert tier_of["O"] == "on_track"
    assert tier_of["O2"] == "on_track"
    assert tier_of["A"] == "ahead"
    assert result["total_students"] == sum(
        result["tiers"][name]["count"] for name in TIER_ORDER)


def test_triage_excludes_unassessed_students():
    brain = Brain()
    assessed = brain.add_student("S1", "Amina")
    assessed.grade_history["Maths"] = [0.42]
    cold = brain.add_student("S2", "Bola")
    cold.performance_score = 0.9  # fallback must NOT create a tier entry

    result = brain.triage("Maths")
    assert result["total_students"] == 1
    names = [s["name"] for t in result["tiers"].values() for s in t["students"]]
    assert names == ["Amina"]
    assert result["scope"] == "assessed_only"


def test_triage_empty_subject_returns_full_shape():
    brain = Brain()
    brain.add_student("S1", "Ada")
    result = brain.triage("NobodyTookThis")
    assert result["total_students"] == 0
    assert set(result["tiers"].keys()) == set(TIER_ORDER)
    assert all(result["tiers"][k]["count"] == 0 for k in TIER_ORDER)
    assert result["scope"] == "assessed_only"


# ── School / class scoping ───────────────────────────────────────────────────

def test_identical_class_labels_across_schools_do_not_collide():
    reg = {"term_subjects": [], "schools": []}
    s1 = ensure_school(reg, "Greenwood Academy")
    s2 = ensure_school(reg, "St. Theresa College")
    c1 = add_class(reg, s1["school_id"], "JSS1A")
    c2 = add_class(reg, s2["school_id"], "JSS1A")

    assert c1 is not None and c2 is not None
    assert c1["label"] == c2["label"] == "JSS1A"
    assert c1["class_id"] != c2["class_id"]

    assert get_class_by_id(reg, c1["class_id"], s2["school_id"]) is None
    assert get_class_by_id(reg, c2["class_id"], s1["school_id"]) is None
    assert get_class_by_id(reg, c1["class_id"], s1["school_id"]) == c1


def test_add_class_collision_enforced_within_school_only():
    reg = {"term_subjects": [], "schools": []}
    s = ensure_school(reg, "Demo School")
    first = add_class(reg, s["school_id"], "JSS1A")
    dup = add_class(reg, s["school_id"], " jss1a ")  # normalized same label
    assert first is not None and dup is None
    assert len(s["classes"]) == 1


def test_students_carry_school_and_class_ids():
    brain = Brain()
    s1 = brain.add_student("S1", "Chinwe", school_id="schA", class_id="clsA")
    s2 = brain.add_student("S2", "Emeka", school_id="schB", class_id="clsA")
    assert s1.school_id == "schA" and s1.class_id == "clsA"
    assert s2.school_id == "schB" and s2.class_id == "clsA"
    assert s1.student_id != s2.student_id


# ── Registry migration ───────────────────────────────────────────────────────

def test_legacy_flat_registry_migrates_to_default_school():
    legacy = {"term_subjects": ["Maths"], "classes": ["300 LEVEL", "JSS 1"]}
    migrated = _migrate(legacy)
    schools = get_schools(migrated)
    assert len(schools) == 1
    assert schools[0]["name"] == "MY SCHOOL"
    labels = [c["label"] for c in schools[0]["classes"]]
    assert sorted(labels) == ["300 LEVEL", "JSS 1"]
    assert all(c["class_id"] for c in schools[0]["classes"])


def test_migration_is_idempotent():
    reg = _migrate({"term_subjects": [], "classes": ["JSS1A", "JSS2B"]})
    again = _migrate(json.loads(json.dumps(reg)))
    assert len(get_schools(again)) == 1
    assert len(get_schools(again)[0]["classes"]) == 2
    assert [c["label"] for c in get_schools(again)[0]["classes"]] == ["JSS1A", "JSS2B"]


def test_save_and_load_registry_roundtrip(tmp_path):
    path = tmp_path / "config.json"
    reg = {"term_subjects": ["Maths"], "schools": []}
    school = ensure_school(reg, "Demo School")
    add_class(reg, school["school_id"], "JSS1A")
    save_registry(reg, str(path))
    loaded = load_registry(str(path))
    assert get_schools(loaded)[0]["name"] == "DEMO SCHOOL"
    assert get_classes(loaded, school["school_id"])[0]["label"] == "JSS1A"


TIER_ORDER = ["remediation", "on_track", "ahead"]