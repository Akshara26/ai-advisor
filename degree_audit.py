import json

# Load course data
with open("data/courses.json") as f:
    all_courses = json.load(f)

uid_to_code = {v["uid"]: v["code"] for v in all_courses.values()}
code_to_course = {v["code"]: v for v in all_courses.values()}

# Real breadth categories from UMN CS Graduate Handbook (page 9)
BREADTH_CATEGORIES = {
    "applications": [
        "CSCI5115", "CSCI5123", "CSCI5125", "CSCI5127", "CSCI5271",
        "CSCI5461", "CSCI5471", "CSCI5511", "CSCI5512", "CSCI5521",
        "CSCI5523", "CSCI5551", "CSCI5561", "CSCI5563", "CSCI5607",
        "CSCI5608", "CSCI5609", "CSCI5611", "CSCI5619", "CSCI5707",
        "CSCI5715"
    ],
    "theory_and_algorithms": [
        "CSCI5302", "CSCI5304", "CSCI5403", "CSCI5421", "CSCI5481",
        "CSCI5525", "CSCI5527"
    ],
    "architecture_systems_software": [
        "CSCI5103", "CSCI5104", "CSCI5105", "CSCI5106", "CSCI5161",
        "CSCI5204", "CSCI5211", "CSCI5221", "CSCI5231", "CSCI5451",
        "CSCI5552", "CSCI5708", "CSCI5751", "CSCI5801", "CSCI5802"
    ]
}

# Required courses
COLLOQUIUM = "CSCI8970"
PHD_INTRO = "CSCI8001"

def degree_audit(completed_courses: list, program: str = "ms") -> str:
    """
    Check degree progress for MS or PhD program.
    completed_courses: list of course codes e.g. ["CSCI5521", "CSCI8970"]
    program: "ms" or "phd"
    """
    completed = [c.upper().replace(" ", "") for c in completed_courses]
    results = []

    # --- Breadth requirement ---
    results.append("BREADTH REQUIREMENTS (one course from each area):")
    breadth_met = {}
    for category, courses in BREADTH_CATEGORIES.items():
        fulfilled = [c for c in completed if c in courses]
        if fulfilled:
            breadth_met[category] = fulfilled[0]
            results.append(f"  ✅ {category.replace('_', ' ').title()}: {fulfilled[0]}")
        else:
            results.append(f"  ❌ {category.replace('_', ' ').title()}: not fulfilled")

    breadth_complete = len(breadth_met) == 3
    results.append("")

    # --- Colloquium ---
    results.append("REQUIRED COURSES:")
    if COLLOQUIUM in completed:
        results.append(f"  ✅ Colloquium (CSCI8970): complete")
    else:
        results.append(f"  ❌ Colloquium (CSCI8970): not completed")

    if program == "phd":
        if PHD_INTRO in completed:
            results.append(f"  ✅ Intro to Research (CSCI8001): complete")
        else:
            results.append(f"  ❌ Intro to Research (CSCI8001): not completed")
    results.append("")

    # --- Credit count ---
    csci_credits = 0
    total_credits = 0
    for code in completed:
        course = code_to_course.get(code)
        if course:
            credits = course.get("cred_min") or 3
            total_credits += credits
            if code.startswith("CSCI"):
                csci_credits += credits

    min_total = 31 if program == "ms" else 52
    min_csci = 16

    results.append("CREDIT PROGRESS:")
    results.append(f"  Total credits (CSCI only): {csci_credits}/{min_csci} required CSCI credits")
    results.append(f"  Note: add non-CSCI courses manually for total credit count")
    results.append("")

    # --- Summary ---
    results.append("SUMMARY:")
    if breadth_complete and COLLOQUIUM in completed:
        results.append("  Core requirements are on track. Focus on reaching 31 total credits.")
    else:
        missing = []
        if not breadth_complete:
            missing_cats = [c for c in BREADTH_CATEGORIES if c not in breadth_met]
            missing.append(f"breadth courses in: {', '.join(missing_cats)}")
        if COLLOQUIUM not in completed:
            missing.append("CSCI8970 colloquium")
        results.append(f"  Still needed: {'; '.join(missing)}")

    return "\n".join(results)


if __name__ == "__main__":
    # Test with sample courses
    test_courses = ["CSCI5521", "CSCI5103", "CSCI8970"]
    print(degree_audit(test_courses, program="ms"))