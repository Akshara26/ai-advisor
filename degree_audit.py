import json

# Load course data
try:
    with open("data/courses.json") as f:
        all_courses = json.load(f)
except FileNotFoundError:
    all_courses = {}
    print("Warning: data/courses.json not found.")

# Load degree requirements from config
try:
    with open("requirements.json") as f:
        REQUIREMENTS = json.load(f)
except FileNotFoundError:
    REQUIREMENTS = {}
    print("Warning: requirements.json not found.")

uid_to_code = {v["uid"]: v["code"] for v in all_courses.values()}
code_to_course = {v["code"]: v for v in all_courses.values()}

def degree_audit(completed_courses: list, program: str = "ms") -> str:
    completed = [c.upper().replace(" ", "") for c in completed_courses]
    
    if program not in REQUIREMENTS:
        return f"Unknown program: {program}. Valid options: ms, phd"
    
    req = REQUIREMENTS[program]
    breadth_categories = req["breadth_categories"]
    colloquium = req["colloquium"]
    results = []

    # Breadth requirement
    results.append("BREADTH REQUIREMENTS (one course from each area):")
    breadth_met = {}
    for category, courses in breadth_categories.items():
        fulfilled = [c for c in completed if c in courses]
        if fulfilled:
            breadth_met[category] = fulfilled[0]
            results.append(f"  ✅ {category.replace('_', ' ').title()}: {fulfilled[0]}")
        else:
            results.append(f"  ❌ {category.replace('_', ' ').title()}: not fulfilled")
    results.append("")

    # Required courses
    results.append("REQUIRED COURSES:")
    if colloquium in completed:
        results.append(f"  ✅ Colloquium ({colloquium}): complete")
    else:
        results.append(f"  ❌ Colloquium ({colloquium}): not completed")

    if program == "phd":
        intro = req.get("intro_research")
        if intro in completed:
            results.append(f"  ✅ Intro to Research ({intro}): complete")
        else:
            results.append(f"  ❌ Intro to Research ({intro}): not completed")
    results.append("")

    # Credit count
    csci_credits = 0
    for code in completed:
        course = code_to_course.get(code)
        if course and code.startswith("CSCI"):
            csci_credits += course.get("cred_min") or 3

    results.append("CREDIT PROGRESS:")
    results.append(f"  CSCI credits completed: {csci_credits}/{req['csci_credits']} required")
    results.append(f"  Note: add non-CSCI courses manually for total credit count")
    results.append("")

    # Summary
    breadth_complete = len(breadth_met) == len(breadth_categories)
    results.append("SUMMARY:")
    if breadth_complete and colloquium in completed:
        results.append("  Core requirements on track. Focus on reaching total credit requirement.")
    else:
        missing = []
        if not breadth_complete:
            missing_cats = [c for c in breadth_categories if c not in breadth_met]
            missing.append(f"breadth in: {', '.join(missing_cats)}")
        if colloquium not in completed:
            missing.append(f"{colloquium} colloquium")
        results.append(f"  Still needed: {'; '.join(missing)}")

    return "\n".join(results)


if __name__ == "__main__":
    test_courses = ["CSCI5521", "CSCI5103", "CSCI8970"]
    print(degree_audit(test_courses, program="ms"))