import json

# Load all course data
with open("data/courses.json") as f:
    all_courses = json.load(f)

# Build a UID to course code map
uid_to_code = {v["uid"]: v["code"] for v in all_courses.values()}

# Filter just CSCI courses
csci_courses = {v["code"]: v for v in all_courses.values() if v.get("subject") == "CSCI"}

def resolve_prereqs(prereq, depth=0):
    """Convert UID-based prereq tree into readable string."""
    if not prereq:
        return "None"
    if isinstance(prereq, str):
        return uid_to_code.get(prereq, prereq)
    if isinstance(prereq, dict):
        if "and" in prereq:
            parts = [resolve_prereqs(p) for p in prereq["and"]]
            return "(" + " AND ".join(parts) + ")"
        if "or" in prereq:
            parts = [resolve_prereqs(p) for p in prereq["or"]]
            return "(" + " OR ".join(parts) + ")"
    return str(prereq)

def check_prerequisites(course_code: str) -> str:
    """Look up prerequisites for a CSCI course."""
    code = course_code.upper().replace(" ", "")
    if code not in csci_courses:
        return f"Course {course_code} not found. Make sure to use the format CSCI5521."
    course = csci_courses[code]
    prereq_str = resolve_prereqs(course["prereq"])
    return f"{code} - {course['name']}\nPrerequisites: {prereq_str}"

if __name__ == "__main__":
    # Test it
    print(check_prerequisites("CSCI5521"))
    print()
    print(check_prerequisites("CSCI4041"))