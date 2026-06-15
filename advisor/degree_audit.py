import json
import os

try:
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "courses.json")) as f:
        all_courses = json.load(f)
except FileNotFoundError:
    all_courses = {}
    print("Warning: data/courses.json not found.")

try:
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "requirements.json")) as f:
        REQUIREMENTS = json.load(f)
except FileNotFoundError:
    REQUIREMENTS = {}
    print("Warning: requirements.json not found.")

uid_to_code = {v["uid"]: v["code"] for v in all_courses.values()}
code_to_course = {v["code"]: v for v in all_courses.values()}

# Known 1-credit courses that may have null cred_min in courses.json
KNOWN_CREDITS = {
    "CSCI8970": 1,
    "CSCI8001": 1,
    "CSCI8002": 1,
}


def degree_audit(completed_courses: list, program: str = "ms") -> str:
    completed = [c.upper().replace(" ", "") for c in completed_courses]

    if program not in REQUIREMENTS:
        return f"Unknown program: {program}. Valid options: ms, phd"

    req = REQUIREMENTS[program]
    breadth_categories = req["breadth_categories"]
    required_breadth_courses = req.get("required_breadth_courses", len(breadth_categories))
    colloquium = req["colloquium"]
    results = []

    # ── Breadth ──────────────────────────────────────────────────────────────
    results.append("BREADTH REQUIREMENTS:")
    # Track ALL matching courses per category (PhD may need >1 total)
    breadth_met = {}   # category -> list[course_code]
    for category, courses in breadth_categories.items():
        fulfilled = [c for c in completed if c in courses]
        breadth_met[category] = fulfilled
        if fulfilled:
            results.append(f"  ✅ {category.replace('_', ' ').title()}: {', '.join(fulfilled)}")
        else:
            eligible = courses[:4]
            results.append(
                f"  ❌ {category.replace('_', ' ').title()}: not fulfilled\n"
                f"     Eligible courses include: {', '.join(eligible)}"
            )

    total_breadth_completed = sum(len(v) for v in breadth_met.values())
    categories_covered = [c for c in breadth_categories if breadth_met.get(c)]
    # breadth_met values are lists; empty list is falsy so this correctly finds unsatisfied categories
    categories_missing = [c for c in breadth_categories if not breadth_met.get(c)]
    all_categories_covered = len(categories_missing) == 0
    breadth_complete = all_categories_covered and total_breadth_completed >= required_breadth_courses

    if breadth_complete:
        results.append(f"  ✅ All {len(breadth_categories)} breadth areas satisfied.")
    else:
        results.append(
            f"  Breadth courses completed: {total_breadth_completed} of "
            f"{required_breadth_courses} required"
        )
    if not breadth_complete:
        if categories_missing:
            results.append(
                f"  ⚠️ Missing category: {', '.join(c.replace('_', ' ').title() for c in categories_missing)}"
            )
        extra_needed = required_breadth_courses - total_breadth_completed
        if extra_needed > 0:
            if all_categories_covered:
                results.append(
                    f"  ⚠️ All categories covered but {extra_needed} more breadth "
                    f"course(s) needed — may be from any breadth area."
                )
            else:
                results.append(
                    f"  ⚠️ Need {extra_needed} more breadth course(s); at least "
                    f"{len(categories_missing)} must satisfy the missing category above."
                )
    results.append("")

    # ── Required courses ──────────────────────────────────────────────────────
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

    # ── Credit count ──────────────────────────────────────────────────────────
    csci_credits = 0
    for code in completed:
        if not code.startswith("CSCI"):
            continue
        course = code_to_course.get(code)
        if course:
            cred = course.get("cred_min")
            if cred is not None:
                csci_credits += cred
            else:
                # Use known credit values before falling back to 3
                csci_credits += KNOWN_CREDITS.get(code, 3)

    results.append("CREDIT PROGRESS:")
    results.append(f"  CSCI credits completed: {csci_credits}/{req['csci_credits']} required")
    results.append(f"  Note: add non-CSCI courses manually for total credit count")
    results.append("")

    # ── Summary ───────────────────────────────────────────────────────────────
    results.append("SUMMARY:")
    if breadth_complete and colloquium in completed:
        results.append(
            "  ✅ Core CSCI requirements appear satisfied based on the courses you provided. "
            "This is a preliminary checklist only — verify your official status via GPAS in "
            "MyU and confirm with csgradmn@umn.edu before making graduation decisions."
        )
    else:
        missing = []
        if categories_missing:
            missing.append(f"breadth in: {', '.join(categories_missing)}")
        extra_needed = required_breadth_courses - total_breadth_completed
        if extra_needed > 0 and all_categories_covered:
            missing.append(f"{extra_needed} additional breadth course(s) from any area")
        if colloquium not in completed:
            missing.append(f"{colloquium} colloquium")
        results.append(f"  Still needed: {'; '.join(missing)}")

    return "\n".join(results)


if __name__ == "__main__":
    test_courses = ["CSCI5521", "CSCI5103", "CSCI8970"]
    print(degree_audit(test_courses, program="ms"))