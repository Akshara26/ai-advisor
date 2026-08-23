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

def _is_advanced_csci_course(
    code: str,
    advanced_config: dict,
    plan_config: dict,
) -> bool:
    """Return True if a course is eligible for the M.S. advanced CSCI requirement."""

    code = code.upper().replace(" ", "")

    # Explicitly approved 5xxx courses
    if code in advanced_config.get("approved_5xxx", []):
        return True

    # Courses that must not count as advanced
    exclusions = set(advanced_config.get("excluded_8xxx", []))
    exclusions.update(plan_config.get("extra_advanced_exclusions", []))

    if code in exclusions:
        return False

    # Any remaining CSCI 8xxx course is advanced-eligible.
    if advanced_config.get("allow_any_8xxx", False) and code.startswith("CSCI"):
        course_part = code[4:]

        # Handles codes such as CSCI8551 and potential suffixes.
        digits = ""
        for char in course_part:
            if char.isdigit():
                digits += char
            else:
                break

        if len(digits) == 4 and digits.startswith("8"):
            return True

    return False

def _calculate_advanced_csci_credits(
    completed_courses: list,
    advanced_config: dict,
    plan_config: dict,
) -> tuple[float, list[str]]:
    """
    Calculate confirmed advanced CSCI credits.

    Variable-credit courses such as CSCI8991/8994 are returned separately
    because a course code alone does not tell us how many credits were earned.
    """

    confirmed_credits = 0
    needs_credit_verification = []

    limited_courses = set(
        advanced_config
        .get("limited_8xxx", {})
        .get("courses", [])
    )

    for raw_code in completed_courses:
        code = raw_code.upper().replace(" ", "")

        if not _is_advanced_csci_course(
            code,
            advanced_config,
            plan_config,
        ):
            continue

        # We know these courses can count, but not how many credits
        # the student actually completed from the course code alone.
        if code in limited_courses:
            needs_credit_verification.append(code)
            continue

        course = code_to_course.get(code)

        if course:
            credits = course.get("cred_min")

            if credits is not None:
                confirmed_credits += credits
            else:
                confirmed_credits += KNOWN_CREDITS.get(code, 3)
        else:
            confirmed_credits += KNOWN_CREDITS.get(code, 3)

    return confirmed_credits, needs_credit_verification

def degree_audit(completed_courses: list, program: str = "ms", plan: str | None = None) -> str:
    if program == "ms" and plan not in ("A", "B", "C"):
        return "M.S. degree audits require a plan: A, B, or C."
    if program == "phd":
        plan = None

    completed = [c.upper().replace(" ", "") for c in completed_courses]

    if program not in REQUIREMENTS:
        return f"Unknown program: {program}. Valid options: ms, phd"

    req = REQUIREMENTS[program]
    plan_req = None

    if program == "ms":
        plan_req = req["plans"][plan]
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
    colloquium = "CSCI8970"

    if colloquium in completed:
        results.append(f"  ✅ Colloquium ({colloquium}): complete")
    else:
        results.append(f"  ❌ Colloquium ({colloquium}): not completed")

    plan_b_project_complete = True

    if program == "ms" and plan == "B":
        project_course = plan_req.get("project_course")
        plan_b_project_complete = project_course in completed

        if plan_b_project_complete:
            results.append(f"  ✅ Plan B project course ({project_course}): complete")
        else:
            results.append(f"  ❌ Plan B project course ({project_course}): not completed")

    if program == "ms" and plan == "A":
        thesis_course = plan_req.get("thesis_course")
        thesis_credits_required = plan_req.get("thesis_credits", 10)

        results.append("PLAN A THESIS REQUIREMENT:")

        if thesis_course in completed:
            results.append(
                f"  ⚠️ {thesis_course} is present, but earned thesis credits "
                f"must be verified manually ({thesis_credits_required} required)"
            )
        else:
            results.append(
                f"  ❌ {thesis_course} not found in provided courses "
                f"({thesis_credits_required} thesis credits required)"
            )

    results.append(
        "  ⚠️ Thesis committee and oral defense require manual/program verification"
    )

    if program == "ms" and plan == "C":
        project_hours = plan_req.get("project_hours", 100)

        results.append("PLAN C PROJECT REQUIREMENT:")
        results.append(
            f"  ⚠️ Not assessed from course codes: "
            f"{project_hours}-hour significant project"
        )
        results.append(
            "  ⚠️ Written project report requires manual/program verification"
        )
        results.append(
            "  ⚠️ Oral project presentation requires manual/program verification"
        )

    results.append("")

    intro_complete = True

    if program == "phd":
        intro = req.get("intro_research")
        intro_complete = intro in completed

        if intro_complete:
            results.append(f"  ✅ Intro to Research ({intro}): complete")
        else:
            results.append(f"  ❌ Intro to Research ({intro}): not completed")

    results.append("")

    # ── Credit count ──────────────────────────────────────────────────────────
    csci_credits = 0
    excluded_courses = []
    csci_credits = 0
    excluded_courses = []
    needs_csci_credit_verification = []

    variable_credit_courses = {"CSCI8991", "CSCI8994"}

    if program == "ms" and plan == "A":
        thesis_course = plan_req.get("thesis_course")

        if thesis_course:
            variable_credit_courses.add(thesis_course)

    for code in completed:
        if not code.startswith("CSCI"):
            continue
        course_number = code[4:]

        if course_number.isdigit() and int(course_number) < 5000:
            excluded_courses.append(
                f"{code} (4xxx-level course)"
            )
            continue

        if code in variable_credit_courses:
            needs_csci_credit_verification.append(code)
            continue

        course = code_to_course.get(code)

        if course:
            cred = course.get("cred_min")
            if cred is not None:
                csci_credits += cred
            else:
                # Use known credit values before falling back to 3
                csci_credits += KNOWN_CREDITS.get(code, 3)

    results.append("CSCI CREDIT REQUIREMENT:")
    results.append(f"  CSCI credits completed: {csci_credits}/{req['csci_credits']} required")
    if needs_csci_credit_verification:
        results.append(
            "  ⚠️ Credit value must be verified for: "
            + ", ".join(needs_csci_credit_verification)
        )
    if excluded_courses:
        results.append(
            f"  ⚠️ Excluded from degree credit count: {', '.join(excluded_courses)}"
        )
    results.append(f"  Note: add non-CSCI courses manually for total credit count")
    results.append("")

    advanced_complete = True
    advanced_credits = 0
    advanced_verify = []

    if program == "ms":
        advanced_config = req["advanced_csci"]

        advanced_credits, advanced_verify = _calculate_advanced_csci_credits(
            completed,
            advanced_config,
            plan_req,
        )

        required_advanced = plan_req["advanced_csci_credits"]

        advanced_complete = advanced_credits >= required_advanced

        results.append("ADVANCED CSCI REQUIREMENT:")
        results.append(
            f"  Confirmed advanced CSCI credits: "
            f"{advanced_credits}/{required_advanced} required"
        )

        if advanced_verify:
            results.append(
                "  ⚠️ Credit value must be verified for: "
                + ", ".join(advanced_verify)
            )

        results.append("")

    # ── Summary ───────────────────────────────────────────────────────────────
    results.append("SUMMARY:")

    csci_credit_complete = csci_credits >= req["csci_credits"]

    if (
        breadth_complete
        and colloquium in completed
        and csci_credit_complete
        and plan_b_project_complete
        and intro_complete
        and advanced_complete
    ):
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
        if not plan_b_project_complete:
            missing.append("CSCI8760 Plan B project course")
        if not intro_complete:
            missing.append(f"{intro} Intro to Research")
        if not advanced_complete:
            advanced_needed = required_advanced - advanced_credits

            if advanced_verify:
                missing.append(
                    f"{advanced_needed:g} more confirmed advanced CSCI credit(s) "
                    f"(verify earned credits for {', '.join(advanced_verify)})"
                )
            else:
                missing.append(
                    f"{advanced_needed:g} more advanced CSCI credit(s)"
                )
        if not csci_credit_complete:
            csci_needed = req["csci_credits"] - csci_credits

            if needs_csci_credit_verification:
                missing.append(
                    f"{csci_needed:g} more confirmed CSCI credit(s) "
                    f"(verify earned credits for {', '.join(needs_csci_credit_verification)})"
                )
            else:
                missing.append(
                    f"{csci_needed:g} more CSCI credit(s)"
                )
        results.append(f"  Still needed: {'; '.join(missing)}")

    return "\n".join(results)


if __name__ == "__main__":
    test_courses = ["CSCI5521", "CSCI5103", "CSCI8970"]
    print(degree_audit(test_courses, program="ms"))