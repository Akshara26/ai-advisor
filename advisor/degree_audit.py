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

def _is_advanced_csci_course(code: str, advanced_config: dict,
    plan_config: dict,) -> bool:
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

def _calculate_advanced_csci_credits(completed_courses: list, advanced_config: dict,
    plan_config: dict,) -> tuple[float, list[str]]:
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

    limited_confirmed_credits = 0

    for item in completed_courses:
        if isinstance(item, dict):
            code = item["code"].upper().replace(" ", "")
            provided_credits = item.get("credits")
        else:
            code = item.upper().replace(" ", "")
            provided_credits = None

        if not _is_advanced_csci_course(
            code,
            advanced_config,
            plan_config,
        ):
            continue

        # We know these courses can count, but not how many credits
        # the student actually completed from the course code alone.
        if code in limited_courses:
            if provided_credits is None:
                needs_credit_verification.append(code)
                continue

            max_limited_credits = (
                advanced_config
                .get("limited_8xxx", {})
                .get("max_combined_credits", 6)
            )

            remaining_allowed = max_limited_credits - limited_confirmed_credits

            if remaining_allowed > 0:
                credits_to_count = min(provided_credits, remaining_allowed)

                confirmed_credits += credits_to_count
                limited_confirmed_credits += credits_to_count

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

def _normalize_completed_courses(completed_courses: list) -> tuple[list[str], list[dict]]:
    """
    Normalize degree-audit input.

    Supports both legacy course-code strings:
        "CSCI5521"

    and richer course records:
        {
            "code": "STAT5302",
            "credits": 3,
            "degree_approved": True,
        }

    Returns:
        completed_codes: normalized course codes
        course_records: normalized records with code/credits/approval metadata
    """

    completed_codes = []
    course_records = []

    for item in completed_courses:
        if isinstance(item, str):
            code = item.upper().replace(" ", "")

            completed_codes.append(code)
            course_records.append({
                "code": code,
                "credits": None,
                "degree_approved": None,
                "phd_credit_type": None,
            })

        elif isinstance(item, dict):
            raw_code = item.get("code")

            if not raw_code:
                continue

            code = raw_code.upper().replace(" ", "")

            completed_codes.append(code)
            course_records.append({
                "code": code,
                "credits": item.get("credits"),
                "degree_approved": item.get("degree_approved"),
                "phd_credit_type": item.get("phd_credit_type"),
            })

    return completed_codes, course_records

def _calculate_non_csci_degree_credits(course_records: list[dict],) -> tuple[float, list[str], list[str]]:
    """
    Count confirmed approved non-CSCI degree credits.

    Returns:
        confirmed_credits
        pending_approval
        pending_credit_verification
    """

    confirmed_credits = 0
    pending_approval = []
    pending_credit_verification = []

    for record in course_records:
        code = record["code"]

        if code.startswith("CSCI"):
            continue

        credits = record.get("credits")
        approved = record.get("degree_approved")

        if approved is True:
            if credits is None:
                pending_credit_verification.append(code)
            else:
                confirmed_credits += credits

        elif approved is None:
            pending_approval.append(code)

    return (
        confirmed_credits,
        pending_approval,
        pending_credit_verification,
    )

def _calculate_phd_supporting_minor_credits(
    course_records: list[dict],
) -> tuple[float, float, list[str], list[str]]:
    """
    Calculate confirmed Ph.D. supporting-program and minor credits.

    Only courses explicitly marked degree_approved=True count.

    Returns:
        supporting_credits
        minor_credits
        pending_supporting
        pending_minor
    """

    supporting_credits = 0
    minor_credits = 0

    pending_supporting = []
    pending_minor = []

    for record in course_records:
        credit_type = record.get("phd_credit_type")

        if credit_type not in ("supporting", "minor"):
            continue

        code = record["code"]
        credits = record.get("credits")
        approved = record.get("degree_approved")

        if approved is not True or credits is None:
            if credit_type == "supporting":
                pending_supporting.append(code)
            else:
                pending_minor.append(code)

            continue

        if credit_type == "supporting":
            supporting_credits += credits
        else:
            minor_credits += credits

    return (
        supporting_credits,
        minor_credits,
        pending_supporting,
        pending_minor,
    )

def degree_audit(completed_courses: list, program: str = "ms", plan: str | None = None) -> str:
    if program == "ms" and plan not in ("A", "B", "C"):
        return "M.S. degree audits require a plan: A, B, or C."
    if program == "phd":
        plan = None

    completed, course_records = _normalize_completed_courses(completed_courses)

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

        thesis_records = [
            record
            for record in course_records
            if record["code"] == thesis_course
        ]

        confirmed_thesis_credits = sum(
            record["credits"]
            for record in thesis_records
            if record.get("credits") is not None
        )

        has_unknown_thesis_credits = any(
            record.get("credits") is None
            for record in thesis_records
        )

        results.append("PLAN A THESIS REQUIREMENT:")

        if not thesis_records:
            results.append(
                f"  ❌ {thesis_course} not found in provided courses "
                f"({thesis_credits_required} thesis credits required)"
            )

        elif confirmed_thesis_credits >= thesis_credits_required:
            results.append(
                f"  ✅ Confirmed thesis credits: "
                f"{confirmed_thesis_credits}/{thesis_credits_required} required"
            )

        elif has_unknown_thesis_credits:
            results.append(
                f"  ⚠️ Confirmed thesis credits: "
                f"{confirmed_thesis_credits}/{thesis_credits_required}; "
                f"additional {thesis_course} credit value requires verification"
            )

        else:
            thesis_needed = thesis_credits_required - confirmed_thesis_credits

            results.append(
                f"  ❌ Confirmed thesis credits: "
                f"{confirmed_thesis_credits}/{thesis_credits_required}; "
                f"{thesis_needed} more thesis credit(s) required"
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

    phd_thesis_complete = True
    confirmed_phd_thesis_credits = 0
    phd_thesis_credits_required = 0
    has_unknown_phd_thesis_credits = False

    intro_complete = True

    if program == "phd":
    # ── Intro to Research ────────────────────────────────────────────────
        intro = req.get("intro_research")
        intro_complete = intro in completed

        if intro_complete:
            results.append(f"  ✅ Intro to Research ({intro}): complete")
        else:
            results.append(f"  ❌ Intro to Research ({intro}): not completed")

    # ── Ph.D. thesis credits ─────────────────────────────────────────────
        phd_thesis_course = req["thesis_course"]
        phd_thesis_credits_required = req["thesis_credits"]

        thesis_records = [
            record
            for record in course_records
            if record["code"] == phd_thesis_course
        ]

        confirmed_phd_thesis_credits = sum(
            record["credits"]
            for record in thesis_records
            if record.get("credits") is not None
        )

        phd_thesis_complete = (
            confirmed_phd_thesis_credits
            >= phd_thesis_credits_required
        )

        has_unknown_phd_thesis_credits = any(
            record.get("credits") is None
            for record in thesis_records
        )

        results.append("PH.D. THESIS REQUIREMENT:")

        if not thesis_records:
            results.append(
                f"  ❌ {phd_thesis_course} not found in provided courses "
                f"({phd_thesis_credits_required} thesis credits required)"
            )

        elif confirmed_phd_thesis_credits >= phd_thesis_credits_required:
            results.append(
                f"  ✅ Confirmed thesis credits: "
                f"{confirmed_phd_thesis_credits}/{phd_thesis_credits_required} required"
            )

        elif has_unknown_phd_thesis_credits:
            results.append(
                f"  ⚠️ Confirmed thesis credits: "
                f"{confirmed_phd_thesis_credits}/{phd_thesis_credits_required}; "
                f"additional {phd_thesis_course} credit value requires verification"
            )

        else:
            thesis_needed = (
                phd_thesis_credits_required
                - confirmed_phd_thesis_credits
            )

            results.append(
                f"  ❌ Confirmed thesis credits: "
                f"{confirmed_phd_thesis_credits}/{phd_thesis_credits_required}; "
                f"{thesis_needed} more thesis credit(s) required"
            )

    results.append("")

    # ── Credit count ──────────────────────────────────────────────────────────
    csci_credits = 0
    excluded_courses = []

    needs_csci_credit_verification = []

    variable_credit_courses = {"CSCI8991", "CSCI8994"}

    if program == "ms" and plan == "A":
        thesis_course = plan_req.get("thesis_course")

        if thesis_course:
            variable_credit_courses.add(thesis_course)

    phd_thesis_course = None

    if program == "phd":
        phd_thesis_course = req.get("thesis_course")

    for record in course_records:
        code = record["code"]
        provided_credits = record.get("credits")

        if not code.startswith("CSCI"):
            continue

        if program == "phd" and code == phd_thesis_course:
        # Thesis credits are tracked separately from
        # the Ph.D. CSCI coursework requirement.
            continue

        course_number = code[4:]

        if course_number.isdigit() and int(course_number) < 5000:
            excluded_courses.append(
                f"{code} (4xxx-level course)"
            )
            continue

        if code in variable_credit_courses:
            if provided_credits is not None:
                csci_credits += provided_credits
            else:
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
    if program == "ms":
        results.append(
            "  Note: approved non-CSCI coursework is counted separately below"
        )
    else:
        results.append(
            "  Note: non-CSCI coursework is not included in this CSCI subtotal"
        )

    results.append("")

    # Ph.D. 28 course-credit requirement
    phd_course_credits_complete = True
    confirmed_phd_course_credits = 0

    if program == "phd":
        (
            approved_non_csci_credits,
            pending_non_csci_approval,
            pending_non_csci_credit,
        ) = _calculate_non_csci_degree_credits(course_records)

        confirmed_phd_course_credits = (
            csci_credits + approved_non_csci_credits
        )

        required_phd_course_credits = req["course_credits"]

        phd_course_credits_complete = (
            confirmed_phd_course_credits
            >= required_phd_course_credits
        )

        results.append("PH.D. COURSE CREDIT REQUIREMENT:")
        results.append(
            f"  Confirmed course credits: "
            f"{confirmed_phd_course_credits}/"
            f"{required_phd_course_credits} required"
        )

        if approved_non_csci_credits:
            results.append(
                f"  Approved non-CSCI course credits counted: "
                f"{approved_non_csci_credits}"
            )

        if pending_non_csci_approval:
            results.append(
                "  ⚠️ Degree applicability must be verified for: "
                + ", ".join(pending_non_csci_approval)
            )

        if pending_non_csci_credit:
            results.append(
                "  ⚠️ Credit value must be verified for: "
                + ", ".join(pending_non_csci_credit)
            )

    results.append("")

    # Ph.D. supporting/minor requirement
    phd_support_minor_complete = True
    confirmed_supporting_credits = 0
    confirmed_minor_credits = 0

    pending_supporting = []
    pending_minor = []

    if program == "phd":
        (
            confirmed_supporting_credits,
            confirmed_minor_credits,
            pending_supporting,
            pending_minor,
        ) = _calculate_phd_supporting_minor_credits(course_records)

        required_supporting = req["supporting_program_credits"]
        required_minor = req["minor_credits"]

        supporting_complete = (
            confirmed_supporting_credits >= required_supporting
        )

        minor_complete = (
            confirmed_minor_credits >= required_minor
        )

        phd_support_minor_complete = (
            supporting_complete or minor_complete
        )

        results.append("PH.D. SUPPORTING / MINOR REQUIREMENT:")
        results.append(
            f"  Supporting-program credits: "
            f"{confirmed_supporting_credits}/{required_supporting} required"
        )
        results.append(
            f"  Minor credits: "
            f"{confirmed_minor_credits}/{required_minor} required"
        )

        if supporting_complete:
            results.append(
                "  ✅ Supporting-program pathway satisfied"
            )
        elif minor_complete:
            results.append(
                "  ✅ Minor pathway satisfied"
            )
        else:
            results.append(
                "  ❌ Neither pathway is confirmed complete"
            )

        if pending_supporting:
            results.append(
                "  ⚠️ Supporting credits pending verification for: "
                + ", ".join(pending_supporting)
            )

        if pending_minor:
            results.append(
                "  ⚠️ Minor credits pending verification for: "
                + ", ".join(pending_minor)
            )

        results.append("")

    phd_total_credits_complete = True
    confirmed_phd_total_credits = 0

    if program == "phd":
        confirmed_phd_total_credits = (
            confirmed_phd_course_credits
            + confirmed_phd_thesis_credits
        )

        required_phd_total_credits = req["total_credits"]

        phd_total_credits_complete = (
            confirmed_phd_total_credits
            >= required_phd_total_credits
        )

        results.append("PH.D. TOTAL CREDIT REQUIREMENT:")
        results.append(
            f"  Confirmed total credits: "
            f"{confirmed_phd_total_credits}/{required_phd_total_credits} required"
        )

        if pending_non_csci_approval:
            results.append(
                "  ⚠️ Total may increase after degree applicability "
                "is verified for: "
                + ", ".join(pending_non_csci_approval)
            )

        if pending_non_csci_credit:
            results.append(
                "  ⚠️ Total may increase after credit values "
                "are verified for: "
                + ", ".join(pending_non_csci_credit)
            )

        if has_unknown_phd_thesis_credits:
            results.append(
                "  ⚠️ Total may increase after verifying "
                "CSCI8888 thesis credits"
            )

        results.append("")

    # M.S. total-degree-credit requirement
    total_degree_complete = True
    confirmed_degree_credits = 0

    if program == "ms":
        (
            approved_non_csci_credits,
            pending_non_csci_approval,
            pending_non_csci_credit,
        ) = _calculate_non_csci_degree_credits(course_records)

        confirmed_degree_credits = csci_credits + approved_non_csci_credits
        required_total_credits = req["total_credits"]

        total_degree_complete = (
            confirmed_degree_credits >= required_total_credits)

        results.append("TOTAL DEGREE CREDIT REQUIREMENT:")
        results.append(
            f"  Confirmed degree credits: "
            f"{confirmed_degree_credits}/{required_total_credits} required"
        )

        if approved_non_csci_credits:
            results.append(
                f"  Approved non-CSCI credits counted: "
                f"{approved_non_csci_credits}"
            )

        if pending_non_csci_approval:
            results.append(
                "  ⚠️ Degree applicability must be verified for: "
                + ", ".join(pending_non_csci_approval)
            )

        if pending_non_csci_credit:
            results.append(
                "  ⚠️ Credit value must be verified for: "
                + ", ".join(pending_non_csci_credit)
            )

        if needs_csci_credit_verification:
            results.append(
                "  ⚠️ Total may increase after verifying CSCI credits for: "
                + ", ".join(needs_csci_credit_verification)
            )

        results.append("")

    advanced_complete = True
    advanced_credits = 0
    advanced_verify = []

    if program == "ms":
        advanced_config = req["advanced_csci"]

        advanced_credits, advanced_verify = _calculate_advanced_csci_credits(
            course_records,
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
        and total_degree_complete
        and phd_course_credits_complete
        and phd_thesis_complete
        and phd_support_minor_complete
        and phd_total_credits_complete
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

        if program == "phd" and not phd_course_credits_complete:
            phd_course_needed = (
                required_phd_course_credits
                - confirmed_phd_course_credits
            )

            if pending_non_csci_approval or pending_non_csci_credit:
                pending_phd_verification = (
                    pending_non_csci_approval
                    + pending_non_csci_credit
                )

                missing.append(
                    f"{phd_course_needed:g} more confirmed Ph.D. course credit(s) "
                    f"(pending verification for: "
                    f"{', '.join(pending_phd_verification)})"
                )
            else:
                missing.append(
                    f"{phd_course_needed:g} more Ph.D. course credit(s)"
                )

        if program == "phd" and not phd_support_minor_complete:
            supporting_needed = (
                required_supporting
                - confirmed_supporting_credits
            )

            minor_needed = (
                required_minor
                - confirmed_minor_credits
            )

            missing.append(
                f"Ph.D. supporting/minor requirement: "
                f"{supporting_needed:g} more supporting-program credit(s) "
                f"OR {minor_needed:g} more minor credit(s)"
            )

        if program == "phd" and not phd_thesis_complete:
            thesis_needed = (
                phd_thesis_credits_required
                - confirmed_phd_thesis_credits
            )

            if has_unknown_phd_thesis_credits:
                missing.append(
                    f"{thesis_needed:g} more confirmed CSCI8888 thesis credit(s) "
                    f"(additional thesis credit value requires verification)"
                )

            else:
                missing.append(
                    f"{thesis_needed:g} more CSCI8888 thesis credit(s)"
                )

        if program == "ms" and not total_degree_complete:
            total_needed = required_total_credits - confirmed_degree_credits

            pending_total_verification = (
                pending_non_csci_approval
                + pending_non_csci_credit
                + needs_csci_credit_verification
            )

            if pending_total_verification:
                missing.append(
                    f"{total_needed:g} more confirmed degree credit(s) "
                    f"(pending verification for: "
                    f"{', '.join(pending_total_verification)})"
                )
            else:
                missing.append(
                    f"{total_needed:g} more degree credit(s)"
                )
        results.append(f"  Still needed: {'; '.join(missing)}")

    return "\n".join(results)


if __name__ == "__main__":
    test_courses = ["CSCI5521", "CSCI5103", "CSCI8970"]
    print(degree_audit(test_courses, program="ms"))