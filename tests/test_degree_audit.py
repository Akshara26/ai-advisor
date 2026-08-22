"""
Unit tests for degree_audit.py

Run with: pytest tests/test_degree_audit.py -v
"""
from unittest import result

import pytest
from unittest.mock import patch

# Minimal mock data — independent of the real data/courses.json
MOCK_REQUIREMENTS = {
    "ms": {
        "total_credits": 31,
        "csci_credits": 16,
        "colloquium": "CSCI8970",
        "advanced_csci": {
            "approved_5xxx": [
                "CSCI5105",
                "CSCI5125",
                "CSCI5127W",
                "CSCI5161",
                "CSCI5204",
                "CSCI5521",
                "CSCI5525",
                "CSCI5527",
                "CSCI5552",
                "CSCI5561",
                "CSCI5608",
                "CSCI5708",
                "CSCI5715",
                "CSCI5802",
            ],
            "allow_any_8xxx": True,
            "excluded_8xxx": [
                "CSCI8001",
                "CSCI8002",
                "CSCI8970",
            ],
            "limited_8xxx": {
                "courses": ["CSCI8991", "CSCI8994"],
                "max_combined_credits": 6,
            },
        },
        "plans": {
            "A": {
                "advanced_csci_credits": 6,
                "thesis_course": "CSCI8777",
                "thesis_credits": 10,
                "extra_advanced_exclusions": ["CSCI8777"],
            },
            "B": {
                "advanced_csci_credits": 3,
                "project_course": "CSCI8760",
                "project_credits": 3,
            },
            "C": {
                "advanced_csci_credits": 6,
                "project_hours": 100,
            },
        },
        "breadth_categories": {
                "applications":                ["CSCI5521", "CSCI5523"],
                "theory_and_algorithms":       ["CSCI5421", "CSCI5525"],
                "architecture_systems_software": ["CSCI5103", "CSCI5801"],
            },
        },
    "phd": {
        "total_credits": 52,
        "csci_credits": 16,
        "required_breadth_courses": 4,
        "colloquium": "CSCI8970",
        "intro_research": "CSCI8001",
        "breadth_categories": {
            "applications":                ["CSCI5521", "CSCI5523"],
            "theory_and_algorithms":       ["CSCI5421", "CSCI5525"],
            "architecture_systems_software": ["CSCI5103", "CSCI5801"],
        },
    },
}

MOCK_CODE_TO_COURSE = {
    "CSCI5521": {"code": "CSCI5521", "cred_min": 3},
    "CSCI5421": {"code": "CSCI5421", "cred_min": 3},
    "CSCI5801": {"code": "CSCI5801", "cred_min": 3},
    "CSCI5103": {"code": "CSCI5103", "cred_min": 3},
    "CSCI8970": {"code": "CSCI8970", "cred_min": 1},
    "CSCI8001": {"code": "CSCI8001", "cred_min": 1},
    "CSCI8760": {"code": "CSCI8760", "cred_min": 3},
    "CSCI5511": {"code": "CSCI5511", "cred_min": 3},
}


@pytest.fixture(autouse=True)
def patch_data():
    """Patch module-level data so tests never depend on real JSON files."""
    with patch("advisor.degree_audit.REQUIREMENTS", MOCK_REQUIREMENTS), \
         patch("advisor.degree_audit.code_to_course", MOCK_CODE_TO_COURSE):
        yield


from advisor.degree_audit import degree_audit


# ── M.S. breadth and colloquium ───────────────────────────────────────────────

class TestMSBreadthAndColloquium:

    def test_all_requirements_met(self):
        courses = [
            "CSCI5521",
            "CSCI5421",
            "CSCI5801",
            "CSCI5511",
            "CSCI8970",
            "CSCI8760",
        ]
        result = degree_audit(courses, "ms", plan="B")
        assert "✅" in result


    def test_missing_theory_breadth(self):
        courses = ["CSCI5521", "CSCI5801", "CSCI8970"]
        result = degree_audit(courses, "ms", plan="B")
        assert "❌ Theory And Algorithms" in result

    def test_missing_architecture_breadth(self):
        courses = ["CSCI5521", "CSCI5421", "CSCI8970"]
        result = degree_audit(courses, "ms", plan="B")
        assert "❌ Architecture Systems Software" in result

    def test_missing_colloquium_only(self):
        # All breadth satisfied, colloquium missing
        courses = ["CSCI5521", "CSCI5421", "CSCI5801"]
        result = degree_audit(courses, "ms", plan="B")
        assert "✅ Applications" in result
        assert "✅ Theory And Algorithms" in result
        assert "✅ Architecture Systems Software" in result
        assert "❌ Colloquium" in result

    def test_empty_course_list(self):
        result = degree_audit([], "ms", plan="B")
        assert result.count("❌") >= 4  # all three breadth areas + colloquium missing

    def test_no_csci_courses(self):
        result = degree_audit(["STAT5302", "MATH5651"], "ms", plan="B")
        assert "❌ Applications" in result
        assert "❌ Colloquium" in result

    def test_ms_requires_plan(self):
        result = degree_audit(["CSCI5521"], "ms")
        assert result == "M.S. degree audits require a plan: A, B, or C."


# ── Course code normalization ─────────────────────────────────────────────────

class TestCourseNormalization:

    def test_codes_with_spaces_are_normalized(self):
        # "CSCI 5521" should be treated as "CSCI5521"
        result = degree_audit(["CSCI 5521", "CSCI 5421", "CSCI 5801", "CSCI 8970"], "ms", plan="B")
        assert "✅ Applications" in result
        assert "✅ Colloquium" in result

    def test_lowercase_codes_are_normalized(self):
        result = degree_audit(["csci5521", "csci5421", "csci5801", "csci8970"], "ms", plan="B")
        assert "✅ Applications" in result

    def test_duplicate_courses_dont_double_count_breadth(self):
        # Two applications courses should still only satisfy Applications once
        result = degree_audit(["CSCI5521", "CSCI5523", "CSCI5421", "CSCI5801", "CSCI8970"], "ms", plan="B")
        assert result.count("✅ Applications") == 1


# ── Ph.D. specific ────────────────────────────────────────────────────────────

class TestPhD:

    def test_phd_requires_intro_research(self):
        courses = ["CSCI5521", "CSCI5421", "CSCI5801", "CSCI8970"]
        result = degree_audit(courses, "phd")
        assert "❌ Intro to Research" in result

    def test_phd_intro_research_complete(self):
        courses = ["CSCI5521", "CSCI5421", "CSCI5801", "CSCI8970", "CSCI8001"]
        result = degree_audit(courses, "phd")
        assert "✅ Intro to Research" in result
        assert "✅ Colloquium" in result

    def test_phd_missing_colloquium(self):
        courses = ["CSCI5521", "CSCI5421", "CSCI5801", "CSCI8001"]
        result = degree_audit(courses, "phd")
        assert "❌ Colloquium" in result

    def test_phd_requires_four_breadth_courses(self):
        courses = [
            "CSCI5521",  # Applications
            "CSCI5421",  # Theory
            "CSCI5801",  # Architecture
            "CSCI8970",
            "CSCI8001",
        ]

        result = degree_audit(courses, "phd")

        assert "Breadth courses completed: 3 of 4 required" in result
        assert "1 additional breadth course(s) from any area" in result


# ── Credit counting ───────────────────────────────────────────────────────────

class TestCreditCounting:

    def test_csci_credits_summed_correctly(self):
        # 3+3+3+1 = 10 CSCI credits
        courses = ["CSCI5521", "CSCI5421", "CSCI5801", "CSCI8970"]
        result = degree_audit(courses, "ms", plan="B")
        assert "10/16" in result

    def test_non_csci_courses_not_counted(self):
        # STAT5302 is not in MOCK_CODE_TO_COURSE, should not add credits
        courses = ["CSCI5521", "STAT5302", "CSCI8970"]
        result = degree_audit(courses, "ms", plan="B")
        assert "4/16" in result  # CSCI5521 (3) + CSCI8970 (1) = 4

    def test_plan_c_reports_missing_advanced_credits(self):
        courses = [
            "CSCI5511",
            "CSCI5421",
            "CSCI5801",
            "CSCI5103",
            "CSCI8970",
        ]

        result = degree_audit(courses, "ms", plan="C")

        assert "Confirmed advanced CSCI credits: 0/6 required" in result
        assert "6 more advanced CSCI credit(s)" in result

    def test_plan_b_project_course_counts_as_advanced(self):
        courses = [
            "CSCI5521",
            "CSCI5421",
            "CSCI5801",
            "CSCI5511",
            "CSCI8970",
            "CSCI8760",
        ]

        result = degree_audit(courses, "ms", plan="B")

        assert "Confirmed advanced CSCI credits: 6/3 required" in result

    def test_variable_credit_course_not_assumed_as_three_credits(self):
        courses = [
            "CSCI5511",
            "CSCI5421",
            "CSCI5801",
            "CSCI5103",
            "CSCI8970",
            "CSCI8991",
        ]

        result = degree_audit(courses, "ms", plan="C")

        assert "CSCI credits completed: 13/16 required" in result
        assert "Credit value must be verified for: CSCI8991" in result
        assert "3 more confirmed CSCI credit(s)" in result

    def test_plan_a_thesis_does_not_count_as_advanced(self):
        courses = [
            "CSCI8777",
            "CSCI5521",
        ]
        result = degree_audit(courses, "ms", plan="A")
        assert "Confirmed advanced CSCI credits: 3/6 required" in result


# # ── Error handling ────────────────────────────────────────────────────────────

# class TestErrorHandling:

#     def test_unknown_program_returns_error_message(self):
#         result = degree_audit(["CSCI5521"], "mcs")
#         assert "Unknown program" in result or "unknown" in result.lower()


# ── Disclaimer wording ────────────────────────────────────────────────────────

class TestDisclaimerWording:

    def test_summary_does_not_claim_official_clearance(self):
        """The summary must never imply the student is cleared to graduate."""
        courses = ["CSCI5521", "CSCI5421", "CSCI5801", "CSCI8970"]
        result = degree_audit(courses, "ms", plan="B")
        forbidden_phrases = [
            "you are cleared",
            "you are eligible",
            "you will graduate",
            "graduation approved",
            "officially",
        ]
        result_lower = result.lower()
        for phrase in forbidden_phrases:
            assert phrase not in result_lower, (
                f"Degree audit output contains '{phrase}' which implies official clearance. "
                "Reword to make clear this is a preliminary checklist only."
            )
