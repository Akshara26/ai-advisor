"""
Unit tests for degree_audit.py

Run with: pytest tests/test_degree_audit.py -v
"""
import pytest
from unittest.mock import patch

# Minimal mock data — independent of the real data/courses.json
MOCK_REQUIREMENTS = {
    "ms": {
        "total_credits": 31,
        "csci_credits": 16,
        "colloquium": "CSCI8970",
        "breadth_categories": {
            "applications":                ["CSCI5521", "CSCI5523"],
            "theory_and_algorithms":       ["CSCI5421", "CSCI5525"],
            "architecture_systems_software": ["CSCI5103", "CSCI5801"],
        },
    },
    "phd": {
        "total_credits": 52,
        "csci_credits": 16,
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
    "CSCI8001": {"code": "CSCI8001", "cred_min": 3},
}


@pytest.fixture(autouse=True)
def patch_data():
    """Patch module-level data so tests never depend on real JSON files."""
    with patch("degree_audit.REQUIREMENTS", MOCK_REQUIREMENTS), \
         patch("degree_audit.code_to_course", MOCK_CODE_TO_COURSE):
        yield


from degree_audit import degree_audit


# ── M.S. breadth and colloquium ───────────────────────────────────────────────

class TestMSBreadthAndColloquium:

    def test_all_requirements_met(self):
        courses = ["CSCI5521", "CSCI5421", "CSCI5801", "CSCI8970"]
        result = degree_audit(courses, "ms")
        assert "✅" in result
        assert "❌" not in result

    def test_missing_theory_breadth(self):
        courses = ["CSCI5521", "CSCI5801", "CSCI8970"]
        result = degree_audit(courses, "ms")
        assert "❌ Theory And Algorithms" in result

    def test_missing_architecture_breadth(self):
        courses = ["CSCI5521", "CSCI5421", "CSCI8970"]
        result = degree_audit(courses, "ms")
        assert "❌ Architecture Systems Software" in result

    def test_missing_colloquium_only(self):
        # All breadth satisfied, colloquium missing
        courses = ["CSCI5521", "CSCI5421", "CSCI5801"]
        result = degree_audit(courses, "ms")
        assert "✅ Applications" in result
        assert "✅ Theory And Algorithms" in result
        assert "✅ Architecture Systems Software" in result
        assert "❌ Colloquium" in result

    def test_empty_course_list(self):
        result = degree_audit([], "ms")
        assert result.count("❌") >= 4  # all three breadth areas + colloquium missing

    def test_no_csci_courses(self):
        result = degree_audit(["STAT5302", "MATH5651"], "ms")
        assert "❌ Applications" in result
        assert "❌ Colloquium" in result


# ── Course code normalization ─────────────────────────────────────────────────

class TestCourseNormalization:

    def test_codes_with_spaces_are_normalized(self):
        # "CSCI 5521" should be treated as "CSCI5521"
        result = degree_audit(["CSCI 5521", "CSCI 5421", "CSCI 5801", "CSCI 8970"], "ms")
        assert "✅ Applications" in result
        assert "✅ Colloquium" in result

    def test_lowercase_codes_are_normalized(self):
        result = degree_audit(["csci5521", "csci5421", "csci5801", "csci8970"], "ms")
        assert "✅ Applications" in result

    def test_duplicate_courses_dont_double_count_breadth(self):
        # Two applications courses should still only satisfy Applications once
        result = degree_audit(["CSCI5521", "CSCI5523", "CSCI5421", "CSCI5801", "CSCI8970"], "ms")
        assert result.count("✅ Applications") == 1


# ── Ph.D. specific ────────────────────────────────────────────────────────────

class TestPhD:

    def test_phd_requires_intro_research(self):
        courses = ["CSCI5521", "CSCI5421", "CSCI5801", "CSCI8970"]
        result = degree_audit(courses, "phd")
        assert "❌ Intro to Research" in result

    def test_phd_with_all_requirements(self):
        courses = ["CSCI5521", "CSCI5421", "CSCI5801", "CSCI8970", "CSCI8001"]
        result = degree_audit(courses, "phd")
        assert "✅ Intro to Research" in result
        assert "✅ Colloquium" in result

    def test_phd_missing_colloquium(self):
        courses = ["CSCI5521", "CSCI5421", "CSCI5801", "CSCI8001"]
        result = degree_audit(courses, "phd")
        assert "❌ Colloquium" in result


# ── Credit counting ───────────────────────────────────────────────────────────

class TestCreditCounting:

    def test_csci_credits_summed_correctly(self):
        # 3+3+3+1 = 10 CSCI credits
        courses = ["CSCI5521", "CSCI5421", "CSCI5801", "CSCI8970"]
        result = degree_audit(courses, "ms")
        assert "10/16" in result

    def test_non_csci_courses_not_counted(self):
        # STAT5302 is not in MOCK_CODE_TO_COURSE, should not add credits
        courses = ["CSCI5521", "STAT5302", "CSCI8970"]
        result = degree_audit(courses, "ms")
        assert "4/16" in result  # CSCI5521 (3) + CSCI8970 (1) = 4


# ── Error handling ────────────────────────────────────────────────────────────

class TestErrorHandling:

    def test_unknown_program_returns_error_message(self):
        result = degree_audit(["CSCI5521"], "mcs")
        assert "Unknown program" in result or "unknown" in result.lower()


# ── Disclaimer wording ────────────────────────────────────────────────────────

class TestDisclaimerWording:

    def test_summary_does_not_claim_official_clearance(self):
        """The summary must never imply the student is cleared to graduate."""
        courses = ["CSCI5521", "CSCI5421", "CSCI5801", "CSCI8970"]
        result = degree_audit(courses, "ms")
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
