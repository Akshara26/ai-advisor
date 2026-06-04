"""
Unit tests for deterministic tools: get_deadline, route_contact

Run with: pytest tests/test_tools.py -v
"""
import pytest
from unittest.mock import patch

MOCK_DEADLINES = {
    "verification_instruction": "Verify exact dates at https://onestop.umn.edu/calendar",
    "process_deadlines": [
        {
            "process": "graduation_application",
            "label": "Apply to Graduate",
            "typical_timing": "First month of intended graduation term",
            "where_to_check": "https://onestop.umn.edu/academics/graduation-steps",
            "notes": "Graduate students must submit month of intended graduation via MyU.",
        },
        {
            "process": "cpt_application_lead_time",
            "label": "CPT Application Lead Time (ISSS)",
            "typical_timing": "At least 3-4 weeks before employment start date",
            "where_to_check": "https://isss.umn.edu/work/cpt",
            "notes": "CRITICAL: Allow extra time for ISSS processing.",
        },
    ],
}

MOCK_ROUTING = {
    "graduation_eligibility": {
        "issue_id": "graduation_eligibility",
        "label": "Whether student is on track to graduate",
        "office_ids": ["cs_grad"],
        "escalation_level": "hard",
        "bot_can_answer": False,
        "bot_answer_note": "Provide preliminary checklist only.",
        "offer_email_draft": True,
    },
    "prerequisite_check": {
        "issue_id": "prerequisite_check",
        "label": "Whether student meets prerequisites",
        "office_ids": [],
        "escalation_level": "auto",
        "bot_can_answer": True,
        "bot_answer_note": "Answer directly using prerequisite tool.",
        "offer_email_draft": False,
    },
}

MOCK_OFFICES = {
    "cs_grad": {
        "id": "cs_grad",
        "name": "CS Graduate Program Office",
        "email": "csgradmn@umn.edu",
        "url": "https://cse.umn.edu/cs/graduate",
    },
}


@pytest.fixture(autouse=True)
def patch_tool_data():
    with patch("tools.DEADLINES_DATA", MOCK_DEADLINES), \
         patch("tools.ROUTING_DATA", MOCK_ROUTING), \
         patch("tools.OFFICES_DATA", MOCK_OFFICES):
        yield


from tools import get_deadline, route_contact


class TestGetDeadline:

    def test_known_process_returns_timing(self):
        result = get_deadline("graduation_application")
        assert "Apply to Graduate" in result
        assert "First month" in result

    def test_known_process_includes_verification_warning(self):
        result = get_deadline("graduation_application")
        assert "onestop.umn.edu/calendar" in result
        assert "Verify" in result

    def test_partial_match_works(self):
        # "graduation" should match "graduation_application"
        result = get_deadline("graduation")
        assert "Apply to Graduate" in result

    def test_cpt_deadline_returns_critical_note(self):
        result = get_deadline("cpt_application_lead_time")
        assert "ISSS" in result
        assert "weeks" in result.lower()

    def test_unknown_process_returns_fallback(self):
        result = get_deadline("nonexistent_process_xyz")
        assert "onestop.umn.edu/calendar" in result
        assert "not found" in result.lower() or "check" in result.lower()


class TestRouteContact:

    def test_known_issue_returns_office(self):
        result = route_contact("graduation_eligibility")
        assert "CS Graduate Program Office" in result
        assert "csgradmn@umn.edu" in result

    def test_known_issue_returns_escalation_level(self):
        result = route_contact("graduation_eligibility")
        assert "hard" in result.lower()

    def test_auto_escalation_issue(self):
        result = route_contact("prerequisite_check")
        assert "auto" in result.lower()

    def test_fuzzy_match_on_label(self):
        # "graduate" should fuzzy-match "graduation_eligibility" label
        result = route_contact("on track to graduate")
        assert "CS Graduate Program Office" in result or "csgradmn" in result

    def test_unknown_issue_returns_default(self):
        result = route_contact("some_completely_unknown_issue_xyz")
        assert "csgradmn@umn.edu" in result
