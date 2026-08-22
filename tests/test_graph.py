"""
Unit tests for parse_state_block, clean_response, and parse_email_block.

These are pure parsing functions — no LLM, no DB, no network.
They are the most brittle part of the advisor pipeline because
the entire routing decision depends on parse_state_block being correct.

Run with: pytest tests/test_graph.py -v
"""
from curses import meta

import pytest
from advisor.graph import parse_email_block, AdvisorMeta

class TestAdvisorMeta:

    def test_valid_high_confidence(self):
        meta = AdvisorMeta(answered=True, confidence="high", question_type="policy", needs_clarification=False,)
        assert meta.answered is True
        assert meta.confidence == "high"

    def test_valid_low_confidence(self):
        meta = AdvisorMeta(answered=False, confidence="low", question_type="unknown", needs_clarification=False,)
        assert meta.answered is False

    def test_invalid_confidence_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AdvisorMeta(answered=True, confidence="very_high", question_type="policy")

    def test_invalid_question_type_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AdvisorMeta(answered=True, confidence="high", question_type="immigration")

    def test_needs_clarification(self):
        meta = AdvisorMeta(
            answered=False,
            confidence="low",
            question_type="unknown",
            needs_clarification=True,
        )

        assert meta.needs_clarification is True


# ── parse_state_block: valid input ────────────────────────────────────────────

# class TestParseStateBlockValid:

#     def test_parses_complete_valid_block(self):
#         response = """Here is my answer to the question.

# ---STATE---
# {
#     "answered": true,
#     "confidence": "high",
#     "question_type": "policy",
#     "reason": "Direct handbook citation supports the answer."
# }
# ---END STATE---"""
#         data, failed = parse_state_block(response)
#         assert failed is False
#         assert data["answered"] is True
#         assert data["confidence"] == "high"
#         assert data["question_type"] == "policy"

#     def test_parses_block_at_end_of_longer_response(self):
#         response = (
#             "The minimum GPA requirement is 3.0 [Handbook p.8]. "
#             "Students who fall below this are placed on academic probation.\n\n"
#             "---STATE---\n"
#             '{"answered": true, "confidence": "high", "question_type": "policy", "reason": "Handbook citation."}\n'
#             "---END STATE---"
#         )
#         data, failed = parse_state_block(response)
#         assert failed is False
#         assert data["answered"] is True

#     def test_all_confidence_values_parsed(self):
#         for confidence in ("high", "medium", "low", "none"):
#             response = (
#                 f"---STATE---\n"
#                 f'{{"answered": true, "confidence": "{confidence}", '
#                 f'"question_type": "policy", "reason": "test"}}\n'
#                 f"---END STATE---"
#             )
#             data, failed = parse_state_block(response)
#             assert failed is False
#             assert data["confidence"] == confidence

#     def test_all_question_types_parsed(self):
#         for qtype in ("policy", "personal", "degree_audit", "deadline",
#                       "procedure", "course_prerequisite", "unknown"):
#             response = (
#                 f"---STATE---\n"
#                 f'{{"answered": true, "confidence": "high", '
#                 f'"question_type": "{qtype}", "reason": "test"}}\n'
#                 f"---END STATE---"
#             )
#             data, failed = parse_state_block(response)
#             assert failed is False
#             assert data["question_type"] == qtype

#     def test_answered_false_parsed(self):
#         response = (
#             "---STATE---\n"
#             '{"answered": false, "confidence": "low", '
#             '"question_type": "unknown", "reason": "Missing context."}\n'
#             "---END STATE---"
#         )
#         data, failed = parse_state_block(response)
#         assert failed is False
#         assert data["answered"] is False
#         assert data["confidence"] == "low"


# # ── parse_state_block: missing block ─────────────────────────────────────────

# class TestParseStateBlockMissing:

#     def test_missing_block_returns_parse_failed_true(self):
#         _, failed = parse_state_block("Here is my answer with no state block.")
#         assert failed is True

#     def test_missing_block_returns_answered_false(self):
#         data, _ = parse_state_block("No state block here.")
#         assert data["answered"] is False

#     def test_missing_block_returns_confidence_none(self):
#         data, _ = parse_state_block("No state block here.")
#         assert data["confidence"] == "none"

#     def test_empty_string_returns_parse_failed(self):
#         data, failed = parse_state_block("")
#         assert failed is True
#         assert data["answered"] is False

#     def test_partial_delimiter_not_matched(self):
#         # Only opening delimiter, no closing — should not parse
#         _, failed = parse_state_block('---STATE---\n{"answered": true}\n')
#         assert failed is True

#     def test_wrong_delimiter_not_matched(self):
#         # Wrong format should not parse
#         _, failed = parse_state_block('<state>{"answered": true}</state>')
#         assert failed is True


# ── parse_state_block: malformed JSON ────────────────────────────────────────

# class TestParseStateBlockMalformed:

#     def test_malformed_json_returns_parse_failed_true(self):
#         response = (
#             "---STATE---\n"
#             '{"answered": true, "confidence": "high", INVALID JSON}\n'
#             "---END STATE---"
#         )
#         _, failed = parse_state_block(response)
#         assert failed is True

#     def test_malformed_json_returns_answered_false(self):
#         response = (
#             "---STATE---\n"
#             "{not valid json at all\n"
#             "---END STATE---"
#         )
#         data, failed = parse_state_block(response)
#         assert failed is True
#         assert data["answered"] is False

#     def test_malformed_json_returns_confidence_none(self):
#         response = (
#             "---STATE---\n"
#             "answered: true\n"  # YAML-style, not JSON
#             "---END STATE---"
#         )
#         data, failed = parse_state_block(response)
#         assert failed is True
#         assert data["confidence"] == "none"

#     def test_empty_block_returns_parse_failed(self):
#         response = "---STATE---\n\n---END STATE---"
#         _, failed = parse_state_block(response)
#         assert failed is True


# ── clean_response ────────────────────────────────────────────────────────────

# class TestCleanResponse:

#     def test_removes_state_block(self):
#         response = (
#             "Here is the answer.\n\n"
#             "---STATE---\n"
#             '{"answered": true, "confidence": "high"}\n'
#             "---END STATE---"
#         )
#         cleaned = clean_response(response)
#         assert "---STATE---" not in cleaned
#         assert "---END STATE---" not in cleaned
#         assert "Here is the answer." in cleaned

#     def test_no_state_block_unchanged(self):
#         response = "Here is the answer with no state block."
#         cleaned = clean_response(response)
#         assert cleaned == response

#     def test_strips_trailing_whitespace(self):
#         response = "Answer text.\n\n---STATE---\n{}\n---END STATE---\n\n"
#         cleaned = clean_response(response)
#         assert not cleaned.endswith("\n")

#     def test_empty_string_returns_empty(self):
#         assert clean_response("") == ""


# ── parse_email_block ─────────────────────────────────────────────────────────

class TestParseEmailBlock:

    def test_parses_valid_email_block(self):
        response = (
            "---EMAIL---\n"
            "Subject: Question about MS Plan B requirements\n\n"
            "I am writing to ask about whether CSCI 5980 can count toward breadth.\n"
            "---END EMAIL---"
        )
        result = parse_email_block(response)
        assert "Subject:" in result
        assert "CSCI 5980" in result
        assert "---EMAIL---" not in result
        assert "---END EMAIL---" not in result

    def test_no_email_block_returns_full_text(self):
        response = "Here is a draft email you can send."
        result = parse_email_block(response)
        assert result == response

    def test_empty_string_returns_empty(self):
        assert parse_email_block("") == ""