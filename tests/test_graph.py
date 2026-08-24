"""
Unit tests for parse_state_block, clean_response, and parse_email_block.

These are pure parsing functions — no LLM, no DB, no network.
They are the most brittle part of the advisor pipeline because
the entire routing decision depends on parse_state_block being correct.

Run with: pytest tests/test_graph.py -v
"""
from curses import meta

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from advisor.graph import parse_email_block, AdvisorMeta, advisor_node, chat

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

class TestDegreeAuditGraphIntegration:

    def test_degree_audit_is_followed_by_handbook_search(self):
        # First model response: request degree_audit
        degree_audit_call = SimpleNamespace(
            id="call_degree_audit",
            type="function",
            function=SimpleNamespace(
                name="degree_audit",
                arguments=(
                    '{"completed_courses":["CSCI5521","CSCI5421",'
                    '"CSCI5801","CSCI8970","CSCI8760"],'
                    '"program":"ms","plan":"B"}'
                ),
            ),
        )

        first_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[degree_audit_call],
                    )
                )
            ]
        )

        # Second model response tries to answer immediately.
        # The graph should reject this and force search_handbook.
        premature_answer = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Your degree audit is complete.",
                        tool_calls=None,
                    )
                )
            ]
        )

        # Third model response: request search_handbook
        handbook_call = SimpleNamespace(
            id="call_handbook",
            type="function",
            function=SimpleNamespace(
                name="search_handbook",
                arguments='{"query":"MS Plan B degree requirements"}',
            ),
        )

        handbook_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[handbook_call],
                    )
                )
            ]
        )

        # Fourth model response: final synthesized answer
        final_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "Based on your degree audit and the handbook, "
                            "you still need additional degree credits."
                        ),
                        tool_calls=None,
                    )
                )
            ]
        )

        # Separate structured-output classifier response
        meta_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        parsed=AdvisorMeta(
                            answered=True,
                            needs_clarification=False,
                            confidence="high",
                            question_type="degree_audit",
                        )
                    )
                )
            ]
        )

        fake_client = MagicMock()

        fake_client.chat.completions.create.side_effect = [
            first_response,
            premature_answer,
            handbook_response,
            final_response,
        ]

        fake_client.beta.chat.completions.parse.return_value = meta_response

        def fake_run_tool(tool_name, tool_args):
            if tool_name == "degree_audit":
                return (
                    "DEGREE AUDIT RESULT: "
                    "Plan B requirements checked; additional credits remain."
                )

            if tool_name == "search_handbook":
                return (
                    "HANDBOOK RESULT: "
                    "M.S. Plan B requires 31 total credits."
                )

            raise AssertionError(f"Unexpected tool called: {tool_name}")

        with patch("advisor.graph.client", fake_client), \
             patch("advisor.graph.run_tool", side_effect=fake_run_tool) as mock_run_tool, \
             patch("advisor.graph.check_hard_escalation", return_value=None):

            initial_state = {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "I'm an M.S. Plan B student. "
                            "I completed CSCI 5521, CSCI 5421, CSCI 5801, "
                            "CSCI 8970, and CSCI 8760. "
                            "Can you run a degree audit?"
                        ),
                    }
                ],
                "answer": "",
                "answered": False,
                "needs_clarification": False,
                "confidence": "none",
                "question_type": "unknown",
                "tools_tried": [],
                "drafted_email": "",
                "tool_contexts": [],
                "escalation_office": "",
            }

            result = advisor_node(initial_state)

        # Final answer came from the last LLM response
        assert (
            "Based on your degree audit and the handbook"
            in result["answer"]
)

        # Degree audit must run first, handbook search second
        called_tools = [
            call.args[0]
            for call in mock_run_tool.call_args_list
        ]

        degree_audit_args = mock_run_tool.call_args_list[0].args[1]

        assert degree_audit_args["program"] == "ms"
        assert degree_audit_args["plan"] == "B"

        assert degree_audit_args["completed_courses"] == [
            "CSCI5521",
            "CSCI5421",
            "CSCI5801",
            "CSCI8970",
            "CSCI8760",
        ]

        assert called_tools == [
            "degree_audit",
            "search_handbook",
        ]

        # Verify the handbook search is about Plan B
        handbook_args = mock_run_tool.call_args_list[1].args[1]

        assert "Plan B" in handbook_args["query"]

        # Both tool results should be exposed by chat()
        assert len(result["tool_contexts"]) == 2
        assert "DEGREE AUDIT RESULT" in result["tool_contexts"][0]
        assert "HANDBOOK RESULT" in result["tool_contexts"][1]

        assert result["answered"] is True
        assert result["confidence"] == "high"
        assert result["question_type"] == "degree_audit"

        # # No email drafting should occur
        # assert result["drafted_email"] == ""

        # # Conversation history should contain user + assistant
        # assert len(result["messages"]) == 2
        # assert result["messages"][0]["role"] == "user"
        # assert result["messages"][1]["role"] == "assistant"


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


class TestChatWrapper:

    def test_chat_builds_state_and_returns_graph_result(self):
        existing_history = [
            {
                "role": "user",
                "content": "What are the M.S. breadth requirements?",
            },
            {
                "role": "assistant",
                "content": "There are three breadth areas.",
            },
        ]

        fake_graph_result = {
            "answer": "Here is your degree audit.",
            "drafted_email": "",
            "tool_contexts": [
                "DEGREE AUDIT RESULT",
                "HANDBOOK RESULT",
            ],
        }

        with patch(
            "advisor.graph.advisor_graph.invoke",
            return_value=fake_graph_result,
        ) as mock_invoke:

            answer, updated_history, drafted_email, tool_contexts = chat(
                "Can you audit my completed courses?",
                existing_history,
            )

        # ── Return values ────────────────────────────────
        assert answer == "Here is your degree audit."
        assert drafted_email == ""

        assert tool_contexts == [
            "DEGREE AUDIT RESULT",
            "HANDBOOK RESULT",
        ]

        # ── Conversation history ─────────────────────────
        assert len(updated_history) == 4

        assert updated_history[-2] == {
            "role": "user",
            "content": "Can you audit my completed courses?",
        }

        assert updated_history[-1] == {
            "role": "assistant",
            "content": "Here is your degree audit.",
        }

        # ── State passed into LangGraph ──────────────────
        mock_invoke.assert_called_once()

        initial_state = mock_invoke.call_args.args[0]

        assert initial_state["messages"] == [
            *existing_history,
            {
                "role": "user",
                "content": "Can you audit my completed courses?",
            },
        ]

        assert initial_state["answer"] == ""
        assert initial_state["answered"] is False
        assert initial_state["needs_clarification"] is False
        assert initial_state["confidence"] == "none"
        assert initial_state["question_type"] == "unknown"
        assert initial_state["tools_tried"] == []
        assert initial_state["drafted_email"] == ""
        assert initial_state["tool_contexts"] == []
        assert initial_state["escalation_office"] == ""


class TestMSDegreeAuditClarification:

    def test_ms_degree_audit_without_plan_asks_for_clarification(self):
        clarification_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "Before I run the degree audit, are you in "
                            "M.S. Plan A, Plan B, or Plan C?"
                        ),
                        tool_calls=None,
                    )
                )
            ]
        )

        meta_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        parsed=AdvisorMeta(
                            answered=False,
                            needs_clarification=True,
                            confidence="high",
                            question_type="degree_audit",
                        )
                    )
                )
            ]
        )

        fake_client = MagicMock()

        fake_client.chat.completions.create.return_value = (
            clarification_response
        )

        fake_client.beta.chat.completions.parse.return_value = (
            meta_response
        )

        initial_state = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "I'm an M.S. student. I completed CSCI 5521, "
                        "CSCI 5421, CSCI 5801, and CSCI 8970. "
                        "Can you run a degree audit?"
                    ),
                }
            ],
            "answer": "",
            "answered": False,
            "needs_clarification": False,
            "confidence": "none",
            "question_type": "unknown",
            "tools_tried": [],
            "drafted_email": "",
            "tool_contexts": [],
            "escalation_office": "",
        }

        with patch("advisor.graph.client", fake_client), \
             patch("advisor.graph.run_tool") as mock_run_tool, \
             patch(
                 "advisor.graph.check_hard_escalation",
                 return_value=None,
             ):

            result = advisor_node(initial_state)

        assert (
            "Plan A, Plan B, or Plan C"
            in result["answer"]
        )

        assert result["answered"] is False
        assert result["needs_clarification"] is True
        assert result["confidence"] == "high"
        assert result["question_type"] == "degree_audit"

        # Most important: do not guess a plan.
        mock_run_tool.assert_not_called()

        assert result["tools_tried"] == []
        assert result["tool_contexts"] == []

class TestPrerequisiteRoutingIntegration:

    def test_prerequisite_question_cannot_rely_on_handbook_alone(self):
        # First response: model incorrectly starts with handbook search
        handbook_call = SimpleNamespace(
            id="call_handbook",
            type="function",
            function=SimpleNamespace(
                name="search_handbook",
                arguments='{"query":"CSCI 5521 prerequisites"}',
            ),
        )

        first_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[handbook_call],
                    )
                )
            ]
        )

        # Second response: model tries to answer without using
        # the dedicated prerequisite tool.
        premature_answer = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "According to the handbook, CSCI 5521 "
                            "has prerequisite coursework."
                        ),
                        tool_calls=None,
                    )
                )
            ]
        )

        # This is what the graph SHOULD force next.
        prerequisite_call = SimpleNamespace(
            id="call_prerequisite",
            type="function",
            function=SimpleNamespace(
                name="check_prerequisites",
                arguments='{"course_code":"CSCI5521"}',
            ),
        )

        prerequisite_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[prerequisite_call],
                    )
                )
            ]
        )

        final_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "The prerequisite tool confirms the "
                            "requirements for CSCI 5521."
                        ),
                        tool_calls=None,
                    )
                )
            ]
        )

        meta_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        parsed=AdvisorMeta(
                            answered=True,
                            needs_clarification=False,
                            confidence="high",
                            question_type="course_prerequisite",
                        )
                    )
                )
            ]
        )

        fake_client = MagicMock()

        fake_client.chat.completions.create.side_effect = [
            first_response,
            premature_answer,
            prerequisite_response,
            final_response,
        ]

        fake_client.beta.chat.completions.parse.return_value = (
            meta_response
        )

        def fake_run_tool(tool_name, tool_args):
            if tool_name == "search_handbook":
                return (
                    "HANDBOOK RESULT: General prerequisite policy."
                )

            if tool_name == "check_prerequisites":
                return (
                    "PREREQUISITE RESULT: CSCI5521 prerequisites checked."
                )

            raise AssertionError(
                f"Unexpected tool called: {tool_name}"
            )

        initial_state = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "What are the prerequisites for CSCI 5521?"
                    ),
                }
            ],
            "answer": "",
            "answered": False,
            "needs_clarification": False,
            "confidence": "none",
            "question_type": "unknown",
            "tools_tried": [],
            "drafted_email": "",
            "tool_contexts": [],
            "escalation_office": "",
        }

        with patch("advisor.graph.client", fake_client), \
             patch(
                 "advisor.graph.run_tool",
                 side_effect=fake_run_tool,
             ) as mock_run_tool, \
             patch(
                 "advisor.graph.check_hard_escalation",
                 return_value=None,
             ):

            result = advisor_node(initial_state)

        called_tools = [
            call.args[0]
            for call in mock_run_tool.call_args_list
        ]

        assert called_tools == [
            "search_handbook",
            "check_prerequisites",
        ]

        prerequisite_args = (
            mock_run_tool.call_args_list[1].args[1]
        )

        assert prerequisite_args["course_code"] == "CSCI5521"

        assert "PREREQUISITE RESULT" in result["tool_contexts"][1]

        assert result["question_type"] == "course_prerequisite"


    def test_prerequisite_question_uses_prerequisite_tool_directly(self):
        prerequisite_call = SimpleNamespace(
            id="call_prerequisite",
            type="function",
            function=SimpleNamespace(
                name="check_prerequisites",
                arguments='{"course_code":"CSCI5521"}',
            ),
        )

        first_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[prerequisite_call],
                    )
                )
            ]
        )

        final_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "CSCI 5521 prerequisites were checked "
                            "using the prerequisite tool."
                        ),
                        tool_calls=None,
                    )
                )
            ]
        )

        meta_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        parsed=AdvisorMeta(
                            answered=True,
                            needs_clarification=False,
                            confidence="high",
                            question_type="course_prerequisite",
                        )
                    )
                )
            ]
        )

        fake_client = MagicMock()

        fake_client.chat.completions.create.side_effect = [
            first_response,
            final_response,
        ]

        fake_client.beta.chat.completions.parse.return_value = (
            meta_response
        )

        def fake_run_tool(tool_name, tool_args):
            if tool_name == "check_prerequisites":
                return (
                    "PREREQUISITE RESULT: "
                    "CSCI5521 prerequisites checked."
                )

            raise AssertionError(
                f"Unexpected tool called: {tool_name}"
            )

        initial_state = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "What are the prerequisites for CSCI 5521?"
                    ),
                }
            ],
            "answer": "",
            "answered": False,
            "needs_clarification": False,
            "confidence": "none",
            "question_type": "unknown",
            "tools_tried": [],
            "drafted_email": "",
            "tool_contexts": [],
            "escalation_office": "",
        }

        with patch("advisor.graph.client", fake_client), \
            patch(
                "advisor.graph.run_tool",
                side_effect=fake_run_tool,
            ) as mock_run_tool, \
            patch(
                "advisor.graph.check_hard_escalation",
                return_value=None,
            ):

            result = advisor_node(initial_state)

        called_tools = [
            call.args[0]
            for call in mock_run_tool.call_args_list
        ]

        assert called_tools == [
            "check_prerequisites",
        ]

        prerequisite_args = (
            mock_run_tool.call_args_list[0].args[1]
        )

        assert prerequisite_args["course_code"] == "CSCI5521"

        assert len(result["tool_contexts"]) == 1
        assert "PREREQUISITE RESULT" in result["tool_contexts"][0]

        assert result["answered"] is True
        assert result["confidence"] == "high"
        assert result["question_type"] == "course_prerequisite"

class TestDeadlineRoutingIntegration:

    def test_deadline_question_cannot_rely_on_handbook_alone(self):
        handbook_call = SimpleNamespace(
            id="call_handbook",
            type="function",
            function=SimpleNamespace(
                name="search_handbook",
                arguments='{"query":"graduation application deadline"}',
            ),
        )

        first_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[handbook_call],
                    )
                )
            ]
        )

        premature_answer = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "The graduation deadline is listed in "
                            "the university calendar."
                        ),
                        tool_calls=None,
                    )
                )
            ]
        )

        deadline_call = SimpleNamespace(
            id="call_deadline",
            type="function",
            function=SimpleNamespace(
                name="get_deadline",
                arguments=(
                    '{"process_name":"graduation_application"}'
                ),
            ),
        )

        deadline_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[deadline_call],
                    )
                )
            ]
        )

        final_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "The deadline tool confirms the "
                            "graduation application timing."
                        ),
                        tool_calls=None,
                    )
                )
            ]
        )

        meta_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        parsed=AdvisorMeta(
                            answered=True,
                            needs_clarification=False,
                            confidence="high",
                            question_type="deadline",
                        )
                    )
                )
            ]
        )

        fake_client = MagicMock()

        fake_client.chat.completions.create.side_effect = [
            first_response,
            premature_answer,
            deadline_response,
            final_response,
        ]

        fake_client.beta.chat.completions.parse.return_value = (
            meta_response
        )

        def fake_run_tool(tool_name, tool_args):
            if tool_name == "search_handbook":
                return "HANDBOOK RESULT: General graduation information."

            if tool_name == "get_deadline":
                return (
                    "DEADLINE RESULT: "
                    "Graduation application timing found."
                )

            raise AssertionError(
                f"Unexpected tool called: {tool_name}"
            )

        initial_state = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "What is the graduation application deadline?"
                    ),
                }
            ],
            "answer": "",
            "answered": False,
            "needs_clarification": False,
            "confidence": "none",
            "question_type": "unknown",
            "tools_tried": [],
            "drafted_email": "",
            "tool_contexts": [],
            "escalation_office": "",
        }

        with patch("advisor.graph.client", fake_client), \
             patch(
                 "advisor.graph.run_tool",
                 side_effect=fake_run_tool,
             ) as mock_run_tool, \
             patch(
                 "advisor.graph.check_hard_escalation",
                 return_value=None,
             ):

            advisor_node(initial_state)

        called_tools = [
            call.args[0]
            for call in mock_run_tool.call_args_list
        ]

        assert called_tools == [
            "search_handbook",
            "get_deadline",
        ]


        def test_deadline_question_uses_deadline_tool_directly(self):
            deadline_call = SimpleNamespace(
                id="call_deadline",
                type="function",
                function=SimpleNamespace(
                    name="get_deadline",
                    arguments='{"process_name":"graduation_application"}',
                ),
            )

            first_response = SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="",
                            tool_calls=[deadline_call],
                        )
                    )
                ]
            )

            final_response = SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                "The graduation application timing was "
                                "checked using the deadline tool."
                            ),
                            tool_calls=None,
                        )
                    )
                ]
            )

            meta_response = SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            parsed=AdvisorMeta(
                                answered=True,
                                needs_clarification=False,
                                confidence="high",
                                question_type="deadline",
                            )
                        )
                    )
                ]
            )

            fake_client = MagicMock()

            fake_client.chat.completions.create.side_effect = [
                first_response,
                final_response,
            ]

            fake_client.beta.chat.completions.parse.return_value = (
                meta_response
            )

            def fake_run_tool(tool_name, tool_args):
                if tool_name == "get_deadline":
                    return (
                        "DEADLINE RESULT: "
                        "Graduation application timing found."
                    )

                raise AssertionError(
                    f"Unexpected tool called: {tool_name}"
                )

            initial_state = {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "What is the graduation application deadline?"
                        ),
                    }
                ],
                "answer": "",
                "answered": False,
                "needs_clarification": False,
                "confidence": "none",
                "question_type": "unknown",
                "tools_tried": [],
                "drafted_email": "",
                "tool_contexts": [],
                "escalation_office": "",
            }

            with patch("advisor.graph.client", fake_client), \
                patch(
                    "advisor.graph.run_tool",
                    side_effect=fake_run_tool,
                ) as mock_run_tool, \
                patch(
                    "advisor.graph.check_hard_escalation",
                    return_value=None,
                ):

                result = advisor_node(initial_state)

            called_tools = [
                call.args[0]
                for call in mock_run_tool.call_args_list
            ]

            assert called_tools == [
                "get_deadline",
            ]

            deadline_args = (
                mock_run_tool.call_args_list[0].args[1]
            )

            assert (
                deadline_args["process_name"]
                == "graduation_application"
            )

            assert len(result["tool_contexts"]) == 1
            assert "DEADLINE RESULT" in result["tool_contexts"][0]

            assert result["answered"] is True
            assert result["confidence"] == "high"
            assert result["question_type"] == "deadline"

class TestCourseDifficultyRoutingIntegration:

    def test_course_difficulty_cannot_rely_on_handbook_alone(self):
        handbook_call = SimpleNamespace(
            id="call_handbook",
            type="function",
            function=SimpleNamespace(
                name="search_handbook",
                arguments='{"query":"CSCI 5521 difficulty workload"}',
            ),
        )

        first_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[handbook_call],
                    )
                )
            ]
        )

        premature_answer = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="CSCI 5521 appears to be a challenging course.",
                        tool_calls=None,
                    )
                )
            ]
        )

        grade_call = SimpleNamespace(
            id="call_grade_distribution",
            type="function",
            function=SimpleNamespace(
                name="get_grade_distribution",
                arguments='{"course_code":"CSCI5521"}',
            ),
        )

        grade_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[grade_call],
                    )
                )
            ]
        )

        final_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "Historical grade-distribution data provides "
                            "a better basis for discussing CSCI 5521 difficulty."
                        ),
                        tool_calls=None,
                    )
                )
            ]
        )

        meta_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        parsed=AdvisorMeta(
                            answered=True,
                            needs_clarification=False,
                            confidence="high",
                            question_type="course_difficulty",
                        )
                    )
                )
            ]
        )

        fake_client = MagicMock()

        fake_client.chat.completions.create.side_effect = [
            first_response,
            premature_answer,
            grade_response,
            final_response,
        ]

        fake_client.beta.chat.completions.parse.return_value = (
            meta_response
        )

        def fake_run_tool(tool_name, tool_args):
            if tool_name == "search_handbook":
                return "HANDBOOK RESULT: General course information."

            if tool_name == "get_grade_distribution":
                return (
                    "GRADE RESULT: Historical grade distribution "
                    "for CSCI5521."
                )

            raise AssertionError(
                f"Unexpected tool called: {tool_name}"
            )

        initial_state = {
            "messages": [
                {
                    "role": "user",
                    "content": "How hard is CSCI 5521?",
                }
            ],
            "answer": "",
            "answered": False,
            "needs_clarification": False,
            "confidence": "none",
            "question_type": "unknown",
            "tools_tried": [],
            "drafted_email": "",
            "tool_contexts": [],
            "escalation_office": "",
        }

        with patch("advisor.graph.client", fake_client), \
             patch(
                 "advisor.graph.run_tool",
                 side_effect=fake_run_tool,
             ) as mock_run_tool, \
             patch(
                 "advisor.graph.check_hard_escalation",
                 return_value=None,
             ):

            advisor_node(initial_state)

        called_tools = [
            call.args[0]
            for call in mock_run_tool.call_args_list
        ]

        assert called_tools == [
            "search_handbook",
            "get_grade_distribution",
        ]

    def test_course_difficulty_uses_grade_distribution_directly(self):
        grade_call = SimpleNamespace(
            id="call_grade_distribution",
            type="function",
            function=SimpleNamespace(
                name="get_grade_distribution",
                arguments='{"course_code":"CSCI5521"}',
            ),
        )

        first_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[grade_call],
                    )
                )
            ]
        )

        final_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "Historical grade-distribution data was used "
                            "to discuss CSCI 5521 difficulty."
                        ),
                        tool_calls=None,
                    )
                )
            ]
        )

        meta_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        parsed=AdvisorMeta(
                            answered=True,
                            needs_clarification=False,
                            confidence="high",
                            question_type="course_difficulty",
                        )
                    )
                )
            ]
        )

        fake_client = MagicMock()

        fake_client.chat.completions.create.side_effect = [
            first_response,
            final_response,
        ]

        fake_client.beta.chat.completions.parse.return_value = (
            meta_response
        )

        def fake_run_tool(tool_name, tool_args):
            if tool_name == "get_grade_distribution":
                return (
                    "GRADE RESULT: Historical grade distribution "
                    "for CSCI5521."
                )

            raise AssertionError(
                f"Unexpected tool called: {tool_name}"
            )

        initial_state = {
            "messages": [
                {
                    "role": "user",
                    "content": "How hard is CSCI 5521?",
                }
            ],
            "answer": "",
            "answered": False,
            "needs_clarification": False,
            "confidence": "none",
            "question_type": "unknown",
            "tools_tried": [],
            "drafted_email": "",
            "tool_contexts": [],
            "escalation_office": "",
        }

        with patch("advisor.graph.client", fake_client), \
            patch(
               "advisor.graph.run_tool",
                side_effect=fake_run_tool,
            ) as mock_run_tool, \
            patch(
                "advisor.graph.check_hard_escalation",
                return_value=None,
            ):

            result = advisor_node(initial_state)

        called_tools = [
            call.args[0]
            for call in mock_run_tool.call_args_list
        ]

        assert called_tools == [
            "get_grade_distribution",
        ]

        grade_args = (
            mock_run_tool.call_args_list[0].args[1]
        )

        assert grade_args["course_code"] == "CSCI5521"

        assert len(result["tool_contexts"]) == 1
        assert "GRADE RESULT" in result["tool_contexts"][0]

        assert result["answered"] is True
        assert result["confidence"] == "high"
        assert result["question_type"] == "course_difficulty"

class TestBreadthEligibilityRoutingIntegration:

    def test_breadth_eligibility_cannot_rely_on_handbook_alone(self):
        handbook_call = SimpleNamespace(
            id="call_handbook",
            type="function",
            function=SimpleNamespace(
                name="search_handbook",
                arguments='{"query":"CSCI 5527 breadth eligibility MS"}',
            ),
        )

        first_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[handbook_call],
                    )
                )
            ]
        )

        premature_answer = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "CSCI 5527 appears to count toward breadth."
                        ),
                        tool_calls=None,
                    )
                )
            ]
        )

        breadth_call = SimpleNamespace(
            id="call_breadth",
            type="function",
            function=SimpleNamespace(
                name="check_breadth_eligibility",
                arguments=(
                    '{"course_code":"CSCI5527","program":"ms"}'
                ),
            ),
        )

        breadth_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[breadth_call],
                    )
                )
            ]
        )

        final_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "CSCI 5527 was checked using the "
                            "breadth eligibility tool."
                        ),
                        tool_calls=None,
                    )
                )
            ]
        )

        meta_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        parsed=AdvisorMeta(
                            answered=True,
                            needs_clarification=False,
                            confidence="high",
                            question_type="policy",
                        )
                    )
                )
            ]
        )

        fake_client = MagicMock()

        fake_client.chat.completions.create.side_effect = [
            first_response,
            premature_answer,
            breadth_response,
            final_response,
        ]

        fake_client.beta.chat.completions.parse.return_value = (
            meta_response
        )

        def fake_run_tool(tool_name, tool_args):
            if tool_name == "search_handbook":
                return (
                    "HANDBOOK RESULT: General breadth information."
                )

            if tool_name == "check_breadth_eligibility":
                return (
                    "BREADTH RESULT: CSCI5527 is listed in an "
                    "approved MS breadth category."
                )

            raise AssertionError(
                f"Unexpected tool called: {tool_name}"
            )

        initial_state = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Does CSCI 5527 count for breadth "
                        "in the MS program?"
                    ),
                }
            ],
            "answer": "",
            "answered": False,
            "needs_clarification": False,
            "confidence": "none",
            "question_type": "unknown",
            "tools_tried": [],
            "drafted_email": "",
            "tool_contexts": [],
            "escalation_office": "",
        }

        with patch("advisor.graph.client", fake_client), \
             patch(
                 "advisor.graph.run_tool",
                 side_effect=fake_run_tool,
             ) as mock_run_tool, \
             patch(
                 "advisor.graph.check_hard_escalation",
                 return_value=None,
             ):

            advisor_node(initial_state)

        called_tools = [
            call.args[0]
            for call in mock_run_tool.call_args_list
        ]

        assert called_tools == [
            "search_handbook",
            "check_breadth_eligibility",
        ]

    def test_breadth_eligibility_uses_breadth_tool_directly(self):
        breadth_call = SimpleNamespace(
            id="call_breadth",
            type="function",
            function=SimpleNamespace(
                name="check_breadth_eligibility",
                arguments=(
                    '{"course_code":"CSCI5527","program":"ms"}'
                ),
            ),
        )

        first_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[breadth_call],
                    )
                )
            ]
        )

        final_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "CSCI 5527 was checked using the "
                            "breadth eligibility tool."
                        ),
                        tool_calls=None,
                    )
                )
            ]
        )

        meta_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        parsed=AdvisorMeta(
                            answered=True,
                            needs_clarification=False,
                            confidence="high",
                            question_type="policy",
                        )
                    )
                )
            ]
        )

        fake_client = MagicMock()

        fake_client.chat.completions.create.side_effect = [
            first_response,
            final_response,
        ]

        fake_client.beta.chat.completions.parse.return_value = (
            meta_response
        )

        def fake_run_tool(tool_name, tool_args):
            if tool_name == "check_breadth_eligibility":
                return (
                    "BREADTH RESULT: CSCI5527 is listed in an "
                    "approved MS breadth category."
                )

            raise AssertionError(
                f"Unexpected tool called: {tool_name}"
            )

        initial_state = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Does CSCI 5527 count for breadth "
                        "in the MS program?"
                    ),
                }
            ],
            "answer": "",
            "answered": False,
            "needs_clarification": False,
            "confidence": "none",
            "question_type": "unknown",
            "tools_tried": [],
            "drafted_email": "",
            "tool_contexts": [],
            "escalation_office": "",
        }

        with patch("advisor.graph.client", fake_client), \
            patch(
                "advisor.graph.run_tool",
                side_effect=fake_run_tool,
            ) as mock_run_tool, \
            patch(
                "advisor.graph.check_hard_escalation",
                return_value=None,
            ):

            result = advisor_node(initial_state)

        called_tools = [
            call.args[0]
            for call in mock_run_tool.call_args_list
        ]

        assert called_tools == [
            "check_breadth_eligibility",
        ]

        breadth_args = (
            mock_run_tool.call_args_list[0].args[1]
        )

        assert breadth_args["course_code"] == "CSCI5527"
        assert breadth_args["program"] == "ms"

        assert len(result["tool_contexts"]) == 1
        assert "BREADTH RESULT" in result["tool_contexts"][0]

        assert result["answered"] is True
        assert result["confidence"] == "high"
        assert result["question_type"] == "policy"


class TestCoursesRequiringRoutingIntegration:

    def test_courses_requiring_cannot_rely_on_handbook_alone(self):
        handbook_call = SimpleNamespace(
            id="call_handbook",
            type="function",
            function=SimpleNamespace(
                name="search_handbook",
                arguments='{"query":"courses requiring CSCI 5521"}',
            ),
        )

        first_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[handbook_call],
                    )
                )
            ]
        )

        premature_answer = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "Several courses may require CSCI 5521."
                        ),
                        tool_calls=None,
                    )
                )
            ]
        )

        courses_requiring_call = SimpleNamespace(
            id="call_courses_requiring",
            type="function",
            function=SimpleNamespace(
                name="get_courses_requiring",
                arguments='{"course_code":"CSCI5521"}',
            ),
        )

        courses_requiring_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[courses_requiring_call],
                    )
                )
            ]
        )

        final_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "The reverse prerequisite lookup was used "
                            "to find courses requiring CSCI 5521."
                        ),
                        tool_calls=None,
                    )
                )
            ]
        )

        meta_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        parsed=AdvisorMeta(
                            answered=True,
                            needs_clarification=False,
                            confidence="high",
                            question_type="course_prerequisite",
                        )
                    )
                )
            ]
        )

        fake_client = MagicMock()

        fake_client.chat.completions.create.side_effect = [
            first_response,
            premature_answer,
            courses_requiring_response,
            final_response,
        ]

        fake_client.beta.chat.completions.parse.return_value = (
            meta_response
        )

        def fake_run_tool(tool_name, tool_args):
            if tool_name == "search_handbook":
                return (
                    "HANDBOOK RESULT: General course information."
                )

            if tool_name == "get_courses_requiring":
                return (
                    "COURSES REQUIRING RESULT: Courses that "
                    "require CSCI5521 were found."
                )

            raise AssertionError(
                f"Unexpected tool called: {tool_name}"
            )

        initial_state = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Which courses require CSCI 5521?"
                    ),
                }
            ],
            "answer": "",
            "answered": False,
            "needs_clarification": False,
            "confidence": "none",
            "question_type": "unknown",
            "tools_tried": [],
            "drafted_email": "",
            "tool_contexts": [],
            "escalation_office": "",
        }

        with patch("advisor.graph.client", fake_client), \
             patch(
                 "advisor.graph.run_tool",
                 side_effect=fake_run_tool,
             ) as mock_run_tool, \
             patch(
                 "advisor.graph.check_hard_escalation",
                 return_value=None,
             ):

            advisor_node(initial_state)

        called_tools = [
            call.args[0]
            for call in mock_run_tool.call_args_list
        ]

        assert called_tools == [
            "search_handbook",
            "get_courses_requiring",
        ]


    def test_reverse_prerequisite_wording_does_not_force_direct_prerequisite_tool(self):
        handbook_call = SimpleNamespace(
            id="call_handbook",
            type="function",
            function=SimpleNamespace(
                name="search_handbook",
                arguments='{"query":"courses requiring CSCI 5521 as prerequisite"}',
            ),
        )

        first_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[handbook_call],
                    )
                )
            ]
        )

        premature_answer = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Several courses may require CSCI 5521.",
                        tool_calls=None,
                    )
                )
            ]
        )

        courses_requiring_call = SimpleNamespace(
            id="call_courses_requiring",
            type="function",
            function=SimpleNamespace(
                name="get_courses_requiring",
                arguments='{"course_code":"CSCI5521"}',
            ),
        )

        courses_requiring_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[courses_requiring_call],
                    )
                )
            ]
        )

        final_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "These are the courses that require "
                            "CSCI 5521 as a prerequisite."
                        ),
                        tool_calls=None,
                    )
                )
            ]
        )

        # If the graph incorrectly forces check_prerequisites,
        # this fifth response exposes that bad path.
        wrong_prerequisite_call = SimpleNamespace(
            id="call_wrong_prerequisite",
            type="function",
            function=SimpleNamespace(
                name="check_prerequisites",
                arguments='{"course_code":"CSCI5521"}',
            ),
        )

        wrong_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[wrong_prerequisite_call],
                    )
                )
            ]
        )

        meta_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        parsed=AdvisorMeta(
                            answered=True,
                            needs_clarification=False,
                            confidence="high",
                            question_type="course_prerequisite",
                        )
                    )
                )
            ]
        )

        fake_client = MagicMock()

        fake_client.chat.completions.create.side_effect = [
            first_response,
            premature_answer,
            courses_requiring_response,
            final_response,
            wrong_response,
        ]

        fake_client.beta.chat.completions.parse.return_value = (
            meta_response
        )

        def fake_run_tool(tool_name, tool_args):
            if tool_name == "search_handbook":
                return "HANDBOOK RESULT: General course information."

            if tool_name == "get_courses_requiring":
                return (
                    "COURSES REQUIRING RESULT: Courses requiring "
                    "CSCI5521 were found."
                )

            if tool_name == "check_prerequisites":
                return (
                    "WRONG TOOL RESULT: Direct prerequisites "
                    "for CSCI5521."
                )

            raise AssertionError(
                f"Unexpected tool called: {tool_name}"
            )

        initial_state = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Which courses require CSCI 5521 "
                        "as a prerequisite?"
                    ),
                }
            ],
            "answer": "",
            "answered": False,
            "needs_clarification": False,
            "confidence": "none",
            "question_type": "unknown",
            "tools_tried": [],
            "drafted_email": "",
            "tool_contexts": [],
            "escalation_office": "",
        }

        with patch("advisor.graph.client", fake_client), \
            patch(
                "advisor.graph.run_tool",
                side_effect=fake_run_tool,
            ) as mock_run_tool, \
            patch(
                "advisor.graph.check_hard_escalation",
                return_value=None,
            ):

            advisor_node(initial_state)

        called_tools = [
            call.args[0]
            for call in mock_run_tool.call_args_list
        ]

        assert called_tools == [
            "search_handbook",
            "get_courses_requiring",
        ]

    def test_courses_requiring_uses_reverse_lookup_directly(self):
        courses_requiring_call = SimpleNamespace(
            id="call_courses_requiring",
            type="function",
            function=SimpleNamespace(
                name="get_courses_requiring",
                arguments='{"course_code":"CSCI5521"}',
            ),
        )

        first_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[courses_requiring_call],
                    )
                )
            ]
        )

        final_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "The reverse prerequisite lookup was used "
                            "to find courses requiring CSCI 5521."
                        ),
                        tool_calls=None,
                    )
                )
            ]
        )

        meta_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        parsed=AdvisorMeta(
                            answered=True,
                            needs_clarification=False,
                            confidence="high",
                            question_type="course_prerequisite",
                        )
                    )
                )
            ]
        )

        fake_client = MagicMock()

        fake_client.chat.completions.create.side_effect = [
            first_response,
            final_response,
        ]

        fake_client.beta.chat.completions.parse.return_value = (
            meta_response
        )

        def fake_run_tool(tool_name, tool_args):
            if tool_name == "get_courses_requiring":
                return (
                    "COURSES REQUIRING RESULT: Courses requiring "
                    "CSCI5521 were found."
                )

            raise AssertionError(
                f"Unexpected tool called: {tool_name}"
            )

        initial_state = {
            "messages": [
                {
                    "role": "user",
                    "content": "Which courses require CSCI 5521?",
                }
            ],
            "answer": "",
            "answered": False,
            "needs_clarification": False,
            "confidence": "none",
            "question_type": "unknown",
            "tools_tried": [],
            "drafted_email": "",
            "tool_contexts": [],
            "escalation_office": "",
        }

        with patch("advisor.graph.client", fake_client), \
            patch(
                "advisor.graph.run_tool",
                side_effect=fake_run_tool,
            ) as mock_run_tool, \
            patch(
                "advisor.graph.check_hard_escalation",
                return_value=None,
            ):

            result = advisor_node(initial_state)

        called_tools = [
            call.args[0]
            for call in mock_run_tool.call_args_list
        ]

        assert called_tools == [
            "get_courses_requiring",
        ]

        lookup_args = (
            mock_run_tool.call_args_list[0].args[1]
        )

        assert lookup_args["course_code"] == "CSCI5521"

        assert len(result["tool_contexts"]) == 1
        assert (
            "COURSES REQUIRING RESULT"
            in result["tool_contexts"][0]
        )

        assert result["answered"] is True
        assert result["confidence"] == "high"
        assert result["question_type"] == "course_prerequisite"