def test_tool_score_passes_after_successful_retry():
    from eval.tool_scoring import score_tool_execution

    question = {
        "id": "E023",
        "required_tool": "degree_audit",
        "expected_program": "ms",
        "expected_plan": "C",
    }

    result = {
        "answer": "You are missing one breadth area.",
        "tool_trace": [
            {
                "name": "degree_audit",
                "arguments": {
                    "program": "ms",
                    "plan": "C",
                },
                "success": False,
                "error": "Temporary failure",
            },
            {
                "name": "degree_audit",
                "arguments": {
                    "program": "ms",
                    "plan": "C",
                    "completed_courses": ["CSCI5511", "CSCI5521"],
                },
                "success": True,
            },
        ],
    }

    score = score_tool_execution(question, result)

    assert score["tool_called"] is True
    assert score["tool_succeeded"] is True
    assert score["program_match"] is True
    assert score["plan_match"] is True
    assert score["answer_nonempty"] is True
    assert score["passed"] is True


def test_tool_score_requires_program_and_plan_on_same_call():
    from eval.tool_scoring import score_tool_execution

    question = {
        "id": "E023",
        "required_tool": "degree_audit",
        "expected_program": "ms",
        "expected_plan": "C",
    }

    result = {
        "answer": "Audit completed.",
        "tool_trace": [
            {
                "name": "degree_audit",
                "arguments": {
                    "program": "ms",
                    "plan": "B",
                },
                "success": True,
            },
            {
                "name": "degree_audit",
                "arguments": {
                    "program": "phd",
                    "plan": "C",
                },
                "success": True,
            },
        ],
    }

    score = score_tool_execution(question, result)

    assert score["tool_called"] is True
    assert score["tool_succeeded"] is True
    assert score["program_match"] is True

    # Plan C appeared, but only on the wrong program call.
    assert score["plan_match"] is False
    assert score["passed"] is False


def test_tool_score_fails_when_required_tool_not_called():
    from eval.tool_scoring import score_tool_execution

    question = {
        "required_tool": "degree_audit",
        "expected_program": "ms",
        "expected_plan": "C",
    }

    result = {
        "answer": "Some answer.",
        "tool_trace": [
            {
                "name": "search_handbook",
                "arguments": {"query": "Plan C requirements"},
                "success": True,
            }
        ],
    }

    score = score_tool_execution(question, result)

    assert score["tool_called"] is False
    assert score["tool_succeeded"] is False
    assert score["passed"] is False


def test_tool_score_fails_when_required_tool_only_failed():
    from eval.tool_scoring import score_tool_execution

    question = {
        "required_tool": "degree_audit",
        "expected_program": "ms",
        "expected_plan": "C",
    }

    result = {
        "answer": "Audit could not be completed.",
        "tool_trace": [
            {
                "name": "degree_audit",
                "arguments": {
                    "program": "ms",
                    "plan": "C",
                },
                "success": False,
                "error": "Temporary failure",
            }
        ],
    }

    score = score_tool_execution(question, result)

    assert score["tool_called"] is True
    assert score["tool_succeeded"] is False
    assert score["passed"] is False


def test_tool_score_fails_for_wrong_program():
    from eval.tool_scoring import score_tool_execution

    question = {
        "required_tool": "degree_audit",
        "expected_program": "ms",
        "expected_plan": "C",
    }

    result = {
        "answer": "Audit completed.",
        "tool_trace": [
            {
                "name": "degree_audit",
                "arguments": {
                    "program": "phd",
                    "plan": "C",
                },
                "success": True,
            }
        ],
    }

    score = score_tool_execution(question, result)

    assert score["tool_succeeded"] is True
    assert score["program_match"] is False
    assert score["plan_match"] is False
    assert score["passed"] is False


def test_tool_score_fails_for_wrong_plan():
    from eval.tool_scoring import score_tool_execution

    question = {
        "required_tool": "degree_audit",
        "expected_program": "ms",
        "expected_plan": "C",
    }

    result = {
        "answer": "Audit completed.",
        "tool_trace": [
            {
                "name": "degree_audit",
                "arguments": {
                    "program": "ms",
                    "plan": "B",
                },
                "success": True,
            }
        ],
    }

    score = score_tool_execution(question, result)

    assert score["program_match"] is True
    assert score["plan_match"] is False
    assert score["passed"] is False


def test_tool_score_fails_for_empty_final_answer():
    from eval.tool_scoring import score_tool_execution

    question = {
        "required_tool": "degree_audit",
        "expected_program": "ms",
        "expected_plan": "C",
    }

    result = {
        "answer": "   ",
        "tool_trace": [
            {
                "name": "degree_audit",
                "arguments": {
                    "program": "ms",
                    "plan": "C",
                },
                "success": True,
            }
        ],
    }

    score = score_tool_execution(question, result)

    assert score["tool_succeeded"] is True
    assert score["program_match"] is True
    assert score["plan_match"] is True
    assert score["answer_nonempty"] is False
    assert score["passed"] is False


def test_tool_score_does_not_require_plan_for_phd():
    from eval.tool_scoring import score_tool_execution

    question = {
        "id": "E028",
        "required_tool": "degree_audit",
        "expected_program": "phd",
    }

    result = {
        "answer": "CSCI 8001 is still required.",
        "tool_trace": [
            {
                "name": "degree_audit",
                "arguments": {
                    "program": "phd",
                },
                "success": True,
            }
        ],
    }

    score = score_tool_execution(question, result)

    assert score["tool_called"] is True
    assert score["tool_succeeded"] is True
    assert score["program_match"] is True
    assert score["plan_match"] is True
    assert score["answer_nonempty"] is True
    assert score["passed"] is True


def test_tool_score_rejects_nonempty_but_semantically_wrong_answer():
    from eval.tool_scoring import score_tool_execution

    question = {
        "required_tool": "degree_audit",
        "expected_program": "ms",
        "expected_plan": "C",
        "required_answer_terms": [
            "CSCI 4041",
            "non-CSCI",
            "approval",
        ],
    }

    result = {
        "answer": (
            "You need 18 more total degree credits. "
            "CSCI 4041 does not count."
        ),
        "tool_trace": [
            {
                "name": "degree_audit",
                "arguments": {
                    "program": "ms",
                    "plan": "C",
                },
                "success": True,
            }
        ],
    }

    score = score_tool_execution(question, result)

    assert score["answer_nonempty"] is True
    assert score["answer_behavior_match"] is False
    assert score["passed"] is False


def test_tool_score_rejects_forbidden_answer_claim():
    from eval.tool_scoring import score_tool_execution

    question = {
        "required_tool": "degree_audit",
        "expected_program": "ms",
        "expected_plan": "C",
        "required_answer_terms": [
            "non-CSCI",
            "verification",
        ],
        "forbidden_answer_terms": [
            "18 more degree credits",
        ],
    }

    result = {
        "answer": (
            "Your non-CSCI coursework needs verification, "
            "but you still need 18 more degree credits."
        ),
        "tool_trace": [
            {
                "name": "degree_audit",
                "arguments": {
                    "program": "ms",
                    "plan": "C",
                },
                "success": True,
            }
        ],
    }

    score = score_tool_execution(question, result)

    assert score["answer_behavior_match"] is True
    assert score["forbidden_answer_match"] is False
    assert score["passed"] is False


def test_tool_score_accepts_any_required_semantic_alternative():
    from eval.tool_scoring import score_tool_execution

    question = {
        "required_tool": "degree_audit",
        "expected_program": "ms",
        "expected_plan": "C",
        "required_answer_terms": [
            "CSCI 4041",
            "non-CSCI",
        ],
        "required_answer_any_groups": [
            [
                "approval",
                "approved",
                "verification",
                "verified",
            ]
        ],
    }

    result = {
        "answer": (
            "CSCI 4041 does not count. "
            "Your non-CSCI credits must be verified "
            "before they can count toward the degree."
        ),
        "tool_trace": [
            {
                "name": "degree_audit",
                "arguments": {
                    "program": "ms",
                    "plan": "C",
                },
                "success": True,
            }
        ],
    }

    score = score_tool_execution(question, result)

    assert score["answer_behavior_match"] is True
    assert score["answer_any_group_match"] is True
    assert score["passed"] is True