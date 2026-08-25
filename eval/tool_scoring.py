def score_tool_execution(question: dict, result: dict) -> dict:
    required_tool = question.get("required_tool")
    expected_program = question.get("expected_program")
    expected_plan = question.get("expected_plan")

    required_answer_terms = question.get(
        "required_answer_terms",
        [],
    )

    required_answer_any_groups = question.get(
        "required_answer_any_groups",
        [],
    )

    forbidden_answer_terms = question.get(
        "forbidden_answer_terms",
        [],
    )

    tool_trace = result.get("tool_trace", [])

    matching_calls = [
        trace
        for trace in tool_trace
        if trace.get("name") == required_tool
    ]

    successful_calls = [
        trace
        for trace in matching_calls
        if trace.get("success") is True
    ]

    tool_called = bool(matching_calls)
    tool_succeeded = bool(successful_calls)

    program_matching_calls = (
        successful_calls
        if expected_program is None
        else [
            trace
            for trace in successful_calls
            if trace.get("arguments", {}).get("program")
            == expected_program
        ]
    )

    program_match = (
        True
        if expected_program is None
        else bool(program_matching_calls)
    )

    plan_match = (
        True
        if expected_plan is None
        else any(
            trace.get("arguments", {}).get("plan")
            == expected_plan
            for trace in program_matching_calls
        )
    )

    answer_nonempty = bool(
        result.get("answer", "").strip()
    )

    answer_text = result.get("answer", "").lower()

    answer_behavior_match = all(
        term.lower() in answer_text
        for term in required_answer_terms
    )

    answer_any_group_match = all(
        any(
            term.lower() in answer_text
            for term in group
        )
        for group in required_answer_any_groups
    )

    forbidden_answer_match = not any(
        term.lower() in answer_text
        for term in forbidden_answer_terms
    )

    passed = (
        tool_called
        and tool_succeeded
        and program_match
        and plan_match
        and answer_nonempty
        and answer_behavior_match
        and forbidden_answer_match
        and answer_any_group_match
    )

    return {
        "tool_called": tool_called,
        "tool_succeeded": tool_succeeded,
        "program_match": program_match,
        "plan_match": plan_match,
        "answer_nonempty": answer_nonempty,
        "answer_behavior_match": answer_behavior_match,
        "forbidden_answer_match": forbidden_answer_match,
        "answer_any_group_match": answer_any_group_match,
        "passed": passed,
    }