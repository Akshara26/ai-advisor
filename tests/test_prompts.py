from advisor.prompts import ADVISOR_SYSTEM_PROMPT


def test_advisor_prompt_preserves_degree_audit_credit_metadata():
    prompt = ADVISOR_SYSTEM_PROMPT

    # M.S. degree audits must carry the student's plan.
    assert 'plan="C"' in prompt

    # The prompt must teach the model that richer course records are allowed.
    assert "degree_approved" in prompt
    assert "credits" in prompt

    # Aggregate non-CSCI credits must be preserved when course codes are unavailable.
    assert "non_csci_credit_summary" in prompt
    assert "pending_approval" in prompt

    # The prompt must explicitly prevent unsafe inference/double counting.
    assert "Do not infer" in prompt
    assert "Do not double-count" in prompt
    