from advisor.tools import tools


def test_degree_audit_schema_exposes_non_csci_credit_summary():
    degree_audit_tool = next(
        tool
        for tool in tools
        if tool["function"]["name"] == "degree_audit"
    )

    properties = (
        degree_audit_tool["function"]["parameters"]["properties"]
    )

    assert "non_csci_credit_summary" in properties

    summary_schema = properties["non_csci_credit_summary"]

    assert summary_schema["type"] == "object"

    assert (
        summary_schema["properties"]["approved"]["type"]
        == "number"
    )

    assert (
        summary_schema["properties"]["pending_approval"]["type"]
        == "number"
    )


def test_degree_audit_schema_accepts_rich_course_records():
    degree_audit_tool = next(
        tool
        for tool in tools
        if tool["function"]["name"] == "degree_audit"
    )

    completed_courses_schema = (
        degree_audit_tool["function"]["parameters"]["properties"]
        ["completed_courses"]
    )

    items_schema = completed_courses_schema["items"]

    # Must support both legacy strings and rich course records.
    assert "anyOf" in items_schema

    rich_record_schema = next(
        option
        for option in items_schema["anyOf"]
        if option.get("type") == "object"
    )

    properties = rich_record_schema["properties"]

    assert properties["code"]["type"] == "string"
    assert properties["credits"]["type"] == "number"
    assert properties["degree_approved"]["type"] == "boolean"
    assert properties["phd_credit_type"]["type"] == "string"