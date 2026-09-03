import json
from pathlib import Path

from advisor.graph import chat_for_evaluation
from eval.tool_scoring import score_tool_execution


HOLDOUT_PATH = Path("eval/tool_holdout_v2.json")


def main():
    with HOLDOUT_PATH.open() as f:
        questions = json.load(f)

    tool_questions = [
        q for q in questions
        if q.get("required_tool")
    ]

    passed = 0

    for question in tool_questions:
        result = chat_for_evaluation(
            question["user_question"],
            [],
        )

        score = score_tool_execution(
            question,
            result,
        )

        status = (
            "PASS"
            if score["passed"]
            else "FAIL"
        )

        print("=" * 80)
        print(
            f"{question['id']} {status}"
        )
        print(score)

        print("\nTool trace:")
        for trace in result.get(
            "tool_trace",
            [],
        ):
            print(trace)

        print("\nAnswer:")
        print(
            result.get(
                "answer",
                "",
            )
        )

        if score["passed"]:
            passed += 1

    total = len(tool_questions)

    print("\n" + "=" * 80)
    print(
        f"FROZEN HOLDOUT V2 TOOL TRACK: "
        f"{passed}/{total} passed "
        f"({passed / total:.0%})"
    )


if __name__ == "__main__":
    main()