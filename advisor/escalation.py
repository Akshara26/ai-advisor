import json
import logging
import os
import re

logger = logging.getLogger(__name__)


def load_json(filename: str) -> dict:
    path = os.path.join(os.path.dirname(__file__), "..", "data", filename)
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filename} not found — related features disabled")
        return {}


ESCALATION_CONFIG = load_json("escalation_config.json")
OFFICE_DIRECTORY = {
    o["id"]: o
    for o in load_json("office_directory.json").get("offices", [])
}
HARD_RULES = [
    r for r in ESCALATION_CONFIG.get("intent_escalation_rules", [])
    if r.get("escalation_level") == "hard"
]


def check_hard_escalation(question: str) -> dict | None:
    """
    Check if the question matches a hard escalation intent pattern.
    Runs BEFORE the LLM — matched questions never reach the advisor loop.
    Returns the matching rule dict or None.
    """
    q_lower = question.lower()
    for rule in HARD_RULES:
        if any(
            re.search(
                rf"\b{re.escape(pattern.lower())}\b",
                q_lower,
            )
            for pattern in rule.get("intent_pattern", [])
        ):
            logger.info(f"Hard escalation matched: {rule.get('intent_pattern', [])[:2]}")
            return rule
    return None
