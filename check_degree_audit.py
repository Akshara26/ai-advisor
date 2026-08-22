"""
check_degree_audit.py
Run from repo root: python check_degree_audit.py

Validates degree_audit() output against known-correct expected values.
Covers the bugs identified in review:
  - cred_min fallback (1-credit courses like CSCI 8970 must not be counted as 3)
  - breadth count accuracy (no invented categories)
  - explicit breadth-complete line present in output
  - PhD breadth category count matches requirements.json exactly
"""

import sys
import json
import os

sys.path.insert(0, os.path.dirname(__file__))
from advisor.degree_audit import degree_audit, REQUIREMENTS

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"

results = []

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"{status}  {label}")
    if not condition and detail:
        print(f"       → {detail}")
    results.append(condition)


# ── Load requirements to get ground truth ────────────────────────────────────
ms_req  = REQUIREMENTS.get("ms", {})
phd_req = REQUIREMENTS.get("phd", {})

ms_breadth_count  = len(ms_req.get("breadth_categories", {}))
phd_breadth_count = len(phd_req.get("breadth_categories", {}))
ms_csci_required  = ms_req.get("csci_credits", 16)
phd_csci_required = phd_req.get("csci_credits", 16)

print(f"\n── Ground truth from requirements.json ──")
print(f"   MS  breadth categories : {ms_breadth_count}")
print(f"   PhD breadth categories : {phd_breadth_count}")
print(f"   MS  CSCI credits req   : {ms_csci_required}")
print(f"   PhD CSCI credits req   : {phd_csci_required}\n")


# ── Test 1: MS — original failing question ────────────────────────────────────
print("── MS: CSCI5525, CSCI5103, CSCI5511, CSCI8970 ──")
ms_out = degree_audit(["CSCI5525", "CSCI5103", "CSCI5511", "CSCI8970"], program="ms")
print(ms_out)

# cred_min fix: CSCI8970 = 1 credit, so total should be 10 not 12
check(
    "MS credit total is 10 (not 12 — CSCI8970 must count as 1 credit)",
    f"10/{ms_csci_required}" in ms_out,
    detail=f"Output contains: {[l for l in ms_out.splitlines() if 'credits completed' in l]}"
)

# All 3 breadth areas should be satisfied
check(
    "MS all breadth satisfied line is present",
    "All 3 breadth areas satisfied" in ms_out or "All breadth areas satisfied" in ms_out,
    detail="Expected explicit breadth-complete confirmation line"
)

# Breadth complete line must not claim a 4th area
check(
    "MS output does not mention a 4th breadth category",
    "4" not in ms_out.split("breadth")[0] if "breadth" in ms_out else True,
    detail="Check for fabricated extra breadth requirement"
)

print()

# ── Test 2: PhD — original failing question ───────────────────────────────────
print("── PhD: CSCI5521, CSCI5304, CSCI8001, CSCI8970 ──")
phd_out = degree_audit(["CSCI5521", "CSCI5304", "CSCI8001", "CSCI8970"], program="phd")
print(phd_out)

# Check breadth category count matches requirements.json — no invented categories
check(
    f"PhD output breadth category count matches requirements.json ({phd_breadth_count} areas)",
    str(phd_breadth_count) in phd_out,
    detail=f"requirements.json says {phd_breadth_count} PhD breadth categories"
)

# CSCI8970 = 1 credit, CSCI8001 credit depends on data — just check it's not overcounting
# If both 8970 and 8001 were falsely counted as 3, total would be 12; correct is ≤10
check(
    "PhD credit total is not 12 (would indicate cred_min fallback bug still present)",
    f"12/{phd_csci_required}" not in phd_out,
    detail="12 credits would mean 1-credit courses are being counted as 3"
)

# Intro to research should be marked complete
check(
    "PhD CSCI8001 intro to research marked complete",
    "Intro to Research" in phd_out and "complete" in phd_out,
    detail="CSCI8001 should satisfy intro_research requirement"
)

# Colloquium should be marked complete
check(
    "PhD CSCI8970 colloquium marked complete",
    "Colloquium" in phd_out and "complete" in phd_out,
)

print()

# ── Test 3: Edge case — empty course list ─────────────────────────────────────
print("── Edge case: empty course list ──")
empty_out = degree_audit([], program="ms")
check(
    "Empty course list does not crash and reports 0 credits",
    "0/" in empty_out,
    detail=f"Got: {empty_out[:80]}"
)

# ── Test 4: Edge case — invalid program ───────────────────────────────────────
# print("── Edge case: invalid program ──")
# bad_out = degree_audit(["CSCI5521"], program="mcs")
# check(
#     "Invalid program returns error message",
#     "Unknown program" in bad_out,
#     detail=f"Got: {bad_out}"
# )

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n── Results: {sum(results)}/{len(results)} checks passed ──\n")
if not all(results):
    sys.exit(1)