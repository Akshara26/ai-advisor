"""
Transcript parsing utilities — extracted from app.py so they can be
imported and unit-tested without triggering Streamlit initialization.
"""
import io
import re
from pypdf import PdfReader


def parse_transcript(pdf_bytes: bytes) -> dict:
    """
    Parse a UMN unofficial transcript PDF.
    Returns completed course codes (e.g. CSCI5523) and cumulative GPA.
    Only includes courses with earned credits > 0 and a valid grade.
    Does NOT store or log any PII — raw bytes are processed in memory only.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    courses, gpa = [], None

    for line in text.split("\n"):
        line = line.strip()
        if not line or "TERM GPA" in line or "TERM TOTALS" in line:
            continue

        cum_match = re.search(r"CUM GPA:\s*([\d.]+)", line)
        if cum_match:
            gpa = cum_match.group(1)

        dept_match = re.match(r"^([A-Z]{2,5})\s+(\d{4})\b", line)
        if not dept_match:
            continue

        dept, num = dept_match.group(1), dept_match.group(2)

        # Completed courses: earned credits > 0 and a valid letter/S/U grade
        grade_match = re.search(
            r"(\d+\.\d+)\s+(\d+\.\d+)\s+([A-Z][+-]?|S|U)\s+\d+\.\d+\s*$",
            line
        )
        if grade_match:
            _, earned, _ = grade_match.groups()
            if float(earned) > 0:
                courses.append(f"{dept}{num}")

    return {"courses": courses, "gpa": gpa}