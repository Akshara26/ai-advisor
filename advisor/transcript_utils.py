"""
Transcript parsing utilities — extracted from app.py so they can be
imported and unit-tested without triggering Streamlit initialization.
"""
import io
import re
from pypdf import PdfReader

DEGREE_ELIGIBLE_GRADES = {
    "A", "A-",
    "B+", "B", "B-",
    "C+", "C", "C-",
    "S",
}


def parse_transcript(pdf_bytes: bytes) -> dict:
    """
    Parse a UMN unofficial transcript PDF.

    Returns:
    - completed degree-eligible course codes
    - richer course records containing code, earned credits, and grade
    - cumulative GPA

    Only includes courses with earned credits > 0 and a degree-eligible grade.
    Raw PDF bytes are processed in memory only and are not stored or logged.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    courses, course_records, gpa = [], [], None

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
            _, earned, grade = grade_match.groups()

            if float(earned) > 0 and grade in DEGREE_ELIGIBLE_GRADES:
                code = f"{dept}{num}"

                courses.append(code)
                course_records.append({
                    "code": code,
                    "credits": float(earned),
                    "grade": grade,
                })

    return {"courses": courses, "course_records": course_records, "gpa": gpa}