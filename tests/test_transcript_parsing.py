"""
Unit tests for transcript_utils.parse_transcript()

Run with: pytest tests/test_transcript_parsing.py -v
"""
import pytest
from unittest.mock import MagicMock, patch
from transcript_utils import parse_transcript


def _make_pdf(text: str) -> bytes:
    """
    Return fake PDF bytes backed by a mock PdfReader that returns `text`.
    We mock at the PdfReader level so the real pypdf is bypassed.
    """
    return text.encode()  # content doesn't matter — PdfReader is mocked


def _patch_reader(text: str):
    """Context manager: patch PdfReader so it returns `text` from page 0."""
    mock_page = MagicMock()
    mock_page.extract_text.return_value = text
    mock_reader = MagicMock()
    mock_reader.pages = [mock_page]
    return patch("transcript_utils.PdfReader", return_value=mock_reader)


# ── Typical transcript format ─────────────────────────────────────────────────

TYPICAL_TRANSCRIPT = """
Fall Semester 2023
CSCI 5523 Introduction to Data Mining 3.00 3.00 B+ 9.999
CSCI 5707 Principles of DB Systems 3.00 3.00 B+ 9.999
STAT 5302 Applied Regression Analysis 4.00 4.00 C 8.000
TERM GPA : 2.800 TERM TOTALS : 10.00 10.00 10.00 27.998

Spring Semester 2024
CSCI 5561 Computer Vision 3.00 3.00 A 12.000
TERM GPA : 4.000 TERM TOTALS : 3.00 3.00 3.00 12.000

CUM GPA: 3.518 UM TOTALS: 13.00 13.00 13.00 39.998
"""


class TestTypicalTranscript:

    def test_csci_courses_extracted(self):
        with _patch_reader(TYPICAL_TRANSCRIPT):
            result = parse_transcript(b"fake")
        assert "CSCI5523" in result["courses"]
        assert "CSCI5707" in result["courses"]
        assert "CSCI5561" in result["courses"]

    def test_non_csci_courses_included(self):
        # parse_transcript returns ALL completed courses, not just CSCI
        with _patch_reader(TYPICAL_TRANSCRIPT):
            result = parse_transcript(b"fake")
        assert "STAT5302" in result["courses"]

    def test_gpa_extracted(self):
        with _patch_reader(TYPICAL_TRANSCRIPT):
            result = parse_transcript(b"fake")
        assert result["gpa"] == "3.518"

    def test_course_codes_have_no_spaces(self):
        with _patch_reader(TYPICAL_TRANSCRIPT):
            result = parse_transcript(b"fake")
        for code in result["courses"]:
            assert " " not in code, f"Course code '{code}' contains a space"


# ── In-progress courses must be excluded ─────────────────────────────────────

IN_PROGRESS_TRANSCRIPT = """
Spring Semester 2025
CSCI 8980 Special Advanced Topics CS 3.00 3.00 A 12.000
DSCI 8760 Data Science MS Plan B Project 3.00 0.00 0.000
PE 1037 Squash 1.00 0.00 0.000
TERM GPA : 0.000 TERM TOTALS : 7.00 3.00 3.00 12.000
"""


class TestInProgressFiltering:

    def test_zero_earned_courses_excluded(self):
        with _patch_reader(IN_PROGRESS_TRANSCRIPT):
            result = parse_transcript(b"fake")
        assert "DSCI8760" not in result["courses"]
        assert "PE1037" not in result["courses"]

    def test_completed_courses_included(self):
        with _patch_reader(IN_PROGRESS_TRANSCRIPT):
            result = parse_transcript(b"fake")
        assert "CSCI8980" in result["courses"]


# ── No CSCI courses ───────────────────────────────────────────────────────────

NO_CSCI_TRANSCRIPT = """
Fall Semester 2023
STAT 5302 Applied Regression Analysis 4.00 4.00 A 16.000
MATH 5652 Introduction to Probability 4.00 4.00 B+ 13.332
CUM GPA: 3.750 UM TOTALS: 8.00 8.00 8.00 29.332
"""


class TestNoCSCICourses:

    def test_returns_empty_csci_list(self):
        with _patch_reader(NO_CSCI_TRANSCRIPT):
            result = parse_transcript(b"fake")
        csci = [c for c in result["courses"] if c.startswith("CSCI")]
        assert len(csci) == 0

    def test_gpa_still_extracted(self):
        with _patch_reader(NO_CSCI_TRANSCRIPT):
            result = parse_transcript(b"fake")
        assert result["gpa"] == "3.750"

    def test_non_csci_courses_present(self):
        with _patch_reader(NO_CSCI_TRANSCRIPT):
            result = parse_transcript(b"fake")
        assert len(result["courses"]) > 0


# ── Missing GPA ───────────────────────────────────────────────────────────────

NO_GPA_TRANSCRIPT = """
Fall Semester 2023
CSCI 5521 Machine Learning I 3.00 3.00 A 12.000
TERM GPA : 4.000 TERM TOTALS : 3.00 3.00 3.00 12.000
"""


class TestMissingGPA:

    def test_gpa_is_none_when_missing(self):
        with _patch_reader(NO_GPA_TRANSCRIPT):
            result = parse_transcript(b"fake")
        assert result["gpa"] is None

    def test_courses_still_extracted_without_gpa(self):
        with _patch_reader(NO_GPA_TRANSCRIPT):
            result = parse_transcript(b"fake")
        assert "CSCI5521" in result["courses"]


# ── Empty / blank transcript ──────────────────────────────────────────────────

class TestEmptyTranscript:

    def test_empty_text_returns_empty_lists(self):
        with _patch_reader(""):
            result = parse_transcript(b"fake")
        assert result["courses"] == []
        assert result["gpa"] is None

    def test_only_headers_returns_empty(self):
        with _patch_reader("University of Minnesota Unofficial Transcript\nName: Student"):
            result = parse_transcript(b"fake")
        assert result["courses"] == []


# ── S/U graded courses ────────────────────────────────────────────────────────

SU_TRANSCRIPT = """
Fall Semester 2023
CSCI 8970 Data Science M.S. Colloquium 1.00 1.00 S 0.000
CSCI 5521 Machine Learning I 3.00 3.00 A 12.000
"""


class TestSUGradedCourses:

    def test_s_grade_course_included(self):
        # S (satisfactory) is a valid completed grade
        with _patch_reader(SU_TRANSCRIPT):
            result = parse_transcript(b"fake")
        assert "CSCI8970" in result["courses"]
        assert "CSCI5521" in result["courses"]
