import sqlite3
import json

conn = sqlite3.connect("data/grades.db")

def get_grade_distribution(course_code: str) -> str:
    """Get historical grade distribution for a UMN course."""
    # Parse input like "CSCI5521" into dept="CSCI" and num="5521"
    course_code = course_code.upper().replace(" ", "")
    dept = ''.join(filter(str.isalpha, course_code))
    num = ''.join(filter(str.isdigit, course_code))

    cursor = conn.cursor()
    cursor.execute("""
        SELECT dept_abbr, course_num, class_desc, total_students, total_grades
        FROM classdistribution
        WHERE dept_abbr=? AND course_num=?
    """, (dept, num))

    row = cursor.fetchone()
    if not row:
        return f"No grade data found for {course_code}."

    dept_abbr, course_num, desc, total, grades_json = row
    grades = json.loads(grades_json)

    # Calculate GPA
    grade_points = {"A": 4.0, "A-": 3.67, "B+": 3.33, "B": 3.0,
                    "B-": 2.67, "C+": 2.33, "C": 2.0, "C-": 1.67,
                    "D+": 1.33, "D": 1.0, "F": 0.0}
    
    total_points = sum(grades.get(g, 0) * pts for g, pts in grade_points.items())
    graded_students = sum(grades.get(g, 0) for g in grade_points)
    avg_gpa = round(total_points / graded_students, 2) if graded_students > 0 else 0

    # Calculate A/B/C/D/F percentages
    def pct(keys):
        count = sum(grades.get(k, 0) for k in keys)
        return round(count / total * 100, 1) if total > 0 else 0

    result = f"{dept_abbr} {course_num} - {desc}\n"
    result += f"Total students (all time): {total:,}\n"
    result += f"Average GPA: {avg_gpa}\n"
    result += f"A/A-: {pct(['A','A-'])}%  "
    result += f"B range: {pct(['B+','B','B-'])}%  "
    result += f"C range: {pct(['C+','C','C-'])}%  "
    result += f"D/F: {pct(['D+','D','F'])}%  "
    result += f"Withdrew: {pct(['W'])}%"
    return result

if __name__ == "__main__":
    print(get_grade_distribution("CSCI2011"))
    print()
    print(get_grade_distribution("CSCI5521"))