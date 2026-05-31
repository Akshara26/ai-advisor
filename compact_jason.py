import json
from pathlib import Path

input_file = Path("data/courses.json")
output_file = Path(r"/Users/aksharamadhu/Downloads/courses.min.json")

with input_file.open("r", encoding="utf-8") as f:
    data = json.load(f)

smaller = {}
for k, v in data.items():
    smaller[k] = {
        "uid": v.get("uid"),
        "code": v.get("code"),
        "subject": v.get("subject"),
        "number": v.get("number"),
        "name": v.get("name"),
        "prereq": v.get("prereq")
    }

with output_file.open("w", encoding="utf-8") as f:
    json.dump(smaller, f, separators=(",", ":"), ensure_ascii=False)