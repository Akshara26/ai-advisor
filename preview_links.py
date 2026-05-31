"""
Preview high-value URLs from the categorized CSV before ingestion.
Also checks live HTTP status so you know what's actually scrapeable.

Usage:
    python preview_links.py                    # show filtered URLs, no HTTP check
    python preview_links.py --check            # also check HTTP status for each URL
    python preview_links.py --check --category immigration/status
"""

import argparse
import csv
import time
from collections import Counter

import requests

CSV_PATH = "umn_advisor_links.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; UMN CS Advisor research bot)"}

HIGH_VALUE_CATEGORIES = {
    "degree requirements",
    "policy",
    "forms",
    "funding",
    "immigration/status",
    "assistantships",
    "exams & committees",
    "academic calendar",
    "student support",
    "career",
}


def check_url(url: str) -> tuple[int, str]:
    try:
        r = requests.get(url, timeout=8, headers=HEADERS, allow_redirects=True)
        ct = r.headers.get("content-type", "")
        kind = "html" if "text/html" in ct else ("pdf" if "pdf" in ct else ct[:20])
        return r.status_code, kind
    except Exception as e:
        return 0, str(e)[:30]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Check HTTP status for each URL")
    parser.add_argument("--category", default=None, help="Filter to a single category")
    args = parser.parse_args()

    with open(CSV_PATH, newline="") as f:
        all_rows = list(csv.DictReader(f))

    rows = [
        r for r in all_rows
        if r["category"] in HIGH_VALUE_CATEGORIES
        and (args.category is None or r["category"] == args.category)
    ]

    print(f"\n{len(rows)} high-value URLs")
    if args.category:
        print(f"Category filter: {args.category}")

    # Category summary
    cats = Counter(r["category"] for r in rows)
    print("\nBy category:")
    for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {n:3d}  {cat}")

    print()

    scrapeable, blocked, errors = [], [], []

    for r in rows:
        url = r["url"]
        cat = r["category"]

        if args.check:
            status, kind = check_url(url)
            time.sleep(0.3)

            if status == 200 and kind == "html":
                symbol = "✅"
                scrapeable.append(url)
            elif status == 200 and kind == "pdf":
                symbol = "📄"
                blocked.append(url)
            elif status in (401, 403):
                symbol = "🔒"
                blocked.append(url)
            elif status == 404:
                symbol = "💀"
                errors.append(url)
            else:
                symbol = f"⚠ {status}"
                errors.append(url)

            print(f"{symbol} [{cat}] {url}")
        else:
            print(f"  [{cat}] {url}")

    if args.check:
        print(f"\n✅ Scrapeable:  {len(scrapeable)}")
        print(f"🔒 Blocked/PDF: {len(blocked)}")
        print(f"💀 Dead/error:  {len(errors)}")

        if blocked:
            print("\nBlocked URLs (candidates for reference stubs):")
            for u in blocked:
                print(f"  {u}")


if __name__ == "__main__":
    main()
