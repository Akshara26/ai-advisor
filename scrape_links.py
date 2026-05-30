import requests
from bs4 import BeautifulSoup
import time
import json

# Only scrape pages with actual policy/procedure content
URLS_TO_SCRAPE = [
    "https://policy.umn.edu/education",
    "https://onestop.umn.edu/academics/degree-completion-steps",
    "https://onestop.umn.edu/academics/grad-and-professional/degree-completion-steps",
    "https://onestop.umn.edu/academics/examination-committees",
    "https://onestop.umn.edu/academics/submit-your-gpas-approval",
    "https://onestop.umn.edu/academics/registration-times",
    "https://onestop.umn.edu/academics/doctoral-oral-exam-scheduling",
    "https://grad.umn.edu/funding",
    "https://grad.umn.edu/funding/current-students/apply-through-program/doctoral-dissertation-fellowship",
    "https://isss.umn.edu/fstudents/employment/cpt",
    "https://isss.umn.edu/fstudents/employment/opt",
    "https://cei.umn.edu/services-graduate-students-and-postdocs",
    "https://hr.umn.edu/Jobs/Student-Job-Center/Graduate-Assistant-Employment/About-Graduate-Assistant-Employment",
    "https://cse.umn.edu/cs/graduate-cpt",
    "https://www.cs.umn.edu/academics/graduate/phd/transfer-credits",
    "https://cse.umn.edu/cs/ms-overview#ProjectCoursework",
    "https://csgrad.appointments.umn.edu",
    "https://faculty-roles.umn.edu/",
    "https://cse.umn.edu/cs/faculty-instructor-directory",
    "http://www.cs.umn.edu",
    "http://www.unite.umn.edu/",
    "https://cse.umn.edu/cs/integrated-program-bachelorsmasters",
    "https://cse.umn.edu/r/career-center/",
    "https://grad.umn.edu/funding/current-students/apply-through-program/doctoral-dissertation-fellowship",
    "https://grad.umn.edu/admissions/readmission",
    "https://cse.umn.edu/cseit/classrooms-labs",
    "https://cseit.umn.edu/"
]

def scrape_page(url: str) -> dict:
    """Scrape a page and return clean text content."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; UMN CS Advisor research bot)"}
        response = requests.get(url, timeout=10, headers=headers)
        
        if response.status_code != 200:
            print(f"  SKIP {response.status_code}: {url}")
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove nav, footer, scripts, styles
        for tag in soup(["nav", "footer", "script", "style", "header", 
                         "aside", "iframe", "noscript"]):
            tag.decompose()

        # Get page title
        title = soup.find("title")
        title_text = title.get_text(strip=True) if title else url

        # Get main content — try common content containers first
        main = (
            soup.find("main") or
            soup.find(id="main-content") or
            soup.find(class_="field-items") or
            soup.find(class_="content") or
            soup.find("article") or
            soup.body
        )

        if not main:
            return None

        text = main.get_text(separator="\n", strip=True)

        # Clean up excessive whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        # Skip pages with too little content
        if len(clean_text) < 200:
            print(f"  SKIP (too short): {url}")
            return None

        print(f"  OK ({len(clean_text)} chars): {url}")
        return {
            "url": url,
            "title": title_text,
            "content": clean_text
        }

    except Exception as e:
        print(f"  ERROR: {url} — {e}")
        return None

print("Scraping UMN pages...")
pages = []

for url in URLS_TO_SCRAPE:
    result = scrape_page(url)
    if result:
        pages.append(result)
    time.sleep(1)  # be polite

print(f"\nSuccessfully scraped {len(pages)} pages")

# Save to file
with open("data/scraped_pages.json", "w") as f:
    json.dump(pages, f, indent=2)

print("Saved to data/scraped_pages.json")