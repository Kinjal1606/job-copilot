import requests
import gspread
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

# ── Google Sheets Setup ──────────────────────────────────────────────
gc = gspread.service_account(filename='credentials.json')
sh = gc.open("My_Job_Copilot_DB")
worksheet = sh.sheet1

# Add headers if sheet is empty
if not worksheet.get_all_values():
    worksheet.append_row(["Company","Title","Location","Description","URL","Source","Date_Added"])

# ── Helpers ───────────────────────────────────────────────────────────
DS_KEYWORDS = ["data scientist","applied scientist","machine learning","ml engineer",
               "ai engineer","data science","research scientist","forward deployed"]

def is_relevant(title: str) -> bool:
    return any(kw in title.lower() for kw in DS_KEYWORDS)

def already_in_sheet(url: str, existing_urls: list) -> bool:
    return url in existing_urls

# ── Source 1: Greenhouse (Anthropic, Uber, Lyft) ──────────────────────
def fetch_greenhouse(company_token: str) -> list:
    url = f"https://boards-api.greenhouse.io/v1/boards/{company_token}/jobs?content=true"
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        print(f"  ⚠️  Greenhouse failed for {company_token}")
        return []
    jobs = []
    for job in resp.json().get("jobs", []):
        if is_relevant(job.get("title", "")):
            jobs.append({
                "Company": company_token.capitalize(),
                "Title": job.get("title"),
                "Location": job.get("location", {}).get("name", "N/A"),
                "Description": job.get("content", "")[:2000],  # cap at 2000 chars
                "URL": job.get("absolute_url"),
                "Source": "Greenhouse",
                "Date_Added": datetime.now().strftime("%Y-%m-%d")
            })
    return jobs

# ── Source 2: JSearch API (Meta, Microsoft, Apple, Netflix, AMD) ───────
# Sign up free at rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch
JSEARCH_KEY = os.getenv("JSEARCH_API_KEY")

def fetch_jsearch(company: str) -> list:
    if not JSEARCH_KEY:
        print("  ⚠️  No JSEARCH_API_KEY in .env — skipping JSearch")
        return []
    url = "https://jsearch.p.rapidapi.com/search"
    params = {"query": f"data scientist at {company}", "num_pages": "1", "date_posted": "today"}
    headers = {"X-RapidAPI-Key": JSEARCH_KEY, "X-RapidAPI-Host": "jsearch.p.rapidapi.com"}
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    if resp.status_code != 200:
        print(f"  ⚠️  JSearch failed for {company}")
        return []
    jobs = []
    for job in resp.json().get("data", []):
        if is_relevant(job.get("job_title", "")):
            jobs.append({
                "Company": company,
                "Title": job.get("job_title"),
                "Location": f"{job.get('job_city','')}, {job.get('job_country','')}",
                "Description": job.get("job_description", "")[:2000],
                "URL": job.get("job_apply_link"),
                "Source": "JSearch",
                "Date_Added": datetime.now().strftime("%Y-%m-%d")
            })
    return jobs

# ── Main ──────────────────────────────────────────────────────────────
greenhouse_companies = ["anthropic", "uberatg", "lyft"]
jsearch_companies    = ["Meta", "Microsoft", "Apple", "Netflix", "AMD"]

all_jobs = []
for c in greenhouse_companies:
    print(f"Fetching Greenhouse: {c}")
    all_jobs.extend(fetch_greenhouse(c))

for c in jsearch_companies:
    print(f"Fetching JSearch: {c}")
    all_jobs.extend(fetch_jsearch(c))

# ── Dedup against existing sheet ──────────────────────────────────────
existing = worksheet.get_all_records()
existing_urls = [r.get("URL") for r in existing]
new_jobs = [j for j in all_jobs if not already_in_sheet(j["URL"], existing_urls)]

# ── Push new jobs to Sheet ────────────────────────────────────────────
if new_jobs:
    rows = [[j["Company"],j["Title"],j["Location"],j["Description"],
             j["URL"],j["Source"],j["Date_Added"]] for j in new_jobs]
    worksheet.append_rows(rows)
    print(f"\n✅ Added {len(new_jobs)} new DS roles to Google Sheets!")
else:
    print("\n✅ No new jobs today — sheet is up to date.")
