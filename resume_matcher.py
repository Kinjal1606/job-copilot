import pdfplumber
import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import re

load_dotenv()

# 1. Extract resume text
print("Loading resume...")
with pdfplumber.open("personal_docs/Kinjal_Saxena Resume.pdf") as pdf:
    resume_text = "\n".join(
        page.extract_text() for page in pdf.pages if page.extract_text()
    )
print(f"   Extracted {len(resume_text)} characters")

# 2. Load all jobs from ChromaDB
print("\nLoading all jobs from ChromaDB...")
embeddings = OllamaEmbeddings(model="mistral")
job_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="job_postings"
)

# Pull ALL jobs (not just top-k) for full scoring
all_jobs_raw = job_store._collection.get()
total = len(all_jobs_raw["ids"])
print(f"   Found {total} jobs to score")

# Reconstruct as list of dicts
all_jobs = []
for i in range(total):
    all_jobs.append({
        "content":  all_jobs_raw["documents"][i],
        "metadata": all_jobs_raw["metadatas"][i]
    })

# 3. Load Mistral
llm = OllamaLLM(model="mistral", temperature=0.1)

# 4. Scoring prompt
score_prompt = PromptTemplate.from_template("""
You are a recruiter evaluating job fit for a senior data scientist.

Candidate resume:
{resume}

Job posting:
{job}

Evaluate fit. Respond in EXACTLY this format — no extra text:
Score: [0-100]
Match: [comma separated matching skills]
Gaps: [comma separated missing keywords from job not in resume]
Verdict: [one sentence]
LinkedIn: Hi [Hiring Manager], I came across the [Job Title] role at [Company]. My experience in [top 2 matching skills] aligns well with [one specific job requirement]. I'd love to connect. Best, Kinjal
""")

score_chain = score_prompt | llm | StrOutputParser()

# 5. Score every job
print("\n Scoring all jobs (this will take a few minutes)...\n")
results = []

for i, job in enumerate(all_jobs, 1):
    meta    = job["metadata"]
    title   = meta.get("title",    "Unknown")
    company = meta.get("company",  "Unknown")
    location= meta.get("location", "Unknown")
    url     = meta.get("url",      "")
    date    = meta.get("date_added","")

    print(f"   [{i}/{total}] Scoring: {title} at {company}...")

    output = score_chain.invoke({
        "resume": resume_text[:3000],
        "job":    job["content"][:2000]
    })

    # Parse fields
    def extract(label, text):
        try:
            return text.split(f"{label}:")[1].split("\n")[0].strip()
        except:
            return "N/A"

    score_str = extract("Score", output)
    try:
        score = int(re.sub(r"[^0-9]", "", score_str))
    except:
        score = 0

    results.append({
        "Score":     score,
        "Title":     title,
        "Company":   company,
        "Location":  location,
        "Date":      date,
        "URL":       url,
        "Match":     extract("Match",    output),
        "Gaps":      extract("Gaps",     output),
        "Verdict":   extract("Verdict",  output),
        "LinkedIn":  extract("LinkedIn", output)
    })

# 6. Sort by score
df = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)

# Save full results to CSV
df.to_csv("job_scores.csv", index=False)
print(f"\nSaved all {len(df)} scored jobs to job_scores.csv")

# 7. Top matches (≥75)
top_df = df[df["Score"] >= 75]
print(f"\n{len(top_df)} jobs scored ≥75/100\n")
print(df[["Score","Title","Company","Location","Gaps"]].to_string(index=False))

# 8. Email top matches
print("\nSending email with top matches...")

def build_email_html(top_df):
    rows = ""
    for _, row in top_df.iterrows():
        rows += f"""
        <tr>
            <td><b>{row['Score']}/100</b></td>
            <td><a href="{row['URL']}">{row['Title']}</a></td>
            <td>{row['Company']}</td>
            <td>{row['Location']}</td>
            <td>{row['Match']}</td>
            <td style="color:red">{row['Gaps']}</td>
            <td>{row['Verdict']}</td>
            <td><a href="{row['URL']}">Apply</a></td>

        </tr>
        <tr>
            <td colspan="8" style="background:#f9f9f9;padding:8px">
                <i>LinkedIn: {row['LinkedIn']}</i>
            </td>
        </tr>
        """
    return f"""
    <html><body>
    <h2>Top Job Matches Today</h2>
    <table border="1" cellpadding="8" cellspacing="0" style="border-collapse:collapse;width:100%">
        <tr style="background:#4CAF50;color:white">
            <th>Score</th><th>Title</th><th>Company</th>
            <th>Location</th><th>Matches</th><th>Gaps </th><th>Verdict</th><th>Apply</th>
        </tr>
        {rows}
    </table>
    <br><small>Personal research tool only. Not career advice.</small>
    </body></html>
    """

msg = MIMEMultipart("alternative")
msg["Subject"] = f"Job Copilot: {len(top_df)} Top Matches Today"
msg["From"]    = os.getenv("EMAIL_SENDER")
msg["To"]      = os.getenv("EMAIL_RECEIVER")
msg.attach(MIMEText(build_email_html(top_df), "html"))

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
    server.login(os.getenv("EMAIL_SENDER"), os.getenv("EMAIL_APP_PASSWORD"))
    server.sendmail(os.getenv("EMAIL_SENDER"), os.getenv("EMAIL_RECEIVER"), msg.as_string())

print(f" Email sent with {len(top_df)} top matches!")
print("\n All done!")
