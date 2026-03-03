import gspread
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import html
import re

load_dotenv()

def clean_html(text):
    text = html.unescape(text)           # converts &lt; back to <
    text = re.sub(r'<[^>]+>', ' ', text) # removes <div>, <p> tags
    text = re.sub(r'\s+', ' ', text)     # collapses extra spaces
    return text.strip()

# ── 1. Load jobs from Google Sheets ──────────────────────────────────
print("📋 Loading jobs from Google Sheets...")
gc = gspread.service_account(filename='credentials.json')
sh = gc.open("My_Job_Copilot_DB")
worksheet = sh.sheet1
df = pd.DataFrame(worksheet.get_all_records())
print(df.columns.tolist())
print(f"   Found {len(df)} rows")

# Drop rows with empty descriptions
df = df[df["Description"].str.strip() != ""]
print(f"   {len(df)} rows with descriptions after cleaning")

# ____Cleaning Data

df["Description"] = df["Description"].apply(clean_html)

# ── 2. Convert rows to LangChain Documents ────────────────────────────
print("\n📄 Converting to Documents...")
docs = []
for _, row in df.iterrows():
    # Combining key fields into one rich text chunk
    content = f"""
    Company: {row['Company']}
    Title: {row['Title']}
    Location: {row['Location']}
    Description: {row['Description']}
    """.strip()

    # Metadata for filtering
    metadata = {
        "company": row["Company"],
        "title": row["Title"],
        "location": row["Location"],
        "url": row["URL"],
        "date_added": row["Date_Added"],
        "source": row["Source"]
    }
    docs.append(Document(page_content=content, metadata=metadata))

print(f"   Created {len(docs)} documents")

# ── 3. Embedding + Storing in ChromaDB ──────────────────────────────────────
print("\n🔢 Embedding and storing in ChromaDB...")
print("   (This takes 2-5 mins for 62 jobs — Mistral running locally)")

embeddings = OllamaEmbeddings(model="mistral")

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db",   # saved locally
    collection_name="job_postings"
)

print(f"\n Vector store built! {len(docs)} jobs indexed in ./chroma_db")

# ── 4. Quick sanity test ──────────────────────────────────────────────
print("\n Testing retrieval...")
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
results = retriever.invoke("causal inference machine learning scientist")

print(f"   Top 3 matches for 'causal inference machine learning scientist':")
for i, doc in enumerate(results):
    print(f"   {i+1}. {doc.metadata['company']} — {doc.metadata['title']} ({doc.metadata['location']})")

print("\n Done!")
